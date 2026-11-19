use crate::error::RuntimeError;
use crate::heap::{HeapObject, HeapRef, ObjectKey};
use num_bigint::BigInt;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

pub(crate) const MAX_RUNTIME_TYPE_DEPTH: usize = 64;
pub(crate) const MAX_RUNTIME_TYPE_NODES: usize = 1_000_000;
pub(crate) const MAX_RUNTIME_TYPE_NAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct RuntimeTypeShape {
    depth: usize,
    nodes: usize,
    name_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
enum RuntimeTypeLimit {
    Depth,
    Nodes,
    NameBytes,
}

impl RuntimeTypeLimit {
    fn into_error(self) -> RuntimeError {
        RuntimeError::ResourceLimit {
            resource: match self {
                Self::Depth => "array type depth",
                Self::Nodes => "array type nodes",
                Self::NameBytes => "array type names",
            },
        }
    }
}

/// Runtime identity carried by each shared array object. The descriptor is
/// derived from values when an array is created and is fixed by its first
/// insertion when an empty array is used without an explicit runtime hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeType {
    Int,
    UInt,
    BigInt,
    Float,
    Bool,
    Char,
    Nil,
    Str,
    Error(String),
    Array(Box<RuntimeType>),
    Record,
    Variant(String),
    Function,
    Other,
    Nominal(String),
    Nullable(Box<RuntimeType>),
    Tuple(Vec<RuntimeType>),
}

/// Orders an exact signed integer against an f64 without first rounding the
/// integer into floating-point. `None` represents NaN, like `partial_cmp`.
pub(crate) fn compare_int_float(integer: i64, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float >= 9223372036854775808.0 {
        return Some(Ordering::Less);
    }
    if float < i64::MIN as f64 {
        return Some(Ordering::Greater);
    }
    let truncated = float as i64;
    match integer.cmp(&truncated) {
        Ordering::Equal => (truncated as f64).partial_cmp(&float),
        order => Some(order),
    }
}

pub(crate) fn compare_int_uint(integer: i64, unsigned: u64) -> Ordering {
    if integer < 0 {
        Ordering::Less
    } else {
        (integer as u64).cmp(&unsigned)
    }
}

pub(crate) fn compare_uint_float(unsigned: u64, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float < 0.0 {
        return Some(Ordering::Greater);
    }
    if float >= 18446744073709551616.0 {
        return Some(Ordering::Less);
    }
    let truncated = float as u64;
    match unsigned.cmp(&truncated) {
        Ordering::Equal => (truncated as f64).partial_cmp(&float),
        order => Some(order),
    }
}

pub(crate) fn compare_bigint_float(integer: &BigInt, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }
    let bits = float.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1_u64 << 52), exponent_bits - 1023 - 52)
    };
    let mut significand = BigInt::from(significand);
    if negative {
        significand = -significand;
    }
    if exponent >= 0 {
        Some(integer.cmp(&(significand << exponent as usize)))
    } else {
        Some((integer << (-exponent) as usize).cmp(&significand))
    }
}

pub(crate) fn bigint_exceeds_decimal_digits(value: &BigInt, limit: usize) -> bool {
    let bits = value.bits();
    if bits == 0 {
        return limit == 0;
    }
    let limit = u64::try_from(limit).unwrap_or(u64::MAX);
    let lower_digits = bits.saturating_sub(1).saturating_mul(30_102) / 100_000 + 1;
    if lower_digits > limit {
        return true;
    }
    let upper_digits = bits.saturating_mul(30_103).saturating_add(99_999) / 100_000;
    if upper_digits <= limit {
        return false;
    }
    value.to_str_radix(10).trim_start_matches('-').len() > limit as usize
}

/// Shared array storage with checked element and tuple contracts.
///
/// Read elements through [`Self::borrow`] or [`Self::try_borrow`], and mutate
/// through the checked methods so type metadata and iterator versions stay valid.
/// The backing cell is deliberately unavailable to callers:
///
/// ```compile_fail
/// use cog::value::{ArrayValue, Value};
/// use std::cell::RefCell;
///
/// let array = ArrayValue::new(vec![Value::Int(1)]).unwrap();
/// let unchecked: &RefCell<Vec<Value>> = &array;
/// unchecked.borrow_mut()[0] = Value::Str("wrong type".into());
/// ```
#[derive(Debug)]
pub struct ArrayValue {
    values: RefCell<Vec<Value>>,
    element_type: RefCell<Option<RuntimeType>>,
    tuple_types: RefCell<Option<Vec<RuntimeType>>>,
    tuple_contract_fixed: Cell<bool>,
    /// Arrays decoded from dynamic formats such as JSON do not have a single
    /// element type. They retain normal mutable array operations while staying
    /// distinct from source arrays whose empty type has not yet been inferred.
    dynamic: Cell<bool>,
    /// Bounds descriptor copies before `runtime_type()` clones them.
    runtime_type_shape: Cell<RuntimeTypeShape>,
    structure_version: Cell<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ArrayTypeError;

/// Tracks the contracts that a whole checked operation will apply to shared
/// arrays. Independent tuple fields can refer to the same empty array.
#[derive(Default)]
struct TypeValidation<'a> {
    arrays: HashMap<*const ArrayValue, &'a RuntimeType>,
}

fn array_type_error() -> RuntimeError {
    RuntimeError::TypeMismatch { op: "array_type" }
}

fn type_error_for(error: RuntimeError, op: &'static str) -> RuntimeError {
    match error {
        RuntimeError::TypeMismatch { .. } => RuntimeError::TypeMismatch { op },
        error => error,
    }
}

/// Compiler-created state for fail-fast array/string traversal. It is a heap
/// object so the source collection remains rooted for the loop's lifetime.
pub struct IteratorValue {
    source: Value,
    index: usize,
    byte_offset: usize,
    expected_structure_version: Option<u64>,
    current: Option<Value>,
}

impl IteratorValue {
    pub(crate) fn new(source: Value) -> Result<Self, RuntimeError> {
        let expected_structure_version = match &source {
            Value::Array(array) => Some(array.structure_version()),
            Value::Str(_) => None,
            _ => {
                return Err(RuntimeError::TypeMismatch {
                    op: "for iteration",
                });
            }
        };
        Ok(Self {
            source,
            index: 0,
            byte_offset: 0,
            expected_structure_version,
            current: None,
        })
    }

    pub(crate) fn next(&mut self) -> Result<bool, RuntimeError> {
        self.current = None;
        match self.source.clone() {
            Value::Array(array) => {
                if Some(array.structure_version()) != self.expected_structure_version {
                    return Err(RuntimeError::InvalidArgument {
                        op: "array iteration invalidated",
                    });
                }
                let values =
                    array
                        .values
                        .try_borrow()
                        .map_err(|_| RuntimeError::InvalidArgument {
                            op: "array iteration",
                        })?;
                let Some(value) = values.get(self.index).cloned() else {
                    return Ok(false);
                };
                self.index += 1;
                self.current = Some(value);
                Ok(true)
            }
            Value::Str(text) => {
                let Some(rest) = text.get(self.byte_offset..) else {
                    return Err(RuntimeError::InvalidArgument {
                        op: "string iteration",
                    });
                };
                let Some(character) = rest.chars().next() else {
                    return Ok(false);
                };
                self.byte_offset += character.len_utf8();
                self.current = Some(Value::Char(character));
                self.index += 1;
                Ok(true)
            }
            _ => Err(RuntimeError::TypeMismatch {
                op: "for iteration",
            }),
        }
    }

    pub(crate) fn current(&self) -> Result<Value, RuntimeError> {
        self.current.clone().ok_or(RuntimeError::InvalidArgument {
            op: "iterator has no current value",
        })
    }
}

impl ArrayValue {
    /// Borrow the array's elements for reading.
    ///
    /// Checked mutations fail while this shared borrow is held.
    pub fn borrow(&self) -> std::cell::Ref<'_, Vec<Value>> {
        self.values.borrow()
    }

    /// Try to borrow the array's elements for reading without panicking.
    pub fn try_borrow(&self) -> Result<std::cell::Ref<'_, Vec<Value>>, std::cell::BorrowError> {
        self.values.try_borrow()
    }

    pub fn new(values: Vec<Value>) -> Result<Self, ArrayTypeError> {
        Self::try_new(values).map_err(|_| ArrayTypeError)
    }

    pub(crate) fn try_new(values: Vec<Value>) -> Result<Self, RuntimeError> {
        for value in &values {
            ensure_array_child_type_shape(value)?;
        }
        // Error is orthogonal to the normal element type. Infer an array's
        // element contract from a normal value even when an Error appears first.
        let element_type = values
            .iter()
            .filter(|value| !matches!(value, Value::Error(_)))
            .map(RuntimeType::of)
            .find(|ty| !has_undetermined_array_type(ty))
            .or_else(|| {
                values
                    .iter()
                    .map(RuntimeType::of)
                    .find(|ty| !has_undetermined_array_type(ty))
            });
        if !values.is_empty() && element_type.is_none() {
            return Err(RuntimeError::TypeMismatch { op: "array" });
        }
        if let Some(expected) = &element_type {
            let mut validation = TypeValidation::default();
            for value in &values {
                validate_value_type(expected, value, &mut validation)
                    .map_err(|error| type_error_for(error, "array"))?;
            }
            for value in &values {
                apply_value_type(expected, value)
                    .map_err(|error| type_error_for(error, "array"))?;
            }
        }
        let runtime_type_shape = array_runtime_type_shape(element_type.as_ref(), None)
            .map_err(RuntimeTypeLimit::into_error)?;
        Ok(Self {
            values: RefCell::new(values),
            element_type: RefCell::new(element_type),
            tuple_types: RefCell::new(None),
            tuple_contract_fixed: Cell::new(false),
            dynamic: Cell::new(false),
            runtime_type_shape: Cell::new(runtime_type_shape),
            structure_version: Cell::new(0),
        })
    }

    /// Build a mutable heterogeneous array for decoded dynamic data formats.
    pub fn dynamic(values: Vec<Value>) -> Self {
        let runtime_type_shape = array_runtime_type_shape(None, None)
            .expect("the built-in dynamic array type is within runtime limits");
        Self {
            values: RefCell::new(values),
            element_type: RefCell::new(None),
            tuple_types: RefCell::new(None),
            tuple_contract_fixed: Cell::new(false),
            dynamic: Cell::new(true),
            runtime_type_shape: Cell::new(runtime_type_shape),
            structure_version: Cell::new(0),
        }
    }

    pub fn tuple(values: Vec<Value>) -> Self {
        Self::try_tuple(values).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_tuple(values: Vec<Value>) -> Result<Self, RuntimeError> {
        let runtime_type_shape =
            tuple_value_shape(&values).map_err(RuntimeTypeLimit::into_error)?;
        let mut tuple_types = Vec::new();
        tuple_types
            .try_reserve_exact(values.len())
            .map_err(|_| RuntimeError::ResourceLimit { resource: "array" })?;
        tuple_types.extend(values.iter().map(RuntimeType::of));
        Ok(Self {
            values: RefCell::new(values),
            element_type: RefCell::new(None),
            tuple_types: RefCell::new(Some(tuple_types)),
            tuple_contract_fixed: Cell::new(false),
            dynamic: Cell::new(false),
            runtime_type_shape: Cell::new(runtime_type_shape),
            structure_version: Cell::new(0),
        })
    }

    pub(crate) fn try_typed_tuple(
        values: Vec<Value>,
        expected: Vec<RuntimeType>,
    ) -> Result<Self, RuntimeError> {
        if values.len() != expected.len() || expected.len() > 1024 {
            return Err(array_type_error());
        }
        let runtime_type_shape = array_runtime_type_shape(None, Some(&expected))
            .map_err(RuntimeTypeLimit::into_error)?;
        let mut validation = TypeValidation::default();
        for value in &values {
            ensure_value_runtime_type_shape(value).map_err(RuntimeTypeLimit::into_error)?;
        }
        for (expected, value) in expected.iter().zip(&values) {
            validate_value_type(expected, value, &mut validation)
                .map_err(|error| type_error_for(error, "tuple"))?;
        }
        for (expected, value) in expected.iter().zip(&values) {
            apply_value_type(expected, value).map_err(|error| type_error_for(error, "tuple"))?;
        }
        Ok(Self {
            values: RefCell::new(values),
            element_type: RefCell::new(None),
            tuple_types: RefCell::new(Some(expected)),
            tuple_contract_fixed: Cell::new(true),
            dynamic: Cell::new(false),
            runtime_type_shape: Cell::new(runtime_type_shape),
            structure_version: Cell::new(0),
        })
    }

    pub fn typed(values: Vec<Value>, expected: RuntimeType) -> Result<Self, ArrayTypeError> {
        Self::try_typed(values, expected).map_err(|_| ArrayTypeError)
    }

    pub(crate) fn try_typed(
        values: Vec<Value>,
        expected: RuntimeType,
    ) -> Result<Self, RuntimeError> {
        let runtime_type_shape = array_runtime_type_shape(Some(&expected), None)
            .map_err(RuntimeTypeLimit::into_error)?;
        let mut validation = TypeValidation::default();
        for value in &values {
            ensure_value_runtime_type_shape(value).map_err(RuntimeTypeLimit::into_error)?;
            validate_value_type(&expected, value, &mut validation)
                .map_err(|error| type_error_for(error, "array"))?;
        }
        for value in &values {
            apply_value_type(&expected, value).map_err(|error| type_error_for(error, "array"))?;
        }
        Ok(Self {
            values: RefCell::new(values),
            element_type: RefCell::new(Some(expected)),
            tuple_types: RefCell::new(None),
            tuple_contract_fixed: Cell::new(false),
            dynamic: Cell::new(false),
            runtime_type_shape: Cell::new(runtime_type_shape),
            structure_version: Cell::new(0),
        })
    }

    pub fn runtime_type(&self) -> RuntimeType {
        if let Some(types) = self.tuple_types.borrow().as_ref() {
            RuntimeType::Tuple(types.clone())
        } else {
            RuntimeType::Array(Box::new(
                self.element_type
                    .borrow()
                    .clone()
                    .unwrap_or(RuntimeType::Other),
            ))
        }
    }

    /// Build an independent container with this container's fixed array or
    /// tuple contract, validating copied values before applying constraints.
    pub fn with_values_like(&self, values: Vec<Value>) -> Result<Self, ArrayTypeError> {
        self.try_with_values_like(values)
            .map_err(|_| ArrayTypeError)
    }

    pub(crate) fn try_with_values_like(&self, values: Vec<Value>) -> Result<Self, RuntimeError> {
        if self.dynamic.get() {
            return Ok(Self::dynamic(values));
        }

        let tuple_types = self
            .tuple_types
            .try_borrow()
            .map_err(|_| RuntimeError::InvalidArgument { op: "clone" })?;
        if let Some(tuple_types) = tuple_types.as_ref() {
            let copy = Self::try_tuple(values)?;
            copy.constrain_tuple(tuple_types)
                .map_err(|error| type_error_for(error, "clone"))?;
            return Ok(copy);
        }
        drop(tuple_types);

        let element_type = self
            .element_type
            .try_borrow()
            .map_err(|_| RuntimeError::InvalidArgument { op: "clone" })?
            .clone();
        if let Some(element_type) = element_type {
            Self::try_typed(values, element_type).map_err(|error| match error {
                RuntimeError::TypeMismatch { .. } => RuntimeError::TypeMismatch { op: "clone" },
                error => error,
            })
        } else {
            Self::try_new(values).map_err(|error| match error {
                RuntimeError::TypeMismatch { .. } => RuntimeError::TypeMismatch { op: "clone" },
                error => error,
            })
        }
    }

    pub fn element_type(&self) -> Option<RuntimeType> {
        self.element_type.borrow().clone()
    }

    /// Sorts a primitive array in Rust when its runtime element type matches
    /// one of Cog's built-in Ord types. `None` requests the source fallback.
    pub(crate) fn try_sort_primitive(&self, stable: bool) -> Result<Option<Value>, RuntimeError> {
        let expected = self.element_type();
        let Some(expected) = expected else {
            return if self
                .values
                .try_borrow()
                .map_err(|_| RuntimeError::InvalidArgument { op: "sort" })?
                .is_empty()
            {
                Ok(Some(Value::Bool(true)))
            } else {
                Ok(None)
            };
        };
        let supported = matches!(
            &expected,
            RuntimeType::Int
                | RuntimeType::UInt
                | RuntimeType::BigInt
                | RuntimeType::Float
                | RuntimeType::Bool
                | RuntimeType::Char
                | RuntimeType::Str
        );
        if !supported {
            return Ok(None);
        }

        let mut values = self
            .values
            .try_borrow_mut()
            .map_err(|_| RuntimeError::InvalidArgument { op: "sort" })?;
        for value in values.iter() {
            if matches!(value, Value::Error(_)) {
                return Ok(Some(value.clone()));
            }
            let matches_type = matches!(
                (&expected, value),
                (RuntimeType::Int, Value::Int(_))
                    | (RuntimeType::UInt, Value::UInt(_))
                    | (RuntimeType::BigInt, Value::BigInt(_))
                    | (RuntimeType::Float, Value::Float(_))
                    | (RuntimeType::Bool, Value::Bool(_))
                    | (RuntimeType::Char, Value::Char(_))
                    | (RuntimeType::Str, Value::Str(_))
            );
            if !matches_type {
                return Err(RuntimeError::TypeMismatch { op: "sort" });
            }
            if matches!((&expected, value), (RuntimeType::Float, Value::Float(value)) if value.is_nan())
            {
                return Ok(Some(Value::Error(Box::new(RuntimeError::Unordered))));
            }
        }

        let compare = |left: &Value, right: &Value| match (left, right) {
            (Value::Int(left), Value::Int(right)) => left.cmp(right),
            (Value::UInt(left), Value::UInt(right)) => left.cmp(right),
            (Value::BigInt(left), Value::BigInt(right)) => left.cmp(right),
            (Value::Float(left), Value::Float(right)) => {
                left.partial_cmp(right).unwrap_or(Ordering::Equal)
            }
            (Value::Bool(left), Value::Bool(right)) => left.cmp(right),
            (Value::Char(left), Value::Char(right)) => left.cmp(right),
            (Value::Str(left), Value::Str(right)) => left.cmp(right),
            _ => Ordering::Equal,
        };
        if stable {
            values.sort_by(compare);
        } else {
            values.sort_unstable_by(compare);
        }
        Ok(Some(Value::Bool(true)))
    }

    pub fn needs_type_context(&self) -> bool {
        if let Some(tuple_types) = self.tuple_types.borrow().as_ref() {
            tuple_types.iter().any(has_undetermined_array_type)
        } else {
            self.element_type
                .borrow()
                .as_ref()
                .is_none_or(has_undetermined_array_type)
        }
    }

    pub(crate) fn structure_version(&self) -> u64 {
        self.structure_version.get()
    }

    #[cfg(test)]
    fn runtime_type_depth(&self) -> usize {
        self.runtime_type_shape.get().depth
    }

    pub fn constrain_from(&self, template: &Self) -> Result<(), ArrayTypeError> {
        self.try_constrain_from(template)
            .map_err(|_| ArrayTypeError)
    }

    pub(crate) fn try_constrain_from(&self, template: &Self) -> Result<(), RuntimeError> {
        if self.needs_type_context() {
            self.try_constrain_like(template)?;
        }
        Ok(())
    }

    pub(crate) fn try_constrain_like(&self, template: &Self) -> Result<(), RuntimeError> {
        let tuple_types = template.tuple_types.borrow().clone();
        if let Some(expected) = tuple_types {
            self.constrain_tuple(&expected)
        } else {
            let expected = template.element_type().ok_or_else(array_type_error)?;
            self.try_constrain(expected)
        }
    }

    /// Insert only if the value agrees with the object's established type.
    /// A previously empty array acquires its invariant from its first value.
    pub fn push(&self, value: Value, front: bool) -> Result<(), ArrayTypeError> {
        self.try_push(value, front).map_err(|_| ArrayTypeError)
    }

    /// Runtime mutation variant that preserves the distinction between an
    /// invalid value, a conflicting embedder borrow, and allocation failure.
    pub fn try_push(&self, value: Value, front: bool) -> Result<(), RuntimeError> {
        let op = if front { "arrprepend" } else { "arrpush" };
        let next_version =
            self.structure_version
                .get()
                .checked_add(1)
                .ok_or(RuntimeError::ResourceLimit {
                    resource: "array structure version",
                })?;
        if self.tuple_types.borrow().is_some() {
            return Err(RuntimeError::TypeMismatch { op });
        }
        let expected = self.element_type.borrow().clone();
        let actual = if self.dynamic.get() {
            None
        } else {
            ensure_value_runtime_type_shape(&value).map_err(RuntimeTypeLimit::into_error)?;
            if expected.is_none() {
                ensure_array_child_type_shape(&value)?;
                Some(RuntimeType::of(&value))
            } else {
                None
            }
        };
        if self.dynamic.get() {
            // Dynamic arrays intentionally accept heterogeneous values.
        } else if let Some(expected) = expected.as_ref() {
            validate_value_type(expected, &value, &mut TypeValidation::default())
                .map_err(|error| type_error_for(error, op))?;
        } else {
            if has_undetermined_array_type(
                actual.as_ref().ok_or(RuntimeError::TypeMismatch { op })?,
            ) {
                return Err(RuntimeError::TypeMismatch { op });
            }
        }
        let mut values = self
            .values
            .try_borrow_mut()
            .map_err(|_| RuntimeError::InvalidArgument { op })?;
        values
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit { resource: "array" })?;
        if !self.dynamic.get()
            && let Some(expected) = expected.as_ref()
        {
            apply_value_type(expected, &value).map_err(|error| type_error_for(error, op))?;
        }
        if expected.is_none() && !self.dynamic.get() {
            let actual = actual.ok_or(RuntimeError::TypeMismatch { op })?;
            self.runtime_type_shape.set(
                array_runtime_type_shape(Some(&actual), None)
                    .map_err(RuntimeTypeLimit::into_error)?,
            );
            *self.element_type.borrow_mut() = Some(actual);
        }
        if front {
            values.insert(0, value);
        } else {
            values.push(value);
        }
        self.structure_version.set(next_version);
        Ok(())
    }

    pub fn set(&self, index: usize, value: Value) -> Result<(), ArrayTypeError> {
        self.try_set(index, value).map_err(|_| ArrayTypeError)
    }

    pub(crate) fn try_set(&self, index: usize, value: Value) -> Result<(), RuntimeError> {
        let len = self
            .values
            .try_borrow()
            .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?
            .len();
        if index >= len {
            return Err(RuntimeError::IndexOutOfBounds {
                index: i64::try_from(index).unwrap_or(i64::MAX),
                len,
            });
        }
        if self.dynamic.get() {
            let mut values = self
                .values
                .try_borrow_mut()
                .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?;
            values[index] = value;
            return Ok(());
        }
        ensure_value_runtime_type_shape(&value).map_err(RuntimeTypeLimit::into_error)?;
        if self.tuple_types.borrow().is_some() {
            let expected = {
                let tuple_types = self
                    .tuple_types
                    .try_borrow()
                    .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?;
                let types = tuple_types
                    .as_ref()
                    .ok_or(RuntimeError::TypeMismatch { op: "indexset" })?;
                let expected = types
                    .get(index)
                    .cloned()
                    .ok_or(RuntimeError::TypeMismatch { op: "indexset" })?;
                validate_value_type(&expected, &value, &mut TypeValidation::default())
                    .map_err(|error| type_error_for(error, "indexset"))?;
                expected
            };
            let mut values = self
                .values
                .try_borrow_mut()
                .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?;
            apply_value_type(&expected, &value)
                .map_err(|error| type_error_for(error, "indexset"))?;
            if !self.tuple_contract_fixed.get() {
                let (next, shape) = {
                    let current = self
                        .tuple_types
                        .try_borrow()
                        .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?;
                    let current = current
                        .as_ref()
                        .ok_or(RuntimeError::TypeMismatch { op: "indexset" })?;
                    let mut next = Vec::new();
                    next.try_reserve_exact(current.len()).map_err(|_| {
                        RuntimeError::ResourceLimit {
                            resource: "array type",
                        }
                    })?;
                    next.extend(current.iter().cloned());
                    next[index] = RuntimeType::of(&value);
                    let shape = array_runtime_type_shape(None, Some(&next))
                        .map_err(RuntimeTypeLimit::into_error)?;
                    (next, shape)
                };
                *self
                    .tuple_types
                    .try_borrow_mut()
                    .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })? = Some(next);
                self.runtime_type_shape.set(shape);
            }
            values[index] = value;
            return Ok(());
        }
        let expected = self.element_type.borrow().clone();
        if let Some(expected) = expected.as_ref() {
            validate_value_type(expected, &value, &mut TypeValidation::default())
                .map_err(|error| type_error_for(error, "indexset"))?;
        }
        let actual = if expected.is_none() {
            let actual = RuntimeType::of(&value);
            if has_undetermined_array_type(&actual) {
                return Err(RuntimeError::TypeMismatch { op: "indexset" });
            }
            Some(actual)
        } else {
            None
        };
        let next_shape = actual
            .as_ref()
            .map(|actual| array_runtime_type_shape(Some(actual), None))
            .transpose()
            .map_err(RuntimeTypeLimit::into_error)?;
        let mut values = self
            .values
            .try_borrow_mut()
            .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?;
        if let Some(expected) = expected.as_ref() {
            apply_value_type(expected, &value)
                .map_err(|error| type_error_for(error, "indexset"))?;
        } else {
            *self.element_type.borrow_mut() = actual;
        }
        values[index] = value;
        if let Some(next_shape) = next_shape {
            self.runtime_type_shape.set(next_shape);
        }
        Ok(())
    }

    pub fn append_all(&self, values: &[Value]) -> Result<(), ArrayTypeError> {
        self.try_append_all(values).map_err(|_| ArrayTypeError)
    }

    /// Runtime bulk mutation with checked borrowing and fallible capacity growth.
    pub fn try_append_all(&self, values: &[Value]) -> Result<(), RuntimeError> {
        if self.tuple_types.borrow().is_some() {
            return Err(RuntimeError::TypeMismatch {
                op: "array::append_all",
            });
        }
        let next_version = if values.is_empty() {
            self.structure_version.get()
        } else {
            self.structure_version
                .get()
                .checked_add(1)
                .ok_or(RuntimeError::ResourceLimit {
                    resource: "array structure version",
                })?
        };
        let current_type = self.element_type.borrow().clone();
        let infer_type = !self.dynamic.get() && current_type.is_none();
        if !self.dynamic.get() {
            for value in values {
                ensure_value_runtime_type_shape(value).map_err(RuntimeTypeLimit::into_error)?;
                if infer_type {
                    ensure_array_child_type_shape(value)?;
                }
            }
        }
        let inferred = if self.dynamic.get() {
            None
        } else {
            current_type.or_else(|| {
                values
                    .iter()
                    .map(RuntimeType::of)
                    .find(|ty| !has_undetermined_array_type(ty))
            })
        };
        if !self.dynamic.get() && !values.is_empty() && inferred.is_none() {
            return Err(RuntimeError::TypeMismatch {
                op: "array::append_all",
            });
        }
        let inferred_shape = if infer_type {
            inferred
                .as_ref()
                .map(|inferred| array_runtime_type_shape(Some(inferred), None))
                .transpose()
                .map_err(RuntimeTypeLimit::into_error)?
        } else {
            None
        };
        let mut items =
            self.values
                .try_borrow_mut()
                .map_err(|_| RuntimeError::InvalidArgument {
                    op: "array::append_all",
                })?;
        if !self.dynamic.get()
            && let Some(expected) = &inferred
        {
            let mut validation = TypeValidation::default();
            for value in values {
                validate_value_type(expected, value, &mut validation)
                    .map_err(|error| type_error_for(error, "array::append_all"))?;
            }
        }
        items
            .try_reserve(values.len())
            .map_err(|_| RuntimeError::ResourceLimit { resource: "array" })?;
        if !self.dynamic.get()
            && let Some(expected) = &inferred
        {
            for value in values {
                apply_value_type(expected, value)
                    .map_err(|error| type_error_for(error, "array::append_all"))?;
            }
        }
        if infer_type {
            if let Some(inferred_shape) = inferred_shape {
                self.runtime_type_shape.set(inferred_shape);
            }
            *self.element_type.borrow_mut() = inferred;
        }
        items.extend(values.iter().cloned());
        self.structure_version.set(next_version);
        Ok(())
    }

    pub fn constrain(&self, expected: RuntimeType) -> Result<(), ArrayTypeError> {
        self.try_constrain(expected).map_err(|_| ArrayTypeError)
    }

    pub(crate) fn try_constrain(&self, expected: RuntimeType) -> Result<(), RuntimeError> {
        self.validate_constrain(&expected, &mut TypeValidation::default())?;
        self.apply_constrain(&expected)
    }

    fn validate_tuple<'a>(
        &self,
        expected: &'a [RuntimeType],
        validation: &mut TypeValidation<'a>,
    ) -> Result<(), RuntimeError> {
        array_runtime_type_shape(None, Some(expected)).map_err(RuntimeTypeLimit::into_error)?;
        let current = self
            .tuple_types
            .try_borrow()
            .map_err(|_| array_type_error())?;
        let current = current.as_ref().ok_or_else(array_type_error)?;
        if current.len() != expected.len() {
            return Err(array_type_error());
        }
        for (expected, value) in expected.iter().zip(
            self.values
                .try_borrow()
                .map_err(|_| array_type_error())?
                .iter(),
        ) {
            validate_value_type(expected, value, validation)?;
        }
        Ok(())
    }

    fn constrain_tuple(&self, expected: &[RuntimeType]) -> Result<(), RuntimeError> {
        self.validate_tuple(expected, &mut TypeValidation::default())?;
        self.apply_tuple(expected)
    }

    fn apply_tuple(&self, expected: &[RuntimeType]) -> Result<(), RuntimeError> {
        let values = self.values.try_borrow().map_err(|_| array_type_error())?;
        for (expected, value) in expected.iter().zip(values.iter()) {
            apply_value_type(expected, value)?;
        }
        if !self.tuple_contract_fixed.get() {
            let mut tuple_types = self
                .tuple_types
                .try_borrow_mut()
                .map_err(|_| array_type_error())?;
            let types = tuple_types.as_mut().ok_or_else(array_type_error)?;
            for (ty, value) in types.iter_mut().zip(values.iter()) {
                *ty = RuntimeType::of(value);
            }
            self.runtime_type_shape.set(
                array_runtime_type_shape(None, Some(types))
                    .map_err(RuntimeTypeLimit::into_error)?,
            );
        }
        self.tuple_contract_fixed.set(true);
        Ok(())
    }

    fn validate_constrain<'a>(
        &self,
        expected: &'a RuntimeType,
        validation: &mut TypeValidation<'a>,
    ) -> Result<(), RuntimeError> {
        array_runtime_type_shape(Some(expected), None).map_err(RuntimeTypeLimit::into_error)?;
        if self.tuple_types.borrow().is_some() {
            return Err(array_type_error());
        }
        let identity = self as *const Self;
        let current_type = self.element_type.borrow();
        let current = validation
            .arrays
            .get(&identity)
            .copied()
            .or(current_type.as_ref());
        if let Some(actual) = current
            && actual != expected
            && !matches!((expected, actual),
                (RuntimeType::Array(_), RuntimeType::Array(actual_inner))
                    if **actual_inner == RuntimeType::Other)
        {
            return Err(array_type_error());
        }
        if !validation.arrays.contains_key(&identity) {
            validation
                .arrays
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "array type validation",
                })?;
        }
        validation.arrays.insert(identity, expected);
        for value in self
            .values
            .try_borrow()
            .map_err(|_| array_type_error())?
            .iter()
        {
            validate_value_type(expected, value, validation)?;
        }
        Ok(())
    }

    fn apply_constrain(&self, expected: &RuntimeType) -> Result<(), RuntimeError> {
        for value in self
            .values
            .try_borrow()
            .map_err(|_| array_type_error())?
            .iter()
        {
            apply_value_type(expected, value)?;
        }
        *self.element_type.borrow_mut() = Some(expected.clone());
        self.runtime_type_shape.set(
            array_runtime_type_shape(Some(expected), None).map_err(RuntimeTypeLimit::into_error)?,
        );
        self.dynamic.set(false);
        Ok(())
    }
}

impl HeapObject for ArrayValue {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value)) {
        if let Ok(values) = self.values.try_borrow() {
            for value in values.iter() {
                visit(value);
            }
        }
    }

    fn clear_values(&self) {
        if let Ok(mut values) = self.values.try_borrow_mut() {
            values.clear();
        }
    }
}

fn checked_shape_add(
    left: usize,
    right: usize,
    limit: usize,
    kind: RuntimeTypeLimit,
) -> Result<usize, RuntimeTypeLimit> {
    left.checked_add(right)
        .filter(|sum| *sum <= limit)
        .ok_or(kind)
}

fn runtime_type_shape(
    ty: &RuntimeType,
    nesting: usize,
) -> Result<RuntimeTypeShape, RuntimeTypeLimit> {
    if nesting >= MAX_RUNTIME_TYPE_DEPTH {
        return Err(RuntimeTypeLimit::Depth);
    }
    match ty {
        RuntimeType::Array(inner) | RuntimeType::Nullable(inner) => {
            let child = runtime_type_shape(inner, nesting + 1)?;
            let depth = child.depth + 1;
            if depth > MAX_RUNTIME_TYPE_DEPTH {
                return Err(RuntimeTypeLimit::Depth);
            }
            let nodes = checked_shape_add(
                1,
                child.nodes,
                MAX_RUNTIME_TYPE_NODES,
                RuntimeTypeLimit::Nodes,
            )?;
            Ok(RuntimeTypeShape {
                depth,
                nodes,
                name_bytes: child.name_bytes,
            })
        }
        RuntimeType::Tuple(items) => tuple_type_shape(items, nesting),
        RuntimeType::Error(name) | RuntimeType::Variant(name) | RuntimeType::Nominal(name) => {
            if name.len() > MAX_RUNTIME_TYPE_NAME_BYTES {
                return Err(RuntimeTypeLimit::NameBytes);
            }
            Ok(RuntimeTypeShape {
                depth: 1,
                nodes: 1,
                name_bytes: name.len(),
            })
        }
        _ => Ok(RuntimeTypeShape {
            depth: 1,
            nodes: 1,
            name_bytes: 0,
        }),
    }
}

fn tuple_type_shape(
    items: &[RuntimeType],
    nesting: usize,
) -> Result<RuntimeTypeShape, RuntimeTypeLimit> {
    if nesting >= MAX_RUNTIME_TYPE_DEPTH {
        return Err(RuntimeTypeLimit::Depth);
    }
    let mut shape = RuntimeTypeShape {
        depth: 1,
        nodes: 1,
        name_bytes: 0,
    };
    for item in items {
        let child = runtime_type_shape(item, nesting + 1)?;
        shape.depth = shape.depth.max(child.depth + 1);
        if shape.depth > MAX_RUNTIME_TYPE_DEPTH {
            return Err(RuntimeTypeLimit::Depth);
        }
        shape.nodes = checked_shape_add(
            shape.nodes,
            child.nodes,
            MAX_RUNTIME_TYPE_NODES,
            RuntimeTypeLimit::Nodes,
        )?;
        shape.name_bytes = checked_shape_add(
            shape.name_bytes,
            child.name_bytes,
            MAX_RUNTIME_TYPE_NAME_BYTES,
            RuntimeTypeLimit::NameBytes,
        )?;
    }
    Ok(shape)
}

fn array_runtime_type_shape(
    element_type: Option<&RuntimeType>,
    tuple_types: Option<&[RuntimeType]>,
) -> Result<RuntimeTypeShape, RuntimeTypeLimit> {
    if let Some(tuple_types) = tuple_types {
        return tuple_type_shape(tuple_types, 0);
    }
    let default_type = RuntimeType::Other;
    let child = runtime_type_shape(element_type.unwrap_or(&default_type), 1)?;
    let depth = child.depth + 1;
    if depth > MAX_RUNTIME_TYPE_DEPTH {
        return Err(RuntimeTypeLimit::Depth);
    }
    Ok(RuntimeTypeShape {
        depth,
        nodes: checked_shape_add(
            1,
            child.nodes,
            MAX_RUNTIME_TYPE_NODES,
            RuntimeTypeLimit::Nodes,
        )?,
        name_bytes: child.name_bytes,
    })
}

fn value_runtime_type_shape(value: &Value) -> Result<RuntimeTypeShape, RuntimeTypeLimit> {
    match value {
        Value::Array(array) => Ok(array.runtime_type_shape.get()),
        Value::Record(record) => match &record.nominal {
            Some(name) if name.len() > MAX_RUNTIME_TYPE_NAME_BYTES => {
                Err(RuntimeTypeLimit::NameBytes)
            }
            Some(name) => Ok(RuntimeTypeShape {
                depth: 1,
                nodes: 1,
                name_bytes: name.len(),
            }),
            None => Ok(RuntimeTypeShape {
                depth: 1,
                nodes: 1,
                name_bytes: 0,
            }),
        },
        Value::Variant(tag, _) => {
            let name_bytes = tag
                .rsplit_once("::")
                .map_or(tag.len(), |(parent, _)| parent.len());
            if name_bytes > MAX_RUNTIME_TYPE_NAME_BYTES {
                return Err(RuntimeTypeLimit::NameBytes);
            }
            Ok(RuntimeTypeShape {
                depth: 1,
                nodes: 1,
                name_bytes,
            })
        }
        Value::TypeTag(tag) => {
            if tag.len() > MAX_RUNTIME_TYPE_NAME_BYTES {
                return Err(RuntimeTypeLimit::NameBytes);
            }
            Ok(RuntimeTypeShape {
                depth: 1,
                nodes: 1,
                name_bytes: tag.len(),
            })
        }
        Value::Error(_) => Ok(RuntimeTypeShape {
            depth: 1,
            nodes: 1,
            name_bytes: "error".len(),
        }),
        _ => Ok(RuntimeTypeShape {
            depth: 1,
            nodes: 1,
            name_bytes: 0,
        }),
    }
}

fn ensure_value_runtime_type_shape(value: &Value) -> Result<(), RuntimeTypeLimit> {
    let shape = value_runtime_type_shape(value)?;
    if shape.nodes > MAX_RUNTIME_TYPE_NODES {
        return Err(RuntimeTypeLimit::Nodes);
    }
    if shape.name_bytes > MAX_RUNTIME_TYPE_NAME_BYTES {
        return Err(RuntimeTypeLimit::NameBytes);
    }
    Ok(())
}

fn ensure_array_child_type_shape(value: &Value) -> Result<(), RuntimeError> {
    let child = value_runtime_type_shape(value).map_err(RuntimeTypeLimit::into_error)?;
    if child.depth + 1 > MAX_RUNTIME_TYPE_DEPTH {
        return Err(RuntimeTypeLimit::Depth.into_error());
    }
    checked_shape_add(
        1,
        child.nodes,
        MAX_RUNTIME_TYPE_NODES,
        RuntimeTypeLimit::Nodes,
    )
    .map_err(RuntimeTypeLimit::into_error)?;
    if child.name_bytes > MAX_RUNTIME_TYPE_NAME_BYTES {
        return Err(RuntimeTypeLimit::NameBytes.into_error());
    }
    Ok(())
}

fn tuple_value_shape(values: &[Value]) -> Result<RuntimeTypeShape, RuntimeTypeLimit> {
    let mut shape = RuntimeTypeShape {
        depth: 1,
        nodes: 1,
        name_bytes: 0,
    };
    for value in values {
        let child = value_runtime_type_shape(value)?;
        shape.depth = shape.depth.max(child.depth + 1);
        if shape.depth > MAX_RUNTIME_TYPE_DEPTH {
            return Err(RuntimeTypeLimit::Depth);
        }
        shape.nodes = checked_shape_add(
            shape.nodes,
            child.nodes,
            MAX_RUNTIME_TYPE_NODES,
            RuntimeTypeLimit::Nodes,
        )?;
        shape.name_bytes = checked_shape_add(
            shape.name_bytes,
            child.name_bytes,
            MAX_RUNTIME_TYPE_NAME_BYTES,
            RuntimeTypeLimit::NameBytes,
        )?;
    }
    Ok(shape)
}

fn has_undetermined_array_type(ty: &RuntimeType) -> bool {
    match ty {
        RuntimeType::Array(element) => {
            **element == RuntimeType::Other || has_undetermined_array_type(element)
        }
        RuntimeType::Nullable(inner) => has_undetermined_array_type(inner),
        RuntimeType::Tuple(items) => items.iter().any(has_undetermined_array_type),
        _ => false,
    }
}

fn validate_value_type<'a>(
    expected: &'a RuntimeType,
    value: &Value,
    validation: &mut TypeValidation<'a>,
) -> Result<(), RuntimeError> {
    match (expected, value) {
        (_, Value::Error(_)) => Ok(()),
        (RuntimeType::Nullable(_), Value::Nil) => Ok(()),
        (RuntimeType::Nullable(inner), value) => validate_value_type(inner, value, validation),
        (RuntimeType::Array(inner), Value::Array(array)) => {
            array.validate_constrain(inner, validation)
        }
        (RuntimeType::Tuple(items), Value::Array(array)) => array.validate_tuple(items, validation),
        _ if value_matches_runtime_type(expected, value) => Ok(()),
        _ => Err(array_type_error()),
    }
}

fn value_matches_runtime_type(expected: &RuntimeType, value: &Value) -> bool {
    match (expected, value) {
        (RuntimeType::Int, Value::Int(_))
        | (RuntimeType::UInt, Value::UInt(_))
        | (RuntimeType::BigInt, Value::BigInt(_))
        | (RuntimeType::Float, Value::Float(_))
        | (RuntimeType::Bool, Value::Bool(_))
        | (RuntimeType::Char, Value::Char(_))
        | (RuntimeType::Nil, Value::Nil)
        | (RuntimeType::Str, Value::Str(_) | Value::StrBuf(_))
        | (
            RuntimeType::Function,
            Value::Function(_) | Value::BoundFunction(_) | Value::Closure(_, _),
        )
        | (
            RuntimeType::Other,
            Value::Bytes(_) | Value::Handle(_) | Value::Cell(_) | Value::Iterator(_),
        ) => true,
        (RuntimeType::Record, Value::Record(record)) => record.nominal.is_none(),
        (RuntimeType::Nominal(expected), Value::Record(record)) => {
            record.nominal.as_deref() == Some(expected.as_str())
        }
        (RuntimeType::Nominal(expected), Value::Variant(tag, _)) => {
            tag.rsplit_once("::")
                .map_or(tag.as_ref(), |(parent, _)| parent)
                == expected
        }
        (RuntimeType::Nominal(expected), Value::TypeTag(tag)) => tag.as_ref() == expected,
        (RuntimeType::Nominal(expected), Value::Error(_)) => expected == crate::error::ERROR_TYPE,
        (RuntimeType::Nominal(expected), Value::Task(_)) => expected == "$core::Task",
        (RuntimeType::Variant(expected), Value::Variant(tag, _)) => {
            tag.rsplit_once("::")
                .map_or(tag.as_ref(), |(parent, _)| parent)
                == expected
        }
        (RuntimeType::Variant(expected), Value::TypeTag(tag)) => tag.as_ref() == expected,
        (RuntimeType::Error(expected), Value::Error(_)) => expected == "error",
        _ => false,
    }
}

// Callers must validate the whole operation with one shared session first.
fn apply_value_type(expected: &RuntimeType, value: &Value) -> Result<(), RuntimeError> {
    match (expected, value) {
        (_, Value::Error(_)) => Ok(()),
        (RuntimeType::Nullable(_), Value::Nil) => Ok(()),
        (RuntimeType::Nullable(inner), value) => apply_value_type(inner, value),
        (RuntimeType::Array(inner), Value::Array(array)) => array.apply_constrain(inner),
        (RuntimeType::Tuple(items), Value::Array(array)) => array.apply_tuple(items),
        _ if value_matches_runtime_type(expected, value) => Ok(()),
        _ => Err(array_type_error()),
    }
}

#[derive(Debug)]
/// Runtime record storage. Public accessors clone field values, and mutation
/// goes through checked methods so nominal field contracts cannot be bypassed.
///
/// ```compile_fail
/// use cog::value::{RecordValue, Value};
/// use std::cell::RefCell;
/// use std::collections::HashMap;
///
/// let record = RecordValue::new(HashMap::from([("count".into(), Value::Int(1))]));
/// let unchecked: &RefCell<HashMap<String, Value>> = &record;
/// unchecked.borrow_mut().insert("count".into(), Value::Str("wrong".into()));
/// ```
pub struct RecordValue {
    fields: RefCell<HashMap<String, Value>>,
    nominal: Option<String>,
    field_types: RefCell<HashMap<String, RuntimeType>>,
}

impl RecordValue {
    pub fn new(fields: HashMap<String, Value>) -> Self {
        Self {
            fields: RefCell::new(fields),
            nominal: None,
            field_types: RefCell::new(HashMap::new()),
        }
    }
    pub(crate) fn nominal(fields: HashMap<String, Value>, name: String) -> Self {
        Self {
            fields: RefCell::new(fields),
            nominal: Some(name),
            field_types: RefCell::new(HashMap::new()),
        }
    }

    pub fn runtime_type(&self) -> RuntimeType {
        self.nominal
            .clone()
            .map_or(RuntimeType::Record, RuntimeType::Nominal)
    }

    pub fn is_nominal(&self) -> bool {
        self.nominal.is_some()
    }

    pub(crate) fn try_borrow_fields(
        &self,
    ) -> Result<std::cell::Ref<'_, HashMap<String, Value>>, std::cell::BorrowError> {
        self.fields.try_borrow()
    }

    /// Return a cloned field value without exposing the record's mutable map.
    pub fn try_get_field(&self, name: &str) -> Result<Option<Value>, RuntimeError> {
        self.fields
            .try_borrow()
            .map(|fields| fields.get(name).cloned())
            .map_err(|_| RuntimeError::InvalidArgument { op: "index" })
    }

    /// Return whether the record currently has the named field.
    pub fn try_contains_field(&self, name: &str) -> Result<bool, RuntimeError> {
        self.fields
            .try_borrow()
            .map(|fields| fields.contains_key(name))
            .map_err(|_| RuntimeError::InvalidArgument { op: "record" })
    }

    /// Return the number of fields without exposing the mutable map.
    pub fn try_fields_len(&self) -> Result<usize, RuntimeError> {
        self.fields
            .try_borrow()
            .map(|fields| fields.len())
            .map_err(|_| RuntimeError::InvalidArgument { op: "record" })
    }

    pub(crate) fn try_constrain_fields(
        &self,
        contracts: &[(String, RuntimeType)],
    ) -> Result<(), RuntimeError> {
        if self.nominal.is_none() {
            return Err(RuntimeError::TypeMismatch { op: "record_type" });
        }
        let mut field_types = HashMap::new();
        field_types
            .try_reserve(contracts.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "record type",
            })?;
        let fields = self
            .fields
            .try_borrow()
            .map_err(|_| RuntimeError::InvalidArgument { op: "record_type" })?;
        let mut validation = TypeValidation::default();
        for (name, expected) in contracts {
            array_runtime_type_shape(Some(expected), None).map_err(RuntimeTypeLimit::into_error)?;
            let value = fields
                .get(name)
                .ok_or(RuntimeError::TypeMismatch { op: "record_type" })?;
            validate_value_type(expected, value, &mut validation)
                .map_err(|error| type_error_for(error, "record_type"))?;
            if field_types.insert(name.clone(), expected.clone()).is_some() {
                return Err(RuntimeError::TypeMismatch { op: "record_type" });
            }
        }
        for (name, expected) in contracts {
            if let Some(value) = fields.get(name) {
                apply_value_type(expected, value)
                    .map_err(|error| type_error_for(error, "record_type"))?;
            }
        }
        drop(fields);
        *self
            .field_types
            .try_borrow_mut()
            .map_err(|_| RuntimeError::InvalidArgument { op: "record_type" })? = field_types;
        Ok(())
    }

    /// Insert or replace a field, enforcing any declared field contract.
    pub fn try_set_field(&self, name: &str, value: Value) -> Result<(), RuntimeError> {
        let mut fields = self
            .fields
            .try_borrow_mut()
            .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?;
        let expected = self
            .field_types
            .try_borrow()
            .map_err(|_| RuntimeError::InvalidArgument { op: "indexset" })?
            .get(name)
            .cloned();
        if let Some(expected) = expected {
            ensure_value_runtime_type_shape(&value).map_err(RuntimeTypeLimit::into_error)?;
            validate_value_type(&expected, &value, &mut TypeValidation::default())
                .map_err(|error| type_error_for(error, "indexset"))?;
            apply_value_type(&expected, &value)
                .map_err(|error| type_error_for(error, "indexset"))?;
        }
        if let Some(existing) = fields.get_mut(name) {
            *existing = value;
        } else {
            let mut owned_name = String::new();
            owned_name
                .try_reserve(name.len())
                .map_err(|_| RuntimeError::ResourceLimit { resource: "record" })?;
            owned_name.push_str(name);
            fields
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit { resource: "record" })?;
            fields.insert(owned_name, value);
        }
        Ok(())
    }
}

/// Immutable named values retained by an enum variant or language Error.
/// Interior mutation is private to the collector, which clears unreachable
/// objects before releasing their graph edges.
pub struct ValueFields {
    fields: RefCell<HashMap<String, Value>>,
}

impl ValueFields {
    pub fn new(fields: HashMap<String, Value>) -> Self {
        Self {
            fields: RefCell::new(fields),
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<Value> {
        self.fields.try_borrow().ok()?.get(name).cloned()
    }

    pub(crate) fn try_borrow(
        &self,
    ) -> Result<std::cell::Ref<'_, HashMap<String, Value>>, std::cell::BorrowError> {
        self.fields.try_borrow()
    }
}

impl HeapObject for ValueFields {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value)) {
        let _ = self.fields.try_borrow().map(|fields| {
            for value in fields.values() {
                visit(value);
            }
        });
    }

    fn clear_values(&self) {
        if let Ok(mut fields) = self.fields.try_borrow_mut() {
            fields.clear();
        }
    }
}

/// Immutable closure captures. The collector owns the only mutation path.
pub struct ClosureCaptures {
    values: RefCell<Vec<Value>>,
    owner: Option<Rc<()>>,
}

impl ClosureCaptures {
    /// Create unbound capture storage for a runtime graph. Executable closures
    /// are created by the VM with a program owner; these captures alone cannot
    /// authorize a call into an arbitrary program.
    pub fn new(values: Vec<Value>) -> Self {
        Self {
            values: RefCell::new(values),
            owner: None,
        }
    }

    pub(crate) fn for_program(values: Vec<Value>, owner: &Rc<()>) -> Self {
        Self {
            values: RefCell::new(values),
            owner: Some(owner.clone()),
        }
    }

    pub(crate) fn belongs_to(&self, owner: &Rc<()>) -> bool {
        self.owner
            .as_ref()
            .is_some_and(|current| Rc::ptr_eq(current, owner))
    }

    pub(crate) fn len(&self) -> usize {
        self.values.borrow().len()
    }

    pub(crate) fn get(&self, index: usize) -> Option<Value> {
        self.values.try_borrow().ok()?.get(index).cloned()
    }
}

impl HeapObject for ClosureCaptures {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value)) {
        if let Ok(values) = self.values.try_borrow() {
            for value in values.iter() {
                visit(value);
            }
        }
    }

    fn clear_values(&self) {
        if let Ok(mut values) = self.values.try_borrow_mut() {
            values.clear();
        }
    }
}

impl HeapObject for RecordValue {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value)) {
        if let Ok(fields) = self.fields.try_borrow() {
            for value in fields.values() {
                visit(value);
            }
        }
    }

    fn clear_values(&self) {
        if let Ok(mut fields) = self.fields.try_borrow_mut() {
            fields.clear();
        }
    }
}

impl HeapObject for RefCell<Value> {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value)) {
        if let Ok(value) = self.try_borrow() {
            visit(&value);
        }
    }

    fn clear_values(&self) {
        if let Ok(mut value) = self.try_borrow_mut() {
            *value = Value::Nil;
        }
    }
}

impl HeapObject for RefCell<IteratorValue> {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value)) {
        if let Ok(iterator) = self.try_borrow() {
            visit(&iterator.source);
            if let Some(value) = &iterator.current {
                visit(value);
            }
        }
    }

    fn clear_values(&self) {
        if let Ok(mut iterator) = self.try_borrow_mut() {
            iterator.source = Value::Nil;
            iterator.current = None;
        }
    }
}

impl HeapObject for RefCell<Vec<u8>> {
    fn visit_values(&self, _: &mut dyn FnMut(&Value)) {}

    fn clear_values(&self) {
        if let Ok(mut bytes) = self.try_borrow_mut() {
            bytes.clear();
        }
    }
}

impl HeapObject for RefCell<String> {
    fn visit_values(&self, _: &mut dyn FnMut(&Value)) {}

    fn clear_values(&self) {
        if let Ok(mut text) = self.try_borrow_mut() {
            text.clear();
        }
    }
}

impl RuntimeType {
    pub fn accepts(&self, actual: &Self) -> bool {
        match self {
            Self::Nullable(inner) => {
                actual == self || actual == &Self::Nil || inner.accepts(actual)
            }
            Self::Tuple(expected) => match actual {
                Self::Tuple(actual) => {
                    expected.len() == actual.len()
                        && expected.iter().zip(actual).all(|(expected, actual)| {
                            matches!(actual, Self::Error(_)) || expected.accepts(actual)
                        })
                }
                _ => false,
            },
            Self::Nominal(name) => {
                actual == self
                    || matches!(actual, Self::Variant(parent) if parent == name)
                    || (name == crate::error::ERROR_TYPE && matches!(actual, Self::Error(_)))
            }
            _ => self == actual,
        }
    }

    pub fn of(value: &Value) -> Self {
        match value {
            Value::Int(_) => Self::Int,
            Value::UInt(_) => Self::UInt,
            Value::BigInt(_) => Self::BigInt,
            Value::Float(_) => Self::Float,
            Value::Bool(_) => Self::Bool,
            Value::Char(_) => Self::Char,
            Value::Nil => Self::Nil,
            Value::Str(_) | Value::StrBuf(_) => Self::Str,
            Value::Error(_) => Self::Error("error".to_owned()),
            Value::Array(array) => array.runtime_type(),
            Value::Record(value) => value.runtime_type(),
            Value::Variant(tag, _) => Self::Variant(
                tag.rsplit_once("::")
                    .map_or(tag.as_ref(), |(parent, _)| parent)
                    .to_owned(),
            ),
            Value::TypeTag(tag) => Self::Variant(tag.to_string()),
            Value::Task(_) => Self::Nominal("$core::Task".to_owned()),
            Value::Function(_) | Value::BoundFunction(_) | Value::Closure(_, _) => Self::Function,
            Value::Bytes(_) | Value::Handle(_) | Value::Cell(_) | Value::Iterator(_) => Self::Other,
        }
    }
}

/// Cloneable Cog task handle with a final-drop notification for its executor.
#[derive(Clone)]
pub struct TaskHandle(Rc<TaskLease>);

#[derive(Clone)]
pub(crate) struct WeakTaskHandle(Weak<TaskLease>);

struct TaskLease {
    id: usize,
    abandoned: std::sync::mpsc::Sender<usize>,
    abandoned_pending: Rc<Cell<bool>>,
}

impl TaskHandle {
    pub(crate) fn new(
        id: usize,
        abandoned: std::sync::mpsc::Sender<usize>,
        abandoned_pending: Rc<Cell<bool>>,
    ) -> Self {
        Self(Rc::new(TaskLease {
            id,
            abandoned,
            abandoned_pending,
        }))
    }

    pub(crate) fn id(&self) -> usize {
        self.0.id
    }

    pub(crate) fn downgrade(&self) -> WeakTaskHandle {
        WeakTaskHandle(Rc::downgrade(&self.0))
    }
}

impl WeakTaskHandle {
    pub(crate) fn matches(&self, handle: &TaskHandle) -> bool {
        Weak::ptr_eq(&self.0, &Rc::downgrade(&handle.0))
    }

    pub(crate) fn strong_count(&self) -> usize {
        self.0.strong_count()
    }
}

impl std::fmt::Debug for TaskHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TaskHandle({})", self.id())
    }
}

impl Drop for TaskLease {
    fn drop(&mut self) {
        let _ = self.abandoned.send(self.id);
        self.abandoned_pending.set(true);
    }
}

/// An executable function reference bound to one program. The interpreter
/// creates it from a bytecode function literal; it cannot be serialized.
#[derive(Debug, Clone)]
pub struct BoundFunction {
    pub(crate) entry: usize,
    pub(crate) owner: Rc<()>,
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    UInt(u64),
    BigInt(BigInt),
    Float(f64),
    Bool(bool),
    Char(char),
    Nil,
    Error(Box<RuntimeError>),
    Str(Rc<str>),
    StrBuf(HeapRef<RefCell<String>>),
    /// Fixed-length mutable byte storage. Every alias retains byte/bounds checks.
    Bytes(HeapRef<RefCell<Vec<u8>>>),
    Handle(crate::runtime::ResourceHandle),
    /// A serializable function literal, bound by the interpreter when loaded.
    Function(usize),
    BoundFunction(BoundFunction),
    Array(HeapRef<ArrayValue>),
    Record(HeapRef<RecordValue>),
    /// Immutable nominal tag and payload slots. Nested reference values retain
    /// their own mutability; source-level field writes cannot change the tag.
    Variant(Rc<str>, HeapRef<ValueFields>),
    /// Compiler-generated type metadata for the generic record allocator.
    TypeTag(Rc<str>),
    Cell(HeapRef<RefCell<Value>>),
    Closure(usize, HeapRef<ClosureCaptures>),
    Iterator(HeapRef<RefCell<IteratorValue>>),
    /// Invocation-local handle into the active VM task executor.
    Task(TaskHandle),
}

impl Value {
    pub(crate) fn is_heap_managed(&self) -> bool {
        matches!(
            self,
            Self::Error(_)
                | Self::StrBuf(_)
                | Self::Bytes(_)
                | Self::Array(_)
                | Self::Record(_)
                | Self::Variant(_, _)
                | Self::Cell(_)
                | Self::Closure(_, _)
                | Self::Iterator(_)
        )
    }

    pub(crate) fn visit_heap_objects(
        &self,
        visit: &mut dyn FnMut(ObjectKey),
    ) -> Result<(), RuntimeError> {
        let mut values = Vec::new();
        values
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "VM heap",
            })?;
        values.push(self);
        let mut errors = Vec::new();
        loop {
            while let Some(value) = values.pop() {
                match value {
                    Self::Array(value) => visit(value.key()),
                    Self::Record(value) => visit(value.key()),
                    Self::StrBuf(value) => visit(value.key()),
                    Self::Bytes(value) => visit(value.key()),
                    Self::Variant(_, value) => visit(value.key()),
                    Self::Cell(value) => visit(value.key()),
                    Self::Closure(_, value) => visit(value.key()),
                    Self::Iterator(value) => visit(value.key()),
                    Self::Error(error) => {
                        errors
                            .try_reserve(1)
                            .map_err(|_| RuntimeError::ResourceLimit {
                                resource: "VM heap",
                            })?;
                        errors.push(error.as_ref());
                    }
                    _ => {}
                }
            }
            let Some(error) = errors.pop() else {
                break;
            };
            match error {
                RuntimeError::LanguageError { fields, .. } => {
                    visit(fields.key());
                }
                RuntimeError::Crashed { value, .. } => {
                    values
                        .try_reserve(1)
                        .map_err(|_| RuntimeError::ResourceLimit {
                            resource: "VM heap",
                        })?;
                    values.push(value);
                }
                RuntimeError::WithSuppressed {
                    primary,
                    suppressed,
                } => {
                    let additional =
                        suppressed
                            .len()
                            .checked_add(1)
                            .ok_or(RuntimeError::ResourceLimit {
                                resource: "VM heap",
                            })?;
                    errors
                        .try_reserve(additional)
                        .map_err(|_| RuntimeError::ResourceLimit {
                            resource: "VM heap",
                        })?;
                    errors.push(primary);
                    errors.extend(suppressed.iter());
                }
                _ => {}
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(x) => write!(f, "{x}"),
            Value::UInt(x) => write!(f, "{x}"),
            Value::BigInt(x) => write!(f, "{x}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::Bool(x) => write!(f, "{x}"),
            Value::Char(x) => write!(f, "{x}"),
            Value::Nil => write!(f, "nil"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Handle(_) => write!(f, "<resource>"),
            Value::Function(addr) => write!(f, "<function@{addr}>"),
            Value::BoundFunction(function) => write!(f, "<function@{}>", function.entry),
            Value::Task(task) => write!(f, "<task#{}>", task.id()),
            Value::Array(_)
            | Value::Record(_)
            | Value::Variant(_, _)
            | Value::Cell(_)
            | Value::Iterator(_)
            | Value::StrBuf(_)
            | Value::Bytes(_)
            | Value::Error(_) => {
                // Diagnostics must not recurse without bounds or panic while
                // inspecting a borrowed graph. Language conversions use the
                // same renderer with their explicit host buffer policy.
                match crate::runtime::text::render(self, 64 * 1024, "diagnostic") {
                    Ok(text) => f.write_str(&text),
                    Err(error) => write!(f, "<unrenderable value: {}>", error.kind()),
                }
            }
            Value::TypeTag(tag) => write!(f, "<type:{tag}>"),
            Value::Closure(addr, captures) => {
                write!(f, "<closure@{addr}/{}>", captures.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArrayValue, IteratorValue, MAX_RUNTIME_TYPE_DEPTH, MAX_RUNTIME_TYPE_NODES, RecordValue,
        RuntimeType, Value,
    };
    use crate::backend::CompiledProgram;
    use crate::error::RuntimeError;
    use crate::heap::HeapRef;
    use crate::instruction::Instruction;

    #[test]
    fn nominal_record_public_setter_preserves_declared_field_contracts() {
        let record = RecordValue::nominal(
            std::collections::HashMap::from([("count".into(), Value::Int(1))]),
            "Counter".into(),
        );
        record
            .try_constrain_fields(&[("count".into(), RuntimeType::Int)])
            .unwrap();

        assert!(matches!(
            record.try_set_field("count", Value::Str("invalid".into())),
            Err(RuntimeError::TypeMismatch { op: "indexset" })
        ));
        assert!(matches!(
            record.try_get_field("count").unwrap(),
            Some(Value::Int(1))
        ));
        record.try_set_field("count", Value::Int(2)).unwrap();
        assert!(matches!(
            record.try_get_field("count").unwrap(),
            Some(Value::Int(2))
        ));
    }

    #[test]
    fn tuple_runtime_contracts_accept_component_errors_and_preserve_array_invariance() {
        use RuntimeType as T;
        let expected = T::Tuple(vec![
            T::Nullable(Box::new(T::Int)),
            T::Tuple(vec![T::Nominal("demo::State".into()), T::Str]),
            T::Array(Box::new(T::Int)),
        ]);
        for first in [T::Nil, T::Int, T::Error("error".into())] {
            let actual = T::Tuple(vec![
                first,
                T::Tuple(vec![
                    T::Variant("demo::State".into()),
                    T::Error("error".into()),
                ]),
                T::Array(Box::new(T::Int)),
            ]);
            assert!(expected.accepts(&actual));
        }
        for actual in [
            T::Tuple(vec![
                T::Bool,
                T::Tuple(vec![T::Variant("demo::State".into()), T::Str]),
                T::Array(Box::new(T::Int)),
            ]),
            T::Tuple(vec![
                T::Int,
                T::Tuple(vec![T::Variant("other::State".into()), T::Str]),
                T::Array(Box::new(T::Int)),
            ]),
            T::Tuple(vec![
                T::Int,
                T::Tuple(vec![T::Variant("demo::State".into()), T::Str]),
                T::Array(Box::new(T::Nullable(Box::new(T::Int)))),
            ]),
            T::Tuple(vec![T::Int]),
            T::Array(Box::new(T::Int)),
        ] {
            assert!(!expected.accepts(&actual), "accepted {actual:?}");
        }
        assert!(!T::Int.accepts(&T::Error("error".into())));
        assert!(!T::Array(Box::new(T::Int)).accepts(&T::Array(Box::new(T::Error("error".into())))));
    }

    #[test]
    fn checked_array_mutations_preserve_type_shape_and_iterator_versions() {
        let array = HeapRef::new(ArrayValue::typed(vec![Value::Int(7)], RuntimeType::Int).unwrap());
        let original_shape = array.runtime_type_shape.get();

        assert!(array.set(0, Value::Str("wrong type".into())).is_err());
        assert!(array.push(Value::Str("wrong type".into()), false).is_err());
        assert!(
            array
                .append_all(&[Value::Int(8), Value::Bool(false)])
                .is_err()
        );
        assert!(matches!(array.borrow().as_slice(), [Value::Int(7)]));
        assert_eq!(
            array.runtime_type(),
            RuntimeType::Array(Box::new(RuntimeType::Int))
        );
        let actual_shape = array.runtime_type_shape.get();
        assert_eq!(actual_shape.depth, original_shape.depth);
        assert_eq!(actual_shape.nodes, original_shape.nodes);
        assert_eq!(actual_shape.name_bytes, original_shape.name_bytes);
        assert_eq!(array.structure_version(), 0);

        array.set(0, Value::Int(8)).unwrap();
        assert!(matches!(array.borrow().as_slice(), [Value::Int(8)]));
        assert_eq!(array.structure_version(), 0);

        let mut iterator = IteratorValue::new(Value::Array(array.clone())).unwrap();
        assert!(iterator.next().unwrap());
        array.push(Value::Int(9), false).unwrap();
        assert!(matches!(
            iterator.next(),
            Err(RuntimeError::InvalidArgument {
                op: "array iteration invalidated"
            })
        ));
    }

    #[test]
    fn vm_array_access_turns_internal_borrow_conflicts_into_language_errors() {
        let array = HeapRef::new(ArrayValue::new(vec![Value::Int(7)]).unwrap());
        let _internal_borrow = array.values.borrow_mut();
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Array(array.clone())),
            Instruction::Len,
            Instruction::Halt,
        ]);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::interpreter::run_with(&program, &mut Vec::new())
        }));
        assert!(
            result.is_ok(),
            "array borrow conflict panicked inside the VM"
        );
        assert!(matches!(
            result.unwrap(),
            Err(RuntimeError::InvalidArgument { op: "len" })
        ));
    }

    #[test]
    fn diagnostics_handle_an_internal_mutable_array_borrow() {
        let array = HeapRef::new(ArrayValue::new(vec![Value::Int(1)]).unwrap());
        let _internal_borrow = array.values.borrow_mut();
        let value = Value::Array(array.clone());
        assert!(value.to_string().contains("unrenderable value"));
    }

    #[test]
    fn nested_array_type_descriptors_stop_at_the_runtime_depth_limit() {
        let mut value = Value::Int(0);
        for _ in 0..MAX_RUNTIME_TYPE_DEPTH - 1 {
            value =
                Value::Array(HeapRef::try_new(ArrayValue::try_new(vec![value]).unwrap()).unwrap());
        }
        let Value::Array(array) = &value else {
            unreachable!();
        };
        assert_eq!(array.runtime_type_depth(), MAX_RUNTIME_TYPE_DEPTH);

        assert!(matches!(
            ArrayValue::try_new(vec![value]),
            Err(RuntimeError::ResourceLimit {
                resource: "array type depth"
            })
        ));
    }

    #[test]
    fn explicit_array_type_contracts_obey_the_descriptor_depth_limit() {
        let mut element_type = RuntimeType::Int;
        for _ in 0..MAX_RUNTIME_TYPE_DEPTH - 1 {
            element_type = RuntimeType::Array(Box::new(element_type));
        }
        let empty = ArrayValue::new(Vec::new()).unwrap();

        for result in [
            ArrayValue::try_typed(Vec::new(), element_type.clone()).map(|_| ()),
            empty.try_constrain(element_type).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(RuntimeError::ResourceLimit {
                    resource: "array type depth"
                })
            ));
        }
        assert_eq!(
            empty.runtime_type(),
            RuntimeType::Array(Box::new(RuntimeType::Other))
        );
    }

    #[test]
    fn branching_tuple_descriptors_stop_at_the_runtime_node_limit() {
        let mut value = Value::Int(0);
        for _ in 0..9 {
            let items = vec![value.clone(); 4];
            value = Value::Array(HeapRef::try_new(ArrayValue::try_tuple(items).unwrap()).unwrap());
        }
        let Value::Array(array) = &value else {
            unreachable!();
        };
        assert!(array.runtime_type_shape.get().nodes < MAX_RUNTIME_TYPE_NODES);

        assert!(matches!(
            ArrayValue::try_tuple(vec![value.clone(); 4]),
            Err(RuntimeError::ResourceLimit {
                resource: "array type nodes"
            })
        ));
    }

    #[test]
    fn replacing_a_large_tuple_type_with_error_reduces_its_cached_shape() {
        let mut value = Value::Int(0);
        for _ in 0..6 {
            value = Value::Array(
                HeapRef::try_new(ArrayValue::try_tuple(vec![value.clone(); 4]).unwrap()).unwrap(),
            );
        }
        let tuple = ArrayValue::try_tuple(vec![value]).unwrap();
        assert!(tuple.runtime_type_shape.get().nodes > 1_000);

        tuple
            .set(0, Value::Error(Box::new(RuntimeError::Unordered)))
            .unwrap();

        assert!(tuple.runtime_type_shape.get().nodes < 10);
        assert_eq!(
            tuple.runtime_type(),
            RuntimeType::Tuple(vec![RuntimeType::Error("error".to_owned())])
        );
    }

    #[test]
    fn tuple_set_refreshes_the_type_after_constraining_an_empty_array() {
        let original =
            HeapRef::try_new(ArrayValue::typed(vec![Value::Int(1)], RuntimeType::Int).unwrap())
                .unwrap();
        let tuple = ArrayValue::try_tuple(vec![Value::Array(original)]).unwrap();
        let replacement = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
        tuple.set(0, Value::Array(replacement.clone())).unwrap();
        assert_eq!(
            replacement.runtime_type(),
            RuntimeType::Array(Box::new(RuntimeType::Int))
        );
        assert_eq!(
            tuple.runtime_type(),
            RuntimeType::Tuple(vec![RuntimeType::Array(Box::new(RuntimeType::Int))])
        );
        assert!(tuple.set(0, Value::Bool(false)).is_err());
        assert_eq!(
            tuple.runtime_type(),
            RuntimeType::Tuple(vec![RuntimeType::Array(Box::new(RuntimeType::Int))])
        );
        assert!(replacement.try_borrow().unwrap().is_empty());
    }

    #[test]
    fn tuple_constraint_refreshes_nested_array_type() {
        let untyped_inner = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
        let source = ArrayValue::try_tuple(vec![Value::Array(untyped_inner)]).unwrap();

        let typed_inner =
            HeapRef::try_new(ArrayValue::typed(Vec::new(), RuntimeType::Int).unwrap()).unwrap();
        let template = ArrayValue::try_tuple(vec![Value::Array(typed_inner)]).unwrap();

        source.constrain_from(&template).unwrap();
        assert_eq!(
            source.runtime_type(),
            RuntimeType::Tuple(vec![RuntimeType::Array(Box::new(RuntimeType::Int))])
        );
    }

    #[test]
    fn failed_aliased_tuple_constraint_preserves_nested_array_types() {
        let shared = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
        let tuple = HeapRef::try_new(
            ArrayValue::try_tuple(vec![
                Value::Array(shared.clone()),
                Value::Array(shared.clone()),
            ])
            .unwrap(),
        )
        .unwrap();
        let original_type = tuple.runtime_type();
        let expected = RuntimeType::Tuple(vec![
            RuntimeType::Array(Box::new(RuntimeType::Int)),
            RuntimeType::Array(Box::new(RuntimeType::Str)),
        ]);

        assert!(matches!(
            ArrayValue::try_typed(vec![Value::Array(tuple.clone())], expected),
            Err(RuntimeError::TypeMismatch { .. })
        ));
        assert_eq!(shared.element_type(), None);
        assert_eq!(tuple.runtime_type(), original_type);
    }

    fn contradictory_tuple_contract() -> RuntimeType {
        RuntimeType::Tuple(vec![
            RuntimeType::Array(Box::new(RuntimeType::Int)),
            RuntimeType::Array(Box::new(RuntimeType::Str)),
        ])
    }

    fn tuple_array(values: Vec<Value>) -> Value {
        Value::Array(HeapRef::try_new(ArrayValue::try_tuple(values).unwrap()).unwrap())
    }

    #[test]
    fn batch_constraints_share_alias_facts_across_separate_tuple_roots() {
        for inferred in [false, true] {
            let shared = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
            let other =
                || Value::Array(HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap());
            let first = tuple_array(vec![Value::Array(shared.clone()), other()]);
            let second = tuple_array(vec![other(), Value::Array(shared.clone())]);
            let first_type = RuntimeType::of(&first);
            let second_type = RuntimeType::of(&second);
            let values = if inferred {
                let template = tuple_array(vec![
                    Value::Array(
                        HeapRef::try_new(ArrayValue::typed(Vec::new(), RuntimeType::Int).unwrap())
                            .unwrap(),
                    ),
                    Value::Array(
                        HeapRef::try_new(ArrayValue::typed(Vec::new(), RuntimeType::Str).unwrap())
                            .unwrap(),
                    ),
                ]);
                vec![template, first.clone(), second.clone()]
            } else {
                vec![first.clone(), second.clone()]
            };
            let result = if inferred {
                ArrayValue::try_new(values)
            } else {
                ArrayValue::try_typed(values, contradictory_tuple_contract())
            };
            assert!(matches!(result, Err(RuntimeError::TypeMismatch { .. })));
            assert_eq!(shared.element_type(), None);
            assert_eq!(RuntimeType::of(&first), first_type);
            assert_eq!(RuntimeType::of(&second), second_type);
        }
    }

    #[test]
    fn contradictory_alias_mutations_preserve_values_metadata_and_versions() {
        for operation in 0..6 {
            let shared = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
            let candidate = tuple_array(vec![
                Value::Array(shared.clone()),
                Value::Array(shared.clone()),
            ]);
            let original_type = RuntimeType::of(&candidate);
            let expected = contradictory_tuple_contract();
            let target = if operation == 4 || operation == 5 {
                ArrayValue::dynamic(vec![candidate.clone()])
            } else {
                ArrayValue::try_typed(
                    if operation == 2 {
                        vec![Value::Error(Box::new(RuntimeError::Unordered))]
                    } else {
                        Vec::new()
                    },
                    expected.clone(),
                )
                .unwrap()
            };
            let original_target_type = target.runtime_type();
            let original_len = target.borrow().len();
            let original_version = target.structure_version();
            let result = match operation {
                0 => target.try_push(candidate.clone(), false),
                1 => target.try_push(candidate.clone(), true),
                2 => target.try_set(0, candidate.clone()),
                3 => target.try_append_all(std::slice::from_ref(&candidate)),
                4 => target.try_constrain(expected.clone()),
                _ => {
                    let template = ArrayValue::try_typed(Vec::new(), expected.clone()).unwrap();
                    target
                        .constrain_from(&template)
                        .map_err(|_| super::array_type_error())
                }
            };
            assert!(
                matches!(result, Err(RuntimeError::TypeMismatch { .. })),
                "operation {operation}"
            );
            assert_eq!(shared.element_type(), None, "operation {operation}");
            assert_eq!(RuntimeType::of(&candidate), original_type);
            assert_eq!(target.runtime_type(), original_target_type);
            assert_eq!(target.borrow().len(), original_len);
            assert_eq!(target.structure_version(), original_version);
        }
    }

    #[test]
    fn bulk_append_rejects_conflicting_aliases_across_distinct_roots_atomically() {
        let shared = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
        let other =
            || Value::Array(HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap());
        let first = tuple_array(vec![Value::Array(shared.clone()), other()]);
        let second = tuple_array(vec![other(), Value::Array(shared.clone())]);
        let original_first = RuntimeType::of(&first);
        let target = ArrayValue::try_typed(Vec::new(), contradictory_tuple_contract()).unwrap();
        assert!(matches!(
            target.try_append_all(&[first.clone(), second]),
            Err(RuntimeError::TypeMismatch { .. })
        ));
        assert_eq!(shared.element_type(), None);
        assert_eq!(RuntimeType::of(&first), original_first);
        assert!(target.borrow().is_empty());
        assert_eq!(target.structure_version(), 0);
    }

    #[test]
    fn tuple_replacement_and_copy_reject_conflicting_aliases_before_constraining() {
        let shared = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
        let values = vec![Value::Array(shared.clone()), Value::Array(shared.clone())];
        let candidate = tuple_array(values.clone());
        let original_type = RuntimeType::of(&candidate);
        let template = ArrayValue::try_tuple(vec![
            Value::Array(
                HeapRef::try_new(ArrayValue::typed(Vec::new(), RuntimeType::Int).unwrap()).unwrap(),
            ),
            Value::Array(
                HeapRef::try_new(ArrayValue::typed(Vec::new(), RuntimeType::Str).unwrap()).unwrap(),
            ),
        ])
        .unwrap();
        let target =
            ArrayValue::try_tuple(vec![Value::Array(HeapRef::try_new(template).unwrap())]).unwrap();
        let original_target = target.runtime_type();
        assert!(matches!(
            target.try_set(0, candidate.clone()),
            Err(RuntimeError::TypeMismatch { .. })
        ));
        assert_eq!(target.runtime_type(), original_target);
        let Value::Array(template) = &target.borrow()[0] else {
            unreachable!();
        };
        let Value::Array(candidate_array) = &candidate else {
            unreachable!();
        };
        assert!(candidate_array.constrain_from(template).is_err());
        assert!(matches!(
            template.try_with_values_like(values),
            Err(RuntimeError::TypeMismatch { .. })
        ));
        assert_eq!(shared.element_type(), None);
        assert_eq!(RuntimeType::of(&candidate), original_type);
    }

    #[test]
    fn compatible_tuple_aliases_accept_nullable_scalars_and_array_refinement() {
        let scalar = tuple_array(vec![Value::Int(7)]);
        let container = ArrayValue::try_tuple(vec![scalar.clone(), scalar]).unwrap();
        container
            .constrain_tuple(&[
                RuntimeType::Tuple(vec![RuntimeType::Int]),
                RuntimeType::Tuple(vec![RuntimeType::Nullable(Box::new(RuntimeType::Int))]),
            ])
            .unwrap();

        let outer = HeapRef::try_new(ArrayValue::new(Vec::new()).unwrap()).unwrap();
        outer
            .try_constrain(RuntimeType::Array(Box::new(RuntimeType::Other)))
            .unwrap();
        let value = tuple_array(vec![
            Value::Array(outer.clone()),
            Value::Array(outer.clone()),
        ]);
        ArrayValue::try_typed(
            vec![value],
            RuntimeType::Tuple(vec![
                RuntimeType::Array(Box::new(RuntimeType::Array(Box::new(RuntimeType::Other)))),
                RuntimeType::Array(Box::new(RuntimeType::Array(Box::new(RuntimeType::Int)))),
            ]),
        )
        .unwrap();
        assert_eq!(
            outer.element_type(),
            Some(RuntimeType::Array(Box::new(RuntimeType::Int)))
        );
    }

    #[test]
    fn fallible_tuple_copy_preserves_the_nested_array_contract() {
        let original_inner =
            HeapRef::try_new(ArrayValue::typed(vec![Value::Int(1)], RuntimeType::Int).unwrap())
                .unwrap();
        let original = ArrayValue::try_tuple(vec![Value::Array(original_inner)]).unwrap();

        let copied_inner = HeapRef::try_new(ArrayValue::new(vec![Value::Int(2)]).unwrap()).unwrap();
        let copy = original
            .try_with_values_like(vec![Value::Array(copied_inner)])
            .unwrap();

        assert_eq!(copy.runtime_type(), original.runtime_type());
    }
}
