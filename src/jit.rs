//! Opt-in, profile-guided Cranelift OSR for simple synchronous bytecode loops.
//!
//! The JIT executes against an explicit frame of tagged slots. Opaque Cog
//! values stay rooted in the VM for the duration of the native call. Any
//! operation outside the lowered subset returns to the VM at that exact PC.

use crate::backend::CompiledProgram;
use crate::error::RuntimeError;
use crate::instruction::{Instruction, ValueType};
use crate::value::{RuntimeType, Value};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::immediates::Ieee64;
use cranelift_codegen::ir::{
    AbiParam, BlockArg, FuncRef, InstBuilder, MemFlagsData, UserFuncName, types,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{
    JITBuilder, JITMemoryKind, JITMemoryProvider, JITModule, SystemMemoryProvider,
};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::mem::{offset_of, size_of, transmute};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

const MAX_FUNCTION_INSTRUCTIONS: usize = 512;
const MAX_LOCALS: usize = 256;
const MAX_STACK: usize = 256;
const MAX_JIT_FUNCTIONS: usize = 512;
const MAX_JIT_CODE_BYTES: usize = 64 * 1024 * 1024;
const MAX_JIT_ENUM_VARIANTS: usize = 32;
const MAX_JIT_ROOTS: usize = 65_536;
const MAX_JIT_CALL_ARGS: usize = 16;
const MAX_BOOTSTRAP_PROBES: usize = 3;
const WINDOW_BACKEDGES: u64 = 256;
const REQUIRED_WINDOWS: u32 = 3;
const HOT_CALL_THRESHOLD: u64 = 128;
const MAX_HORIZON_NS: u64 = 1_000_000_000;
const MAX_ACTIVITY_GAP_NS: u64 = 1_000_000_000;
const TOKEN_CAPACITY_NS: u64 = 20_000_000;
const TOKEN_REFILL_PER_NS: f64 = 0.02;
const MAX_PROFILE_METRIC_NS: u64 = 60_000_000_000;
const APE_VERSION: u32 = 16;
static PROFILE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const EXIT_COMPLETED: i32 = 0;
const EXIT_RESUME: i32 = 1;
const EXIT_FAULT: i32 = 2;
const REASON_NORMAL: i64 = 0;
const REASON_PROFILE_SLICE: i64 = 1;

const TAG_UNINITIALIZED: u64 = 0;
const TAG_INT: u64 = 1;
const TAG_UINT: u64 = 2;
const TAG_BIGINT: u64 = 3;
const TAG_FLOAT: u64 = 4;
const TAG_BOOL: u64 = 5;
const TAG_CHAR: u64 = 6;
const TAG_STR: u64 = 7;
const TAG_ERROR: u64 = 8;
const TAG_NIL: u64 = 9;
const TAG_OPAQUE: u64 = 10;
const TAG_ARRAY: u64 = 11;
// Variant tags occupy a disjoint range while the payload continues to point
// at the rooted Cog value. This lets generated code compare nominal identity
// without copying or exposing the variant payload.
const TAG_VARIANT_BASE: u64 = 1 << 32;
// Small BigInts keep their original Cog value rooted. The tag carries the
// root index while the payload carries an exact signed or unsigned i64 value.
const TAG_BIGINT_SMALL_SIGNED_BASE: u64 = 2 << 32;
const TAG_BIGINT_SMALL_UNSIGNED_BASE: u64 = 3 << 32;
// Native arithmetic results have no input root, so restore them from payload.
const TAG_BIGINT_JIT_SIGNED: u64 = 12;
const TAG_BIGINT_JIT_UNSIGNED: u64 = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitExit {
    Completed,
    ResumeAt { pc: usize },
    FaultAt { pc: usize },
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct JitSlot {
    tag: u64,
    payload: u64,
}

#[repr(C)]
struct JitFrame {
    locals: *mut JitSlot,
    stack: *mut JitSlot,
    captures: *mut JitSlot,
    capture_count: u64,
    roots: *mut Vec<Value>,
    runtime_context: *mut std::ffi::c_void,
    pc: u64,
    stack_len: u64,
    propagation_pending: u64,
    backedge_budget: u64,
    backedge_count: u64,
    exit_reason: i64,
}

type JitEntry = unsafe extern "C" fn(*mut JitFrame) -> i32;

#[derive(Clone)]
struct JitCallable {
    entry: JitEntry,
    plan: Arc<FunctionPlan>,
}

struct JitCallFrame {
    frame: JitFrame,
    locals: Vec<JitSlot>,
    stack: Vec<JitSlot>,
    roots: Vec<Value>,
}

impl JitCallFrame {
    fn new(runtime_context: *mut std::ffi::c_void) -> Self {
        let mut locals = Vec::new();
        let mut stack = Vec::new();
        let roots = Vec::new();
        let frame = JitFrame {
            locals: locals.as_mut_ptr(),
            stack: stack.as_mut_ptr(),
            captures: std::ptr::null_mut(),
            capture_count: 0,
            roots: std::ptr::null_mut(),
            runtime_context,
            pc: 0,
            stack_len: 0,
            propagation_pending: 0,
            backedge_budget: 0,
            backedge_count: 0,
            exit_reason: REASON_NORMAL,
        };
        Self {
            frame,
            locals,
            stack,
            roots,
        }
    }

    fn prepare(&mut self, plan: &FunctionPlan, runtime_context: *mut std::ffi::c_void) {
        self.locals.resize(plan.local_count, JitSlot::default());
        self.locals.fill(JitSlot::default());
        self.stack.resize(plan.max_stack.max(1), JitSlot::default());
        self.stack.fill(JitSlot::default());
        self.roots.clear();
        self.frame.locals = self.locals.as_mut_ptr();
        self.frame.stack = self.stack.as_mut_ptr();
        self.frame.roots = &mut self.roots;
        self.frame.runtime_context = runtime_context;
        self.frame.pc = plan.entry as u64;
        self.frame.stack_len = 0;
        self.frame.propagation_pending = 0;
        self.frame.backedge_budget = 0;
        self.frame.backedge_count = 0;
        self.frame.exit_reason = REASON_NORMAL;
    }
}

struct JitExecutionContext<'a> {
    artifact: &'a CompiledProgram,
    variant_tags: &'a VariantTags,
    callables: HashMap<usize, JitCallable>,
    call_frame: Option<Box<JitCallFrame>>,
}

#[derive(Clone)]
struct FunctionPlan {
    entry: usize,
    end: usize,
    instruction_count: usize,
    block_count: usize,
    edge_count: usize,
    phi_estimate: usize,
    local_count: usize,
    max_stack: usize,
    stack_depths: Vec<Option<usize>>,
    osr_entries: Vec<usize>,
    has_backedges: bool,
    callable_from_jit: bool,
}

#[derive(Debug, Clone, Default)]
struct VariantTags {
    names: Vec<String>,
    by_name: HashMap<String, u32>,
    enum_variants: HashMap<String, Vec<u32>>,
}

impl VariantTags {
    fn from_program(artifact: &CompiledProgram) -> Self {
        let mut names = std::collections::BTreeSet::new();
        for instruction in &artifact.instructions {
            match instruction {
                Instruction::Push(Value::TypeTag(name)) => {
                    let name = name.as_ref();
                    if is_user_variant_tag(name) {
                        names.insert(name.to_owned());
                    }
                }
                Instruction::IsType(ValueType::Variant(name)) => {
                    let name = name.as_ref();
                    if is_user_variant_tag(name) {
                        names.insert(name.to_owned());
                    }
                }
                _ => {}
            }
        }

        let names: Vec<_> = names.into_iter().collect();
        let mut by_name = HashMap::with_capacity(names.len());
        let mut enum_variants: HashMap<String, Vec<u32>> = HashMap::new();
        for (index, name) in names.iter().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                // The verified bytecode limit makes this unreachable, but an
                // empty registry keeps the JIT conservative if limits change.
                return Self::default();
            };
            by_name.insert(name.clone(), index);
            if let Some((enum_name, _)) = name.rsplit_once("::") {
                enum_variants
                    .entry(enum_name.to_owned())
                    .or_default()
                    .push(index);
            }
        }
        Self {
            names,
            by_name,
            enum_variants,
        }
    }

    fn slot_tag(&self, name: &str) -> Option<u64> {
        self.by_name
            .get(name)
            .map(|index| TAG_VARIANT_BASE + u64::from(*index))
    }

    fn variant_tags(&self, value_type: &ValueType) -> Option<Vec<u64>> {
        match value_type {
            ValueType::Variant(name) => Some(vec![self.slot_tag(name)?]),
            ValueType::Enum(name) => {
                let variants = self.enum_variants.get(name.as_ref())?;
                if variants.is_empty() || variants.len() > MAX_JIT_ENUM_VARIANTS {
                    return None;
                }
                Some(
                    variants
                        .iter()
                        .map(|index| TAG_VARIANT_BASE + u64::from(*index))
                        .collect(),
                )
            }
            _ => None,
        }
    }
}

fn is_user_variant_tag(name: &str) -> bool {
    !name.starts_with("@struct::")
        && !name
            .strip_prefix(crate::error::ERROR_TYPE)
            .is_some_and(|tail| tail.starts_with("::"))
}

impl FunctionPlan {
    fn feature_vector(&self) -> [f64; 6] {
        [
            1.0,
            self.instruction_count as f64 / 8.0,
            self.block_count as f64,
            self.edge_count as f64,
            self.phi_estimate as f64 / 8.0,
            (self.local_count + self.max_stack) as f64 / 8.0,
        ]
    }

    fn upper_compile_ns(&self, coefficients: &Coefficients) -> u64 {
        coefficients.estimate_ns(self).saturating_mul(3).div_ceil(2)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Coefficients {
    a0: f64,
    instructions: f64,
    blocks: f64,
    edges: f64,
    parameters: f64,
    live_values: f64,
}

impl Coefficients {
    fn seeded() -> Self {
        Self {
            a0: 100_000.0,
            instructions: 3_000.0,
            blocks: 10_000.0,
            edges: 3_000.0,
            parameters: 1_000.0,
            live_values: 500.0,
        }
    }

    fn estimate_ns(&self, plan: &FunctionPlan) -> u64 {
        let x = plan.feature_vector();
        (self.a0
            + self.instructions * x[1]
            + self.blocks * x[2]
            + self.edges * x[3]
            + self.parameters * x[4]
            + self.live_values * x[5])
            .clamp(1_000.0, 1_000_000_000.0) as u64
    }

    fn update(&mut self, plan: &FunctionPlan, actual_ns: u64) {
        let x = plan.feature_vector();
        let predicted = self.estimate_ns(plan) as f64;
        let error = actual_ns as f64 - predicted;
        let denominator = 1.0 + x.iter().map(|value| value * value).sum::<f64>();
        let eta = 0.15;
        let coefficients = [
            &mut self.a0,
            &mut self.instructions,
            &mut self.blocks,
            &mut self.edges,
            &mut self.parameters,
            &mut self.live_values,
        ];
        for (coefficient, feature) in coefficients.into_iter().zip(x) {
            *coefficient = (*coefficient + eta * error * feature / denominator).max(0.0);
        }
    }

    fn is_valid(&self) -> bool {
        [
            self.a0,
            self.instructions,
            self.blocks,
            self.edges,
            self.parameters,
            self.live_values,
        ]
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0e12).contains(&value))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct FunctionProfile {
    compile_ns: u64,
    vm_ns_per_backedge: u64,
    jit_ns_per_backedge: u64,
    #[serde(skip)]
    total_backedges: u64,
    #[serde(skip)]
    total_calls: u64,
    #[serde(skip)]
    hot_series_ns: u64,
    #[serde(skip)]
    active_windows: u32,
    calibrated: bool,
}

impl FunctionProfile {
    fn is_valid(&self) -> bool {
        if [
            self.compile_ns,
            self.vm_ns_per_backedge,
            self.jit_ns_per_backedge,
        ]
        .into_iter()
        .any(|value| value > MAX_PROFILE_METRIC_NS)
        {
            return false;
        }
        if self.calibrated {
            self.compile_ns > 0 && self.vm_ns_per_backedge > 0 && self.jit_ns_per_backedge > 0
        } else {
            self.jit_ns_per_backedge == 0
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ProfileFile {
    key: String,
    coefficients: Option<Coefficients>,
    functions: BTreeMap<usize, FunctionProfile>,
}

struct Candidate {
    profile: FunctionProfile,
    window_edges: u64,
    windows_this_run: u32,
    window_started: Instant,
    last_backedge_at: Option<Instant>,
    last_selection_window: u32,
    retry_after_backedges: u64,
    retry_after_calls: u64,
    last_selection_call: u64,
    compiled: Option<CompiledFunction>,
    disabled: bool,
}

impl Candidate {
    fn new(profile: FunctionProfile) -> Self {
        Self {
            profile,
            window_edges: 0,
            windows_this_run: 0,
            window_started: Instant::now(),
            last_backedge_at: None,
            last_selection_window: 0,
            retry_after_backedges: 0,
            retry_after_calls: 0,
            last_selection_call: 0,
            compiled: None,
            disabled: false,
        }
    }
}

struct CompiledFunction {
    entry: JitEntry,
    probe_pending: bool,
    call_compiled: bool,
}

#[derive(Debug, Clone, Copy)]
enum ApeDecision {
    Profile(&'static str),
    BootstrapCompile {
        priority: f64,
        estimated_cost_ns: u64,
    },
    Compile {
        priority: f64,
        estimated_cost_ns: u64,
    },
    Ineligible(&'static str),
    DeferBudget {
        estimated_cost_ns: u64,
        available_ns: i64,
    },
    SkipCacheFull,
}

fn log_ape_decision(entry: usize, decision: ApeDecision) {
    match decision {
        ApeDecision::Profile(reason) => {
            eprintln!("[jit] decision function={entry} action=profile reason={reason}");
        }
        ApeDecision::BootstrapCompile {
            priority,
            estimated_cost_ns,
        } => eprintln!(
            "[jit] decision function={entry} action=bootstrap priority={priority:.3} estimated_compile_ns={estimated_cost_ns}"
        ),
        ApeDecision::Compile {
            priority,
            estimated_cost_ns,
        } => eprintln!(
            "[jit] decision function={entry} action=compile priority={priority:.3} estimated_compile_ns={estimated_cost_ns}"
        ),
        ApeDecision::Ineligible(reason) => {
            eprintln!("[jit] decision function={entry} action=skip reason={reason}");
        }
        ApeDecision::DeferBudget {
            estimated_cost_ns,
            available_ns,
        } => eprintln!(
            "[jit] decision function={entry} action=defer-budget estimated_compile_ns={estimated_cost_ns} available_ns={available_ns}"
        ),
        ApeDecision::SkipCacheFull => {
            eprintln!("[jit] decision function={entry} action=skip reason=cache-full");
        }
    }
}

pub(crate) enum JitPlanState {
    NotInitialized,
    Unavailable,
    Ready(Box<JitState>),
}

pub(crate) struct JitState {
    plans: BTreeMap<usize, Arc<FunctionPlan>>,
    variant_tags: VariantTags,
    candidates: BTreeMap<usize, Candidate>,
    compiler: JitCompiler,
    coefficients: Coefficients,
    profile_key: String,
    profile_path: Option<PathBuf>,
    bootstrap_probes: usize,
    compile_tokens_ns: i64,
    token_refill_at: Instant,
}

impl JitState {
    pub(crate) fn new(
        artifact: &CompiledProgram,
        profile_path: Option<&Path>,
    ) -> Result<Self, String> {
        crate::bytecode::validate(artifact)
            .map_err(|_| "bytecode verification failed".to_owned())?;
        let variant_tags = VariantTags::from_program(artifact);
        let plans = analyze_program(artifact);
        if plans.is_empty() {
            return Err("no synchronous function is eligible for JIT compilation".into());
        }
        let (compiler, host_key) = JitCompiler::new()?;
        let encoded = crate::bytecode::encode(artifact).map_err(|error| error.to_string())?;
        let bytecode_hash = blake3::hash(&encoded).to_hex().to_string();
        let profile_key = format!(
            "{bytecode_hash}|abi={}|bytecode={}|ape={APE_VERSION}|cranelift={}|host={host_key}",
            crate::sdk::ABI_VERSION,
            crate::bytecode::VERSION,
            cranelift_jit::VERSION,
        );
        let loaded = profile_path.and_then(|path| load_profile(path, &profile_key));
        let coefficients = loaded
            .as_ref()
            .and_then(|profile| profile.coefficients.clone())
            .unwrap_or_else(Coefficients::seeded);
        let mut candidates = BTreeMap::new();
        for entry in plans.keys() {
            let profile = loaded
                .as_ref()
                .and_then(|profile| profile.functions.get(entry))
                .cloned()
                .unwrap_or_default();
            candidates.insert(*entry, Candidate::new(profile));
        }
        Ok(Self {
            plans,
            variant_tags,
            candidates,
            compiler,
            coefficients,
            profile_key,
            profile_path: profile_path.map(Path::to_path_buf),
            bootstrap_probes: 0,
            compile_tokens_ns: TOKEN_CAPACITY_NS as i64,
            token_refill_at: Instant::now(),
        })
    }

    pub(crate) fn contains(&self, entry: usize, target: usize) -> bool {
        self.plans
            .get(&entry)
            .is_some_and(|plan| plan.osr_entries.binary_search(&target).is_ok())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_osr(
        &mut self,
        artifact: &CompiledProgram,
        entry: usize,
        target: usize,
        pc: &mut usize,
        locals: &mut Vec<Option<Value>>,
        bp: usize,
        stack: &mut Vec<Value>,
        operand_base: usize,
        captures: &crate::interpreter::CaptureValues,
        propagation_pending: &mut bool,
        debug: bool,
    ) -> Option<JitExit> {
        let plan = Arc::clone(self.plans.get(&entry)?);
        if artifact.callables.get(&entry)?.captures != captures.len() {
            return None;
        }
        if !self.contains(entry, target) {
            return None;
        }
        let remaining = stack.len().checked_sub(operand_base)?;
        let expected_depth = plan
            .stack_depths
            .get(target.checked_sub(plan.entry)?)?
            .as_ref()
            .copied()?;
        if expected_depth != 0 || remaining != expected_depth {
            self.candidates.get_mut(&entry)?.disabled = true;
            if debug {
                eprintln!("[jit] ineligible function={entry} reason=nonempty-osr-stack");
            }
            return None;
        }
        let now = Instant::now();
        let window_closed = {
            let candidate = self.candidates.get_mut(&entry)?;
            if candidate.disabled {
                return None;
            }
            update_heat(candidate, now)
        };
        if debug
            && window_closed
            && let Some(candidate) = self.candidates.get(&entry)
        {
            eprintln!(
                "[jit] profile function={entry} backedges={} active_windows={} vm_ns_per_backedge={}",
                candidate.profile.total_backedges,
                candidate.profile.active_windows,
                candidate.profile.vm_ns_per_backedge
            );
        }
        self.refill_tokens(now);
        if self.candidates.get(&entry).is_some_and(|candidate| {
            candidate.compiled.is_some()
                && candidate.profile.total_backedges < candidate.retry_after_backedges
        }) {
            if debug && window_closed {
                eprintln!("[jit] decision function={entry} reason=fallback-cooldown");
            }
            return None;
        }

        if self
            .candidates
            .get(&entry)
            .is_some_and(|candidate| candidate.compiled.is_none())
        {
            let (bootstrap_probe, priority, estimated_cost) = match self.candidate_decision(entry) {
                ApeDecision::BootstrapCompile {
                    priority,
                    estimated_cost_ns,
                } => (true, priority, estimated_cost_ns),
                ApeDecision::Compile {
                    priority,
                    estimated_cost_ns,
                } => (false, priority, estimated_cost_ns),
                decision => {
                    if debug && window_closed {
                        log_ape_decision(entry, decision);
                    }
                    return None;
                }
            };
            let selection_window = self.candidates.get(&entry)?.windows_this_run;
            if self.candidates.get(&entry)?.last_selection_window == selection_window {
                return None;
            }
            let best_candidate = self.best_candidate(now);
            self.candidates.get_mut(&entry)?.last_selection_window = selection_window;
            if best_candidate.is_some_and(|best_entry| best_entry != entry) {
                if debug && window_closed {
                    log_ape_decision(
                        entry,
                        ApeDecision::Profile("higher-priority-candidate-active"),
                    );
                }
                return None;
            }
            if self.compiler.functions >= MAX_JIT_FUNCTIONS
                || self.compiler.cache_full.load(Ordering::Relaxed)
                || self.compiler.code_bytes.load(Ordering::Relaxed) >= MAX_JIT_CODE_BYTES
            {
                if debug && window_closed {
                    log_ape_decision(entry, ApeDecision::SkipCacheFull);
                }
                return None;
            }

            // An over-capacity compile reserves a full bucket and creates
            // debt when its measured cost is settled below.
            let reserved_cost =
                match reserve_compile_budget(&mut self.compile_tokens_ns, estimated_cost) {
                    Ok(reserved) => reserved,
                    Err(available_ns) => {
                        if debug && window_closed {
                            let decision = ApeDecision::DeferBudget {
                                estimated_cost_ns: estimated_cost,
                                available_ns,
                            };
                            log_ape_decision(entry, decision);
                        }
                        return None;
                    }
                };
            let started = Instant::now();
            match self.compiler.compile(&plan, artifact, &self.variant_tags) {
                Ok(function) => {
                    let actual = nanos(started.elapsed());
                    self.settle_compile_budget(reserved_cost, actual);
                    self.coefficients.update(&plan, actual);
                    if bootstrap_probe {
                        self.bootstrap_probes += 1;
                    }
                    let candidate = self.candidates.get_mut(&entry)?;
                    candidate.profile.compile_ns = actual;
                    candidate.compiled = Some(CompiledFunction {
                        entry: function,
                        // Measure this function's first native slice even when
                        // adaptive APE used a conservative cross-function rate.
                        probe_pending: true,
                        call_compiled: false,
                    });
                    if debug {
                        eprintln!(
                            "[jit] {} function={entry} priority={priority:.3} compile_ns={actual} estimated_upper_ns={estimated_cost}",
                            if bootstrap_probe {
                                "bootstrap-compile"
                            } else {
                                "compile"
                            }
                        );
                    }
                    self.persist();
                }
                Err(error) => {
                    let actual = nanos(started.elapsed());
                    self.settle_compile_budget(reserved_cost, actual);
                    self.coefficients.update(&plan, actual);
                    if self.compiler.cache_full.load(Ordering::Relaxed) {
                        if debug {
                            log_ape_decision(entry, ApeDecision::SkipCacheFull);
                        }
                        return None;
                    }
                    let candidate = self.candidates.get_mut(&entry)?;
                    candidate.disabled = true;
                    if debug {
                        eprintln!("[jit] jit-failed function={entry} error={error}");
                    }
                    self.persist();
                    return None;
                }
            }
        }

        let (entry_function, probe_pending) = {
            let candidate = self.candidates.get(&entry)?;
            let compiled = candidate.compiled.as_ref()?;
            if candidate.profile.calibrated && adaptive_gain(candidate) <= 0.0 {
                return None;
            }
            (compiled.entry, compiled.probe_pending)
        };
        // OSR timings include VM/JIT entry costs; they do not predict whether
        // calling this compiled function from another JIT function is useful.
        let callables = self
            .candidates
            .iter()
            .filter_map(|(&candidate_entry, candidate)| {
                let compiled = candidate.compiled.as_ref()?;
                let plan = Arc::clone(self.plans.get(&candidate_entry)?);
                let call_profiled = compiled.call_compiled
                    && candidate.profile.total_calls >= HOT_CALL_THRESHOLD
                    && candidate.profile.total_calls >= candidate.retry_after_calls;
                (plan.callable_from_jit
                    && !compiled.probe_pending
                    && !candidate.disabled
                    && (candidate.profile.calibrated || call_profiled)
                    && candidate.profile.total_backedges >= candidate.retry_after_backedges)
                    .then_some((
                        candidate_entry,
                        JitCallable {
                            entry: compiled.entry,
                            plan,
                        },
                    ))
            })
            .collect();
        let mut runtime = JitExecutionContext {
            artifact,
            variant_tags: &self.variant_tags,
            callables,
            call_frame: None,
        };
        let runtime_context = (&mut runtime as *mut JitExecutionContext<'_>).cast();
        let mut roots = Vec::new();
        let mut jit_locals = vec![JitSlot::default(); plan.local_count];
        for (index, slot) in jit_locals.iter_mut().enumerate() {
            if let Some(value) = locals.get(bp + index).and_then(Option::as_ref) {
                *slot = to_slot(value, &mut roots, &self.variant_tags);
            }
        }
        if debug && probe_pending {
            eprintln!("[jit] osr function={entry} pc={target} locals={jit_locals:?}");
        }
        let mut jit_captures = Vec::with_capacity(captures.len());
        for index in 0..captures.len() {
            jit_captures.push(to_slot(
                &captures.get(index)?,
                &mut roots,
                &self.variant_tags,
            ));
        }
        let mut jit_stack = vec![JitSlot::default(); plan.max_stack.max(remaining)];
        for (index, value) in stack[operand_base..].iter().enumerate() {
            jit_stack[index] = to_slot(value, &mut roots, &self.variant_tags);
        }
        let mut frame = JitFrame {
            locals: jit_locals.as_mut_ptr(),
            stack: jit_stack.as_mut_ptr(),
            captures: jit_captures.as_mut_ptr(),
            capture_count: jit_captures.len() as u64,
            roots: &mut roots,
            runtime_context,
            pc: target as u64,
            stack_len: remaining as u64,
            propagation_pending: u64::from(*propagation_pending),
            backedge_budget: if probe_pending { WINDOW_BACKEDGES } else { 0 },
            backedge_count: 0,
            exit_reason: REASON_NORMAL,
        };
        let started = Instant::now();
        // The entry pointer remains owned by compiler.module for this plan's lifetime.
        let status = unsafe { (entry_function)(&mut frame) };
        let elapsed = nanos(started.elapsed());
        drop(runtime);
        let exit = match decode_exit(&plan, artifact, &frame, status) {
            Ok(exit) => exit,
            Err(()) => {
                if let Some(candidate) = self.candidates.get_mut(&entry) {
                    candidate.disabled = true;
                }
                if debug {
                    eprintln!(
                        "[jit] fallback function={entry} reason=invalid-exit status={status} pc={} depth={} reason={}",
                        frame.pc, frame.stack_len, frame.exit_reason
                    );
                }
                *pc = target;
                return Some(JitExit::ResumeAt { pc: target });
            }
        };
        if restore_frame(
            &plan,
            &frame,
            &jit_locals,
            &jit_stack,
            &roots,
            &self.variant_tags,
            locals,
            bp,
            stack,
            operand_base,
        )
        .is_err()
        {
            if let Some(candidate) = self.candidates.get_mut(&entry) {
                candidate.disabled = true;
            }
            if debug {
                eprintln!("[jit] fallback function={entry} reason=invalid-returned-frame");
            }
            // The JIT only mutates its private copy of the frame. Resume from
            // the original OSR point and let the VM execute the loop again.
            *pc = target;
            return Some(JitExit::ResumeAt { pc: target });
        }
        *pc = frame.pc as usize;
        *propagation_pending = frame.propagation_pending != 0;

        let backedges = frame.backedge_count;
        if debug
            && (probe_pending
                || frame.exit_reason != REASON_NORMAL
                || !matches!(exit, JitExit::Completed))
        {
            eprintln!(
                "[jit] exit function={entry} exit={exit:?} backedges={backedges} budget={} reason={} pc={} locals={jit_locals:?}",
                frame.backedge_budget, frame.exit_reason, frame.pc
            );
        }
        let complete_probe = probe_pending
            && backedges > 0
            && (frame.exit_reason == REASON_PROFILE_SLICE || matches!(exit, JitExit::Completed));
        let candidate = self.candidates.get_mut(&entry)?;
        if complete_probe {
            if let Some(compiled) = candidate.compiled.as_mut() {
                compiled.probe_pending = false;
            }
            candidate.profile.jit_ns_per_backedge = elapsed.saturating_div(backedges).max(1);
            candidate.profile.calibrated = true;
            let vm_ns = candidate.profile.vm_ns_per_backedge;
            if adaptive_gain(candidate) <= 0.0 {
                if debug {
                    eprintln!(
                        "[jit] calibrated-unprofitable function={entry} vm_ns_per_backedge={vm_ns} jit_ns_per_backedge={}",
                        candidate.profile.jit_ns_per_backedge
                    );
                }
            } else if debug {
                eprintln!(
                    "[jit] calibrated function={entry} vm_ns_per_backedge={vm_ns} jit_ns_per_backedge={}",
                    candidate.profile.jit_ns_per_backedge
                );
            }
            self.persist();
        } else if matches!(exit, JitExit::ResumeAt { .. })
            && frame.exit_reason != REASON_PROFILE_SLICE
        {
            candidate.retry_after_backedges = candidate
                .profile
                .total_backedges
                .saturating_add(WINDOW_BACKEDGES * u64::from(REQUIRED_WINDOWS));
            if debug {
                eprintln!(
                    "[jit] fallback function={entry} exit={exit:?}; retry_after_backedges={}",
                    candidate.retry_after_backedges
                );
            }
        } else if matches!(exit, JitExit::FaultAt { .. }) {
            candidate.retry_after_backedges = candidate
                .profile
                .total_backedges
                .saturating_add(WINDOW_BACKEDGES);
            if debug {
                eprintln!(
                    "[jit] fault function={entry} exit={exit:?}; retry_after_backedges={}",
                    candidate.retry_after_backedges
                );
            }
        }
        Some(exit)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_call(
        &mut self,
        artifact: &CompiledProgram,
        entry: usize,
        pc: &mut usize,
        locals: &mut Vec<Option<Value>>,
        bp: usize,
        stack: &mut Vec<Value>,
        operand_base: usize,
        captures: &crate::interpreter::CaptureValues,
        propagation_pending: &mut bool,
        debug: bool,
    ) -> Option<JitExit> {
        let plan = Arc::clone(self.plans.get(&entry)?);
        let signature = artifact.callables.get(&entry)?;
        if plan.has_backedges
            || !plan.callable_from_jit
            || signature.asynchronous
            || signature.captures != 0
            || captures.len() != 0
        {
            if debug {
                eprintln!(
                    "[jit] call-profile-skip function={entry} backedges={} async={} captures={}",
                    plan.has_backedges, signature.asynchronous, signature.captures
                );
            }
            return None;
        }
        let remaining = stack.len().checked_sub(operand_base)?;
        if plan.stack_depths.first().copied().flatten()? != remaining {
            return None;
        }

        let now = Instant::now();
        self.refill_tokens(now);
        {
            let candidate = self.candidates.get_mut(&entry)?;
            candidate.profile.total_calls = candidate.profile.total_calls.saturating_add(1);
            if debug
                && (candidate.profile.total_calls == 1
                    || candidate.profile.total_calls % HOT_CALL_THRESHOLD == 0)
            {
                eprintln!(
                    "[jit] call-profile function={entry} calls={}",
                    candidate.profile.total_calls
                );
            }
            if candidate.disabled || candidate.profile.total_calls < candidate.retry_after_calls {
                return None;
            }
        }

        if self
            .candidates
            .get(&entry)
            .is_some_and(|candidate| candidate.compiled.is_none())
        {
            let candidate = self.candidates.get(&entry)?;
            if candidate.profile.total_calls < HOT_CALL_THRESHOLD
                || candidate
                    .profile
                    .total_calls
                    .saturating_sub(candidate.last_selection_call)
                    < HOT_CALL_THRESHOLD
            {
                return None;
            }
            let calls = candidate.profile.total_calls;
            if let Some(candidate) = self.candidates.get_mut(&entry) {
                candidate.last_selection_call = calls;
            }
            if self.compiler.functions >= MAX_JIT_FUNCTIONS
                || self.compiler.cache_full.load(Ordering::Relaxed)
                || self.compiler.code_bytes.load(Ordering::Relaxed) >= MAX_JIT_CODE_BYTES
            {
                if debug {
                    log_ape_decision(entry, ApeDecision::SkipCacheFull);
                }
                return None;
            }
            let estimated_cost = plan.upper_compile_ns(&self.coefficients);
            let reserved_cost =
                match reserve_compile_budget(&mut self.compile_tokens_ns, estimated_cost) {
                    Ok(reserved) => reserved,
                    Err(available_ns) => {
                        if debug {
                            log_ape_decision(
                                entry,
                                ApeDecision::DeferBudget {
                                    estimated_cost_ns: estimated_cost,
                                    available_ns,
                                },
                            );
                        }
                        return None;
                    }
                };
            let started = Instant::now();
            match self.compiler.compile(&plan, artifact, &self.variant_tags) {
                Ok(function) => {
                    let actual = nanos(started.elapsed());
                    self.settle_compile_budget(reserved_cost, actual);
                    self.coefficients.update(&plan, actual);
                    let candidate = self.candidates.get_mut(&entry)?;
                    candidate.profile.compile_ns = actual;
                    candidate.compiled = Some(CompiledFunction {
                        entry: function,
                        probe_pending: false,
                        call_compiled: true,
                    });
                    if debug {
                        eprintln!(
                            "[jit] call-compile function={entry} calls={} compile_ns={actual} estimated_compile_ns={estimated_cost}",
                            candidate.profile.total_calls
                        );
                    }
                    self.persist();
                }
                Err(error) => {
                    let actual = nanos(started.elapsed());
                    self.settle_compile_budget(reserved_cost, actual);
                    self.coefficients.update(&plan, actual);
                    if !self.compiler.cache_full.load(Ordering::Relaxed) {
                        if let Some(candidate) = self.candidates.get_mut(&entry) {
                            candidate.disabled = true;
                        }
                        if debug {
                            eprintln!("[jit] call-compile-failed function={entry} error={error}");
                        }
                    }
                    self.persist();
                    return None;
                }
            }
        }

        let entry_function = {
            let candidate = self.candidates.get(&entry)?;
            if candidate.profile.total_calls < candidate.retry_after_calls {
                return None;
            }
            let compiled = candidate.compiled.as_ref()?;
            compiled.entry
        };
        let mut runtime = JitExecutionContext {
            artifact,
            variant_tags: &self.variant_tags,
            callables: HashMap::new(),
            call_frame: None,
        };
        let runtime_context = (&mut runtime as *mut JitExecutionContext<'_>).cast();
        let mut roots = Vec::new();
        let mut jit_locals = vec![JitSlot::default(); plan.local_count];
        for (index, slot) in jit_locals.iter_mut().enumerate() {
            if let Some(value) = locals.get(bp + index).and_then(Option::as_ref) {
                *slot = to_slot(value, &mut roots, &self.variant_tags);
            }
        }
        let mut jit_stack = vec![JitSlot::default(); plan.max_stack.max(remaining)];
        for (index, value) in stack[operand_base..].iter().enumerate() {
            jit_stack[index] = to_slot(value, &mut roots, &self.variant_tags);
        }
        let mut frame = JitFrame {
            locals: jit_locals.as_mut_ptr(),
            stack: jit_stack.as_mut_ptr(),
            captures: std::ptr::null_mut(),
            capture_count: 0,
            roots: &mut roots,
            runtime_context,
            pc: entry as u64,
            stack_len: remaining as u64,
            propagation_pending: u64::from(*propagation_pending),
            backedge_budget: 0,
            backedge_count: 0,
            exit_reason: REASON_NORMAL,
        };
        let status = unsafe { (entry_function)(&mut frame) };
        drop(runtime);
        let exit = match decode_exit(&plan, artifact, &frame, status) {
            Ok(exit) => exit,
            Err(()) => {
                if let Some(candidate) = self.candidates.get_mut(&entry) {
                    candidate.disabled = true;
                }
                *pc = entry;
                if debug {
                    eprintln!("[jit] call-exit-invalid function={entry}; resuming interpreter");
                }
                return Some(JitExit::ResumeAt { pc: entry });
            }
        };
        if restore_frame(
            &plan,
            &frame,
            &jit_locals,
            &jit_stack,
            &roots,
            &self.variant_tags,
            locals,
            bp,
            stack,
            operand_base,
        )
        .is_err()
        {
            if let Some(candidate) = self.candidates.get_mut(&entry) {
                candidate.disabled = true;
            }
            *pc = entry;
            if debug {
                eprintln!("[jit] call-frame-invalid function={entry}; resuming interpreter");
            }
            return Some(JitExit::ResumeAt { pc: entry });
        }
        *pc = frame.pc as usize;
        *propagation_pending = frame.propagation_pending != 0;
        if matches!(exit, JitExit::ResumeAt { .. } | JitExit::FaultAt { .. })
            && let Some(candidate) = self.candidates.get_mut(&entry)
        {
            candidate.retry_after_calls = candidate
                .profile
                .total_calls
                .saturating_add(HOT_CALL_THRESHOLD);
            if debug {
                eprintln!(
                    "[jit] call-fallback function={entry} exit={exit:?}; retry_after_calls={}",
                    candidate.retry_after_calls
                );
            }
        }
        Some(exit)
    }

    fn candidate_decision(&self, entry: usize) -> ApeDecision {
        let Some(candidate) = self.candidates.get(&entry) else {
            return ApeDecision::Ineligible("missing-candidate");
        };
        if candidate.disabled || candidate.compiled.is_some() {
            return ApeDecision::Ineligible("disabled-or-compiled");
        }
        let Some(plan) = self.plans.get(&entry) else {
            return ApeDecision::Ineligible("missing-plan");
        };
        if plan.instruction_count > MAX_FUNCTION_INSTRUCTIONS {
            return ApeDecision::Ineligible("function-too-large");
        }
        if candidate.windows_this_run < REQUIRED_WINDOWS
            || candidate.profile.active_windows < REQUIRED_WINDOWS
            || candidate.profile.total_backedges < WINDOW_BACKEDGES * u64::from(REQUIRED_WINDOWS)
        {
            return ApeDecision::Profile("warming");
        }

        let estimate = plan.upper_compile_ns(&self.coefficients);
        if !candidate.profile.calibrated && self.bootstrap_probes < MAX_BOOTSTRAP_PROBES {
            let priority = candidate.profile.total_backedges as f64 / estimate.max(1) as f64;
            return ApeDecision::BootstrapCompile {
                priority,
                estimated_cost_ns: estimate,
            };
        }

        let predicted_jit_ns = if candidate.profile.calibrated {
            candidate.profile.jit_ns_per_backedge
        } else {
            let Some(estimate) =
                self.conservative_jit_estimate(candidate.profile.vm_ns_per_backedge)
            else {
                return ApeDecision::Profile("awaiting-shared-calibration");
            };
            estimate
        };
        let compile_upper_ns = if candidate.profile.compile_ns > 0 {
            candidate.profile.compile_ns.saturating_mul(3).div_ceil(2)
        } else {
            estimate
        };
        let gain = adaptive_gain_at(candidate, predicted_jit_ns, compile_upper_ns);
        if gain <= 0.0 {
            return ApeDecision::Profile("not-profitable");
        }
        ApeDecision::Compile {
            priority: gain / estimate.max(1) as f64,
            estimated_cost_ns: estimate,
        }
    }

    fn best_candidate(&self, now: Instant) -> Option<usize> {
        if self.compiler.functions >= MAX_JIT_FUNCTIONS
            || self.compiler.cache_full.load(Ordering::Relaxed)
            || self.compiler.code_bytes.load(Ordering::Relaxed) >= MAX_JIT_CODE_BYTES
        {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for (&entry, candidate) in &self.candidates {
            if candidate
                .last_backedge_at
                .is_none_or(|last| nanos(now.duration_since(last)) > MAX_ACTIVITY_GAP_NS)
            {
                continue;
            }
            let (priority, estimated_cost_ns) = match self.candidate_decision(entry) {
                ApeDecision::BootstrapCompile {
                    priority,
                    estimated_cost_ns,
                }
                | ApeDecision::Compile {
                    priority,
                    estimated_cost_ns,
                } => (priority, estimated_cost_ns),
                _ => continue,
            };
            if self.compile_tokens_ns < estimated_cost_ns.min(TOKEN_CAPACITY_NS) as i64 {
                continue;
            }
            if best.is_none_or(|(best_entry, best_priority)| {
                priority > best_priority || (priority == best_priority && entry < best_entry)
            }) {
                best = Some((entry, priority));
            }
        }
        best.map(|(entry, _)| entry)
    }

    fn conservative_jit_estimate(&self, vm_ns: u64) -> Option<u64> {
        if vm_ns == 0 {
            return None;
        }
        let conservative_speedup = self
            .candidates
            .values()
            .filter_map(|candidate| {
                let profile = &candidate.profile;
                (profile.calibrated
                    && profile.vm_ns_per_backedge > 0
                    && profile.jit_ns_per_backedge > 0)
                    .then_some(
                        profile.vm_ns_per_backedge as f64 / profile.jit_ns_per_backedge as f64,
                    )
            })
            .reduce(f64::min)?;
        if !conservative_speedup.is_finite() || conservative_speedup <= 1.0 {
            return None;
        }
        Some((vm_ns as f64 / conservative_speedup).ceil().max(1.0) as u64)
    }

    fn refill_tokens(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.token_refill_at).as_nanos() as f64;
        let refill = (elapsed * TOKEN_REFILL_PER_NS).min(u64::MAX as f64) as u64;
        self.compile_tokens_ns = self
            .compile_tokens_ns
            .saturating_add(refill.min(i64::MAX as u64) as i64)
            .min(TOKEN_CAPACITY_NS as i64);
        self.token_refill_at = now;
    }

    fn settle_compile_budget(&mut self, reserved_ns: u64, actual_ns: u64) {
        settle_compile_budget(&mut self.compile_tokens_ns, reserved_ns, actual_ns);
    }

    fn persist(&self) {
        let Some(path) = &self.profile_path else {
            return;
        };
        let profile = ProfileFile {
            key: self.profile_key.clone(),
            coefficients: Some(self.coefficients.clone()),
            functions: self
                .candidates
                .iter()
                .map(|(&entry, candidate)| (entry, candidate.profile.clone()))
                .collect(),
        };
        // Profiling is advisory: filesystem errors never affect VM execution.
        let _ = save_profile(path, &profile);
    }
}

fn adaptive_gain(candidate: &Candidate) -> f64 {
    let compile_upper_ns = candidate
        .profile
        .compile_ns
        .max(1)
        .saturating_mul(3)
        .div_ceil(2);
    adaptive_gain_at(
        candidate,
        candidate.profile.jit_ns_per_backedge,
        compile_upper_ns,
    )
}

fn adaptive_gain_at(candidate: &Candidate, jit_ns_per_backedge: u64, compile_upper_ns: u64) -> f64 {
    let vm = candidate.profile.vm_ns_per_backedge;
    let jit = jit_ns_per_backedge;
    if vm == 0 || jit == 0 || jit >= vm {
        return 0.0;
    }
    let horizon_ns = candidate.profile.hot_series_ns.clamp(1, MAX_HORIZON_NS);
    let rate = (1_000_000_000u64 / vm).max(1) / 2;
    let future_edges =
        ((rate as u128 * horizon_ns as u128) / 1_000_000_000).min(u64::MAX as u128) as u64;
    let savings = future_edges.saturating_mul(vm.saturating_sub(jit));
    savings.saturating_sub(compile_upper_ns) as f64
}

fn reserve_compile_budget(available_ns: &mut i64, estimate_ns: u64) -> Result<u64, i64> {
    let reserved_ns = estimate_ns.min(TOKEN_CAPACITY_NS);
    if *available_ns < reserved_ns as i64 {
        return Err(*available_ns);
    }
    *available_ns -= reserved_ns as i64;
    Ok(reserved_ns)
}

fn settle_compile_budget(available_ns: &mut i64, reserved_ns: u64, actual_ns: u64) {
    let adjustment = i128::from(reserved_ns) - i128::from(actual_ns);
    *available_ns = (i128::from(*available_ns) + adjustment)
        .clamp(i128::from(i64::MIN), i128::from(TOKEN_CAPACITY_NS as i64))
        as i64;
}

fn update_heat(candidate: &mut Candidate, now: Instant) -> bool {
    if candidate
        .last_backedge_at
        .is_some_and(|last| nanos(now.duration_since(last)) > MAX_ACTIVITY_GAP_NS)
    {
        candidate.window_edges = 0;
        candidate.windows_this_run = 0;
        candidate.last_selection_window = 0;
        candidate.profile.active_windows = 0;
        candidate.profile.hot_series_ns = 0;
    }
    candidate.last_backedge_at = Some(now);
    if candidate.window_edges == 0 {
        candidate.window_started = now;
    }
    candidate.window_edges += 1;
    candidate.profile.total_backedges = candidate.profile.total_backedges.saturating_add(1);
    if candidate.window_edges < WINDOW_BACKEDGES {
        return false;
    }
    let elapsed = nanos(now.duration_since(candidate.window_started));
    let measured = elapsed.saturating_div(WINDOW_BACKEDGES).max(1);
    candidate.profile.vm_ns_per_backedge = if candidate.profile.vm_ns_per_backedge == 0 {
        measured
    } else {
        measured
            .saturating_mul(3)
            .saturating_add(candidate.profile.vm_ns_per_backedge.saturating_mul(7))
            / 10
    };
    candidate.profile.hot_series_ns = candidate.profile.hot_series_ns.saturating_add(elapsed);
    candidate.profile.active_windows = candidate.profile.active_windows.saturating_add(1);
    candidate.windows_this_run = candidate.windows_this_run.saturating_add(1);
    candidate.window_edges = 0;
    candidate.window_started = now;
    true
}

fn analyze_program(artifact: &CompiledProgram) -> BTreeMap<usize, Arc<FunctionPlan>> {
    let entries: Vec<usize> = artifact.callables.keys().copied().collect();
    let mut plans = BTreeMap::new();
    for (index, entry) in entries.iter().copied().enumerate() {
        let signature = &artifact.callables[&entry];
        let end = entries
            .get(index + 1)
            .copied()
            .unwrap_or(artifact.instructions.len());
        if signature.asynchronous || entry >= end {
            continue;
        }
        if let Some(plan) = analyze_function(artifact, entry, end, signature.parameters) {
            plans.insert(entry, Arc::new(plan));
        }
    }
    plans
}

fn analyze_function(
    artifact: &CompiledProgram,
    entry: usize,
    end: usize,
    parameters: usize,
) -> Option<FunctionPlan> {
    let count = end.checked_sub(entry)?;
    if count == 0 || count > MAX_FUNCTION_INSTRUCTIONS {
        return None;
    }
    let mut local_count = parameters;
    let mut backedges = Vec::new();
    let mut blocks = std::collections::BTreeSet::from([entry]);
    let mut edge_count = 0usize;
    for pc in entry..end {
        let instruction = artifact.instructions.get(pc)?;
        if let Instruction::Load(index) | Instruction::Store(index) = instruction {
            local_count = local_count.max(index.checked_add(1)?);
        }
        if local_count > MAX_LOCALS {
            return None;
        }
        match instruction {
            Instruction::Jump(target) => {
                if !(entry..end).contains(target) {
                    return None;
                }
                blocks.insert(*target);
                edge_count += 1;
                if *target <= pc {
                    backedges.push((pc, *target));
                }
            }
            Instruction::JumpIfFalse(target) | Instruction::JumpIfTrue(target) => {
                if !(entry..end).contains(target) || pc + 1 >= end {
                    return None;
                }
                blocks.insert(*target);
                blocks.insert(pc + 1);
                edge_count += 2;
                if *target <= pc {
                    backedges.push((pc, *target));
                }
            }
            Instruction::JumpIfNoPropagation(target) => {
                if !(entry..end).contains(target) || pc + 1 >= end {
                    return None;
                }
                blocks.insert(*target);
                blocks.insert(pc + 1);
                edge_count += 2;
            }
            _ => {}
        }
    }
    let has_backedges = !backedges.is_empty();

    let mut depths = vec![None; count];
    depths[0] = Some(parameters);
    let mut max_stack = parameters;
    let mut queue = VecDeque::from([entry]);
    while let Some(pc) = queue.pop_front() {
        let depth = depths[pc - entry]?;
        let instruction = artifact.instructions.get(pc)?;
        let (required, produced) = instruction.stack_effect(&artifact.imports).ok()?;
        if depth < required {
            return None;
        }
        let next_depth = match instruction {
            Instruction::StackTruncate(target) => *target,
            _ => depth - required + produced,
        };
        max_stack = max_stack.max(next_depth);
        if max_stack > MAX_STACK {
            return None;
        }
        let successors = match instruction {
            Instruction::Jump(target) => [Some(*target), None],
            Instruction::JumpIfFalse(target)
            | Instruction::JumpIfTrue(target)
            | Instruction::JumpIfNoPropagation(target) => [Some(*target), Some(pc + 1)],
            Instruction::Return
            | Instruction::ReturnPropagate
            | Instruction::Crash(_)
            | Instruction::Halt => [None, None],
            _ if pc + 1 < end => [Some(pc + 1), None],
            _ => return None,
        };
        for successor in successors.into_iter().flatten() {
            if !(entry..end).contains(&successor) {
                return None;
            }
            let slot = &mut depths[successor - entry];
            match *slot {
                Some(existing) if existing != next_depth => return None,
                Some(_) => {}
                None => {
                    *slot = Some(next_depth);
                    queue.push_back(successor);
                }
            }
        }
    }
    let mut predecessors = HashMap::<usize, usize>::new();
    for pc in entry..end {
        if depths[pc - entry].is_none() {
            continue;
        }
        match &artifact.instructions[pc] {
            Instruction::Jump(target) => *predecessors.entry(*target).or_default() += 1,
            Instruction::JumpIfFalse(target)
            | Instruction::JumpIfTrue(target)
            | Instruction::JumpIfNoPropagation(target) => {
                *predecessors.entry(*target).or_default() += 1;
                *predecessors.entry(pc + 1).or_default() += 1;
            }
            Instruction::Return
            | Instruction::ReturnPropagate
            | Instruction::Crash(_)
            | Instruction::Halt => {}
            _ => *predecessors.entry(pc + 1).or_default() += 1,
        }
    }
    let phi_estimate = blocks
        .iter()
        .filter(|pc| predecessors.get(pc).copied().unwrap_or_default() > 1)
        .map(|pc| depths[pc - entry].unwrap_or_default() + local_count)
        .sum();
    let mut osr_entries: Vec<usize> = backedges
        .into_iter()
        .map(|(_, target)| target)
        .filter(|target| {
            depths
                .get(target - entry)
                .is_some_and(|depth| *depth == Some(0))
        })
        .collect();
    osr_entries.sort_unstable();
    osr_entries.dedup();
    let callable_from_jit = (entry..end).all(|pc| {
        let instruction = &artifact.instructions[pc];
        matches!(
            instruction,
            Instruction::Push(_)
                | Instruction::PushArray(_)
                | Instruction::Load(_)
                | Instruction::Store(_)
                | Instruction::Add
                | Instruction::Sub
                | Instruction::Mul
                | Instruction::Div
                | Instruction::Mod
                | Instruction::Eq
                | Instruction::Ne
                | Instruction::Lt
                | Instruction::Le
                | Instruction::Gt
                | Instruction::Ge
                | Instruction::Jump(_)
                | Instruction::JumpIfFalse(_)
                | Instruction::JumpIfTrue(_)
                | Instruction::JumpIfNoPropagation(_)
                | Instruction::IsType(_)
                | Instruction::And
                | Instruction::Or
                | Instruction::Xor
                | Instruction::Not
                | Instruction::Dup
                | Instruction::Pop
                | Instruction::Swap
                | Instruction::StackTruncate(_)
                | Instruction::MakeRecord(_)
                | Instruction::MakeArray(_)
                | Instruction::MakeTuple(_)
                | Instruction::MakeTypedTuple(_, _)
                | Instruction::MakeTypedArray(_, _)
                // Scalar values pass through in the JIT; arrays resume in the
                // VM at this instruction, which keeps the direct-call path safe.
                | Instruction::ConstrainArrayFrom
                | Instruction::Return
                | Instruction::Crash(_)
        ) || matches!(instruction, Instruction::CallNative(index)
            if artifact.imports.get(*index).is_some_and(|import| import.name() == "str::to_bigint"))
    });
    Some(FunctionPlan {
        entry,
        end,
        instruction_count: count,
        block_count: blocks.len(),
        edge_count,
        phi_estimate,
        local_count,
        max_stack,
        stack_depths: depths,
        osr_entries,
        has_backedges,
        callable_from_jit,
    })
}

fn nanos(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

struct JitCompiler {
    module: JITModule,
    code_bytes: Arc<AtomicUsize>,
    cache_full: Arc<AtomicBool>,
    functions: usize,
    pointer_type: cranelift_codegen::ir::Type,
    helpers: JitHelpers,
}

#[derive(Clone, Copy)]
struct JitHelpers {
    bigint_binary: FuncId,
    bigint_compare: FuncId,
    direct_call: FuncId,
    native_call: FuncId,
}

#[derive(Clone, Copy)]
struct JitHelperRefs {
    bigint_binary: FuncRef,
    bigint_compare: FuncRef,
    direct_call: FuncRef,
    native_call: FuncRef,
}

impl JitCompiler {
    fn new() -> Result<(Self, String), String> {
        let mut settings_builder = settings::builder();
        settings_builder
            .set("opt_level", "speed")
            .map_err(|error| error.to_string())?;
        let isa = cranelift_native::builder()
            .map_err(|error| error.to_string())?
            .finish(settings::Flags::new(settings_builder))
            .map_err(|error| error.to_string())?;
        let pointer_type = isa.pointer_type();
        if pointer_type != types::I64 {
            return Err("Cog JIT currently requires a 64-bit host".into());
        }
        let host = format!(
            "{}|{}",
            isa.triple(),
            isa.isa_flags()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let code_bytes = Arc::new(AtomicUsize::new(0));
        let cache_full = Arc::new(AtomicBool::new(false));
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        builder.symbol("cog_jit_bigint_binary", jit_bigint_binary as *const u8);
        builder.symbol("cog_jit_bigint_compare", jit_bigint_compare as *const u8);
        builder.symbol("cog_jit_direct_call", jit_direct_call as *const u8);
        builder.symbol("cog_jit_native_call", jit_native_call as *const u8);
        builder.memory_provider(Box::new(CountingMemoryProvider {
            inner: SystemMemoryProvider::new(),
            code_bytes: Arc::clone(&code_bytes),
            cache_full: Arc::clone(&cache_full),
            limit: MAX_JIT_CODE_BYTES,
        }));
        let mut module = JITModule::new(builder);
        let helpers = JitHelpers {
            bigint_binary: declare_helper(
                &mut module,
                "cog_jit_bigint_binary",
                &[types::I64, types::I64, types::I32],
            )?,
            bigint_compare: declare_helper(
                &mut module,
                "cog_jit_bigint_compare",
                &[types::I64, types::I64, types::I32],
            )?,
            direct_call: declare_helper(
                &mut module,
                "cog_jit_direct_call",
                &[types::I64, types::I64, types::I64, types::I64, types::I64],
            )?,
            native_call: declare_helper(
                &mut module,
                "cog_jit_native_call",
                &[types::I64, types::I64, types::I64, types::I64],
            )?,
        };
        Ok((
            Self {
                module,
                code_bytes,
                cache_full,
                functions: 0,
                pointer_type,
                helpers,
            },
            host,
        ))
    }

    fn compile(
        &mut self,
        plan: &FunctionPlan,
        artifact: &CompiledProgram,
        variant_tags: &VariantTags,
    ) -> Result<JitEntry, String> {
        if self.functions >= MAX_JIT_FUNCTIONS {
            return Err("JIT function cache is full".into());
        }
        let function_id = self
            .module
            .declare_function(
                &format!("cog_jit_{}", plan.entry),
                Linkage::Local,
                &self.signature(),
            )
            .map_err(|error| error.to_string())?;
        let mut context = self.module.make_context();
        context.func.signature = self.signature();
        context.func.name = UserFuncName::user(0, function_id.as_u32());
        let mut builder_context = FunctionBuilderContext::new();
        lower_function(
            &mut self.module,
            &mut context,
            &mut builder_context,
            plan,
            artifact,
            variant_tags,
            self.helpers,
        )?;
        // These environment switches keep expensive diagnostics opt-in and
        // use Cranelift's own post-register-allocation VCode printer.
        if std::env::var_os("COG_JIT_DUMP_IR").is_some() {
            eprintln!("[jit] Cranelift IR function={}", plan.entry);
            eprintln!("{}", context.func.display());
        }
        let dump_disassembly = std::env::var_os("COG_JIT_DISASM").is_some();
        context.set_disasm(dump_disassembly);
        self.module
            .define_function(function_id, &mut context)
            .map_err(|error| error.to_string())?;
        if dump_disassembly {
            match context
                .compiled_code()
                .and_then(|code| code.vcode.as_deref())
            {
                Some(disassembly) => {
                    eprintln!("[jit] machine code function={}", plan.entry);
                    eprintln!("{disassembly}");
                }
                None => eprintln!(
                    "[jit] machine code disassembly unavailable function={}",
                    plan.entry
                ),
            }
        }
        self.module.clear_context(&mut context);
        self.module
            .finalize_definitions()
            .map_err(|error| error.to_string())?;
        self.functions += 1;
        let pointer = self.module.get_finalized_function(function_id);
        // JitModule owns this executable memory until JitCompiler is dropped.
        let function = unsafe { transmute::<*const u8, JitEntry>(pointer) };
        Ok(function)
    }

    fn signature(&self) -> cranelift_codegen::ir::Signature {
        let mut signature = self.module.make_signature();
        signature.params.push(AbiParam::new(self.pointer_type));
        signature.returns.push(AbiParam::new(types::I32));
        signature
    }
}

fn declare_helper(
    module: &mut JITModule,
    name: &str,
    parameters: &[cranelift_codegen::ir::Type],
) -> Result<FuncId, String> {
    let mut signature = module.make_signature();
    signature
        .params
        .extend(parameters.iter().copied().map(AbiParam::new));
    signature.returns.push(AbiParam::new(types::I32));
    module
        .declare_function(name, Linkage::Import, &signature)
        .map_err(|error| error.to_string())
}

struct CountingMemoryProvider {
    inner: SystemMemoryProvider,
    code_bytes: Arc<AtomicUsize>,
    cache_full: Arc<AtomicBool>,
    limit: usize,
}

impl JITMemoryProvider for CountingMemoryProvider {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        let executable = matches!(&kind, JITMemoryKind::Executable);
        if executable && self.code_bytes.load(Ordering::Relaxed).saturating_add(size) > self.limit {
            self.cache_full.store(true, Ordering::Relaxed);
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Cog JIT executable memory limit reached",
            ));
        }
        let pointer = self.inner.allocate(size, align, kind)?;
        if executable {
            self.code_bytes.fetch_add(size, Ordering::Relaxed);
        }
        Ok(pointer)
    }

    unsafe fn free_memory(&mut self) {
        // The provider and all code pointers share the JITModule lifetime.
        unsafe { self.inner.free_memory() };
        self.code_bytes.store(0, Ordering::Relaxed);
    }

    fn finalize(
        &mut self,
        branch_protection: cranelift_jit::BranchProtection,
    ) -> cranelift_module::ModuleResult<()> {
        self.inner.finalize(branch_protection)
    }
}

fn to_slot(value: &Value, roots: &mut Vec<Value>, variant_tags: &VariantTags) -> JitSlot {
    match value {
        Value::Int(value) => JitSlot {
            tag: TAG_INT,
            payload: *value as u64,
        },
        Value::UInt(value) => JitSlot {
            tag: TAG_UINT,
            payload: *value,
        },
        Value::BigInt(integer) => small_bigint_slot(integer, value, roots)
            .unwrap_or_else(|| root_slot(value, roots, TAG_BIGINT)),
        Value::Float(value) => JitSlot {
            tag: TAG_FLOAT,
            payload: value.to_bits(),
        },
        Value::Bool(value) => JitSlot {
            tag: TAG_BOOL,
            payload: u64::from(*value),
        },
        Value::Char(value) => JitSlot {
            tag: TAG_CHAR,
            payload: *value as u32 as u64,
        },
        Value::Str(_) => root_slot(value, roots, TAG_STR),
        Value::Error(_) => root_slot(value, roots, TAG_ERROR),
        Value::Array(_) => root_slot(value, roots, TAG_ARRAY),
        Value::Variant(name, _) => variant_tags
            .slot_tag(name)
            .map(|tag| root_slot(value, roots, tag))
            .unwrap_or_else(|| root_slot(value, roots, TAG_OPAQUE)),
        Value::Nil => JitSlot {
            tag: TAG_NIL,
            payload: 0,
        },
        _ => root_slot(value, roots, TAG_OPAQUE),
    }
}

fn root_slot(value: &Value, roots: &mut Vec<Value>, tag: u64) -> JitSlot {
    let index = roots.len();
    roots.push(value.clone());
    JitSlot {
        tag,
        payload: index as u64,
    }
}

fn small_bigint_slot(
    integer: &num_bigint::BigInt,
    value: &Value,
    roots: &mut Vec<Value>,
) -> Option<JitSlot> {
    let (base, payload) = if let Some(value) = integer.to_i64() {
        (TAG_BIGINT_SMALL_SIGNED_BASE, value as u64)
    } else {
        (TAG_BIGINT_SMALL_UNSIGNED_BASE, integer.to_u64()?)
    };
    let index = roots.len();
    if index >= MAX_JIT_ROOTS {
        return None;
    }
    let tag = base.checked_add(u64::try_from(index).ok()?)?;
    roots.push(value.clone());
    Some(JitSlot { tag, payload })
}

fn rooted_tag_index(tag: u64, base: u64) -> Option<usize> {
    let index = usize::try_from(tag.checked_sub(base)?).ok()?;
    (index < MAX_JIT_ROOTS).then_some(index)
}

fn from_slot(
    slot: JitSlot,
    roots: &[Value],
    variant_tags: &VariantTags,
) -> Result<Option<Value>, ()> {
    let variant_value = |slot_tag: u64| {
        let index = usize::try_from(slot_tag.checked_sub(TAG_VARIANT_BASE)?).ok()?;
        if index >= variant_tags.names.len() {
            return None;
        }
        let expected_name = variant_tags.names.get(index)?;
        match roots.get(slot.payload as usize)? {
            value @ Value::Variant(name, _) if name.as_ref() == expected_name => {
                Some(value.clone())
            }
            _ => None,
        }
    };
    let value = match slot.tag {
        TAG_UNINITIALIZED => return Ok(None),
        TAG_INT => Value::Int(slot.payload as i64),
        TAG_UINT => Value::UInt(slot.payload),
        TAG_BIGINT => match roots.get(slot.payload as usize) {
            Some(value @ Value::BigInt(_)) => value.clone(),
            _ => return Err(()),
        },
        TAG_BIGINT_JIT_SIGNED => Value::BigInt(num_bigint::BigInt::from(slot.payload as i64)),
        TAG_BIGINT_JIT_UNSIGNED => Value::BigInt(num_bigint::BigInt::from(slot.payload)),
        tag if rooted_tag_index(tag, TAG_BIGINT_SMALL_SIGNED_BASE).is_some() => {
            let index = rooted_tag_index(tag, TAG_BIGINT_SMALL_SIGNED_BASE).ok_or(())?;
            match roots.get(index) {
                Some(value @ Value::BigInt(integer))
                    if integer
                        .to_i64()
                        .is_some_and(|value| value as u64 == slot.payload) =>
                {
                    value.clone()
                }
                _ => return Err(()),
            }
        }
        tag if rooted_tag_index(tag, TAG_BIGINT_SMALL_UNSIGNED_BASE).is_some() => {
            let index = rooted_tag_index(tag, TAG_BIGINT_SMALL_UNSIGNED_BASE).ok_or(())?;
            match roots.get(index) {
                Some(value @ Value::BigInt(integer))
                    if integer.to_u64().is_some_and(|value| value == slot.payload) =>
                {
                    value.clone()
                }
                _ => return Err(()),
            }
        }
        TAG_STR => match roots.get(slot.payload as usize) {
            Some(value @ Value::Str(_)) => value.clone(),
            _ => return Err(()),
        },
        TAG_ERROR => match roots.get(slot.payload as usize) {
            Some(value @ Value::Error(_)) => value.clone(),
            _ => return Err(()),
        },
        TAG_ARRAY => match roots.get(slot.payload as usize) {
            Some(value @ Value::Array(_)) => value.clone(),
            _ => return Err(()),
        },
        TAG_OPAQUE => roots.get(slot.payload as usize).ok_or(())?.clone(),
        tag if tag >= TAG_VARIANT_BASE
            && tag < TAG_VARIANT_BASE.saturating_add(variant_tags.names.len() as u64) =>
        {
            variant_value(tag).ok_or(())?
        }
        TAG_FLOAT => Value::Float(f64::from_bits(slot.payload)),
        TAG_BOOL if slot.payload <= 1 => Value::Bool(slot.payload != 0),
        TAG_CHAR => Value::Char(char::from_u32(slot.payload as u32).ok_or(())?),
        TAG_NIL if slot.payload == 0 => Value::Nil,
        _ => return Err(()),
    };
    Ok(Some(value))
}

fn bigint_from_slot<'a>(slot: JitSlot, roots: &'a [Value]) -> Option<Cow<'a, num_bigint::BigInt>> {
    match slot.tag {
        TAG_BIGINT => match roots.get(slot.payload as usize)? {
            Value::BigInt(value) => Some(Cow::Borrowed(value)),
            _ => None,
        },
        TAG_BIGINT_JIT_SIGNED => Some(Cow::Owned(num_bigint::BigInt::from(slot.payload as i64))),
        TAG_BIGINT_JIT_UNSIGNED => Some(Cow::Owned(num_bigint::BigInt::from(slot.payload))),
        tag if rooted_tag_index(tag, TAG_BIGINT_SMALL_SIGNED_BASE).is_some() => {
            let index = rooted_tag_index(tag, TAG_BIGINT_SMALL_SIGNED_BASE)?;
            let Value::BigInt(value) = roots.get(index)? else {
                return None;
            };
            value
                .to_i64()
                .filter(|value| *value as u64 == slot.payload)
                .map(|_| Cow::Borrowed(value))
        }
        tag if rooted_tag_index(tag, TAG_BIGINT_SMALL_UNSIGNED_BASE).is_some() => {
            let index = rooted_tag_index(tag, TAG_BIGINT_SMALL_UNSIGNED_BASE)?;
            let Value::BigInt(value) = roots.get(index)? else {
                return None;
            };
            value
                .to_u64()
                .filter(|value| *value == slot.payload)
                .map(|_| Cow::Borrowed(value))
        }
        _ => None,
    }
}

fn append_owned_value(
    value: Value,
    roots: &mut Vec<Value>,
    variant_tags: &VariantTags,
) -> Option<JitSlot> {
    match value {
        Value::Int(value) => Some(JitSlot {
            tag: TAG_INT,
            payload: value as u64,
        }),
        Value::UInt(value) => Some(JitSlot {
            tag: TAG_UINT,
            payload: value,
        }),
        Value::Float(value) => Some(JitSlot {
            tag: TAG_FLOAT,
            payload: value.to_bits(),
        }),
        Value::Bool(value) => Some(JitSlot {
            tag: TAG_BOOL,
            payload: u64::from(value),
        }),
        Value::Char(value) => Some(JitSlot {
            tag: TAG_CHAR,
            payload: value as u32 as u64,
        }),
        Value::Nil => Some(JitSlot {
            tag: TAG_NIL,
            payload: 0,
        }),
        Value::BigInt(integer) => {
            if let Some(value) = integer.to_i64() {
                Some(JitSlot {
                    tag: TAG_BIGINT_JIT_SIGNED,
                    payload: value as u64,
                })
            } else if let Some(value) = integer.to_u64() {
                Some(JitSlot {
                    tag: TAG_BIGINT_JIT_UNSIGNED,
                    payload: value,
                })
            } else {
                push_root(Value::BigInt(integer), roots, TAG_BIGINT)
            }
        }
        value => {
            let tag = match &value {
                Value::Str(_) => TAG_STR,
                Value::Error(_) => TAG_ERROR,
                Value::Array(_) => TAG_ARRAY,
                Value::Variant(name, _) => variant_tags.slot_tag(name).unwrap_or(TAG_OPAQUE),
                _ => TAG_OPAQUE,
            };
            push_root(value, roots, tag)
        }
    }
}

fn push_root(value: Value, roots: &mut Vec<Value>, tag: u64) -> Option<JitSlot> {
    if roots.len() >= MAX_JIT_ROOTS || roots.try_reserve(1).is_err() {
        return None;
    }
    let index = roots.len();
    roots.push(value);
    Some(JitSlot {
        tag,
        payload: index as u64,
    })
}

unsafe extern "C" fn jit_bigint_binary(frame: *mut JitFrame, depth: u64, operation: i32) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        jit_bigint_binary_inner(frame, depth, operation)
    }))
    .unwrap_or(false) as i32
}

unsafe fn jit_bigint_binary_inner(frame: *mut JitFrame, depth: u64, operation: i32) -> bool {
    let Some(frame) = (unsafe { frame.as_ref() }) else {
        return false;
    };
    let Ok(depth) = usize::try_from(depth) else {
        return false;
    };
    if !(2..=MAX_STACK).contains(&depth) || frame.stack.is_null() || frame.roots.is_null() {
        return false;
    }
    let left_slot = unsafe { *frame.stack.add(depth - 2) };
    let right_slot = unsafe { *frame.stack.add(depth - 1) };
    let result = {
        let roots = unsafe { &*frame.roots };
        let Some(left) = bigint_from_slot(left_slot, roots) else {
            return false;
        };
        let Some(right) = bigint_from_slot(right_slot, roots) else {
            return false;
        };
        match operation {
            0 => left.as_ref() + right.as_ref(),
            1 => left.as_ref() - right.as_ref(),
            2 => left.as_ref() * right.as_ref(),
            3 if !right.is_zero() => left.as_ref() / right.as_ref(),
            4 if !right.is_zero() => left.as_ref() % right.as_ref(),
            _ => return false,
        }
    };
    let roots = unsafe { &mut *frame.roots };
    let Some(slot) = append_owned_value(Value::BigInt(result), roots, &VariantTags::default())
    else {
        return false;
    };
    unsafe { *frame.stack.add(depth - 2) = slot };
    true
}

unsafe extern "C" fn jit_bigint_compare(frame: *mut JitFrame, depth: u64, operation: i32) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        jit_bigint_compare_inner(frame, depth, operation)
    }))
    .unwrap_or(false) as i32
}

unsafe fn jit_bigint_compare_inner(frame: *mut JitFrame, depth: u64, operation: i32) -> bool {
    let Some(frame) = (unsafe { frame.as_ref() }) else {
        return false;
    };
    let Ok(depth) = usize::try_from(depth) else {
        return false;
    };
    if !(2..=MAX_STACK).contains(&depth) || frame.stack.is_null() || frame.roots.is_null() {
        return false;
    }
    let left_slot = unsafe { *frame.stack.add(depth - 2) };
    let right_slot = unsafe { *frame.stack.add(depth - 1) };
    let result = {
        let roots = unsafe { &*frame.roots };
        let Some(left) = bigint_from_slot(left_slot, roots) else {
            return false;
        };
        let Some(right) = bigint_from_slot(right_slot, roots) else {
            return false;
        };
        let ordering = left.cmp(&right);
        match operation {
            0 => ordering.is_eq(),
            1 => !ordering.is_eq(),
            2 => ordering.is_lt(),
            3 => ordering.is_le(),
            4 => ordering.is_gt(),
            5 => ordering.is_ge(),
            _ => return false,
        }
    };
    unsafe {
        *frame.stack.add(depth - 2) = JitSlot {
            tag: TAG_BOOL,
            payload: u64::from(result),
        };
    }
    true
}

unsafe extern "C" fn jit_native_call(
    frame: *mut JitFrame,
    pc: u64,
    depth: u64,
    import_index: u64,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        jit_native_call_inner(frame, pc, depth, import_index)
    }))
    .unwrap_or(false) as i32
}

unsafe fn jit_native_call_inner(
    frame: *mut JitFrame,
    pc: u64,
    depth: u64,
    import_index: u64,
) -> bool {
    let Some(frame_ref) = (unsafe { frame.as_ref() }) else {
        return false;
    };
    if frame_ref.runtime_context.is_null() || frame_ref.stack.is_null() || frame_ref.roots.is_null()
    {
        return false;
    }
    let Ok(pc) = usize::try_from(pc) else {
        return false;
    };
    let Ok(depth) = usize::try_from(depth) else {
        return false;
    };
    let Ok(import_index) = usize::try_from(import_index) else {
        return false;
    };
    let context = unsafe { &mut *frame_ref.runtime_context.cast::<JitExecutionContext<'_>>() };
    let Some(Instruction::CallNative(actual_import)) = context.artifact.instructions.get(pc) else {
        return false;
    };
    if *actual_import != import_index {
        return false;
    }
    let Some(import) = context.artifact.imports.get(import_index) else {
        return false;
    };
    let arity = import.parameters().len();
    let Some(base) = depth.checked_sub(arity) else {
        return false;
    };
    if depth > MAX_STACK || arity > 3 {
        return false;
    }
    let roots = unsafe { &*frame_ref.roots };
    if matches!(import.name(), "bytes::get" | "bytes::set") {
        let expected_arity = if import.name() == "bytes::get" { 2 } else { 3 };
        if arity != expected_arity {
            return false;
        }
        let buffer_slot = unsafe { *frame_ref.stack.add(base) };
        let offset_slot = unsafe { *frame_ref.stack.add(base + 1) };
        if buffer_slot.tag != TAG_OPAQUE || offset_slot.tag != TAG_INT {
            return false;
        }
        let Ok(root_index) = usize::try_from(buffer_slot.payload) else {
            return false;
        };
        let Some(buffer) = roots.get(root_index) else {
            return false;
        };
        let offset = offset_slot.payload as i64;
        if import.name() == "bytes::get" {
            let Some(value) = crate::runtime::jit_bytes_get(buffer, offset) else {
                return false;
            };
            unsafe {
                *frame_ref.stack.add(base) = JitSlot {
                    tag: TAG_INT,
                    payload: i64::from(value) as u64,
                };
            }
            return true;
        }

        let value_slot = unsafe { *frame_ref.stack.add(base + 2) };
        if value_slot.tag != TAG_INT
            || !crate::runtime::jit_bytes_set(buffer, offset, value_slot.payload as i64)
        {
            return false;
        }
        unsafe {
            *frame_ref.stack.add(base) = JitSlot {
                tag: TAG_NIL,
                payload: 0,
            };
        }
        return true;
    }

    let mut arguments = std::array::from_fn::<_, 3, _>(|_| Value::Nil);
    for index in base..depth {
        let slot = unsafe { *frame_ref.stack.add(index) };
        let Ok(Some(value)) = from_slot(slot, roots, context.variant_tags) else {
            return false;
        };
        arguments[index - base] = value;
    }
    let arguments = &arguments[..arity];
    let result = match import.name() {
        "str::to_bigint" => match arguments.first() {
            Some(Value::Error(_)) | Some(Value::BigInt(_)) => arguments[0].clone(),
            Some(Value::Int(value)) => Value::BigInt((*value).into()),
            Some(Value::UInt(value)) => Value::BigInt((*value).into()),
            _ => return false,
        },
        _ => return false,
    };
    let roots = unsafe { &mut *frame_ref.roots };
    let Some(slot) = append_owned_value(result, roots, context.variant_tags) else {
        return false;
    };
    unsafe { *frame_ref.stack.add(base) = slot };
    true
}

unsafe extern "C" fn jit_direct_call(
    frame: *mut JitFrame,
    pc: u64,
    depth: u64,
    target: u64,
    arity: u64,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        jit_direct_call_inner(frame, pc, depth, target, arity)
    }))
    .unwrap_or(false) as i32
}

unsafe fn jit_direct_call_inner(
    frame: *mut JitFrame,
    pc: u64,
    depth: u64,
    target: u64,
    arity: u64,
) -> bool {
    let Some(frame_ref) = (unsafe { frame.as_ref() }) else {
        return false;
    };
    if frame_ref.runtime_context.is_null() || frame_ref.stack.is_null() || frame_ref.roots.is_null()
    {
        return false;
    }
    let (Ok(pc), Ok(depth), Ok(target), Ok(arity)) = (
        usize::try_from(pc),
        usize::try_from(depth),
        usize::try_from(target),
        usize::try_from(arity),
    ) else {
        return false;
    };
    if depth > MAX_STACK || arity > depth {
        return false;
    }
    let context_ptr = frame_ref.runtime_context.cast::<JitExecutionContext<'_>>();
    let (signature, callable) = {
        let context = unsafe { &*context_ptr };
        if !matches!(
            context.artifact.instructions.get(pc),
            Some(Instruction::Call(actual_target, actual_arity))
                if *actual_target == target && *actual_arity == arity
        ) {
            return false;
        }
        let Some(signature) = context.artifact.callables.get(&target).cloned() else {
            return false;
        };
        if signature.asynchronous || signature.captures != 0 || signature.parameters != arity {
            return false;
        }
        let Some(callable) = context.callables.get(&target).cloned() else {
            return false;
        };
        (signature, callable)
    };
    let base = depth - arity;
    if signature.parameter_types.is_none()
        && signature.result_type == Some(crate::value::RuntimeType::Int)
    {
        let Some(result) =
            (unsafe { jit_direct_call_int_slots(frame_ref, base, arity, context_ptr, &callable) })
        else {
            return false;
        };
        unsafe { *frame_ref.stack.add(base) = result };
        return true;
    }

    let variant_tags = unsafe { &(*context_ptr).variant_tags };
    let roots = unsafe { &*frame_ref.roots };
    if arity > MAX_JIT_CALL_ARGS {
        return false;
    }
    let mut arguments = std::array::from_fn::<_, MAX_JIT_CALL_ARGS, _>(|_| Value::Nil);
    for index in base..depth {
        let slot = unsafe { *frame_ref.stack.add(index) };
        let Ok(Some(value)) = from_slot(slot, roots, variant_tags) else {
            return false;
        };
        if matches!(value, Value::Error(_)) {
            return false;
        }
        arguments[index - base] = value;
    }
    if !crate::interpreter::call_arguments_match(&signature, &arguments[..arity]) {
        return false;
    }
    let mut child = unsafe { (&mut *context_ptr).call_frame.take() }
        .unwrap_or_else(|| Box::new(JitCallFrame::new(frame_ref.runtime_context)));
    child.prepare(&callable.plan, frame_ref.runtime_context);
    for (index, value) in arguments.iter_mut().take(arity).enumerate() {
        let value = std::mem::replace(value, Value::Nil);
        let Some(slot) = append_owned_value(value, &mut child.roots, variant_tags) else {
            return false;
        };
        child.stack[index] = slot;
    }
    child.frame.stack_len = arity as u64;
    let status = unsafe { (callable.entry)(&mut child.frame) };
    let result_slot =
        (|| {
            let context = unsafe { &*context_ptr };
            if !matches!(
                decode_exit(&callable.plan, context.artifact, &child.frame, status),
                Ok(JitExit::Completed)
            ) {
                return None;
            }
            let result_depth = usize::try_from(child.frame.stack_len).ok()?;
            let result_index = result_depth.checked_sub(1)?;
            let result_slot = child.stack.get(result_index).copied()?;
            let Ok(Some(mut result)) = from_slot(result_slot, &child.roots, context.variant_tags)
            else {
                return None;
            };
            if signature.result_type.as_ref().is_some_and(|expected| {
                !crate::interpreter::matches_runtime_type(expected, &result)
            }) {
                result = Value::Error(Box::new(RuntimeError::TypeMismatch { op: "return" }));
            }
            append_owned_value(
                result,
                unsafe { &mut *frame_ref.roots },
                context.variant_tags,
            )
        })();
    unsafe { (&mut *context_ptr).call_frame = Some(child) };
    let Some(result_slot) = result_slot else {
        return false;
    };
    unsafe { *frame_ref.stack.add(base) = result_slot };
    true
}

unsafe fn jit_direct_call_int_slots(
    frame: &JitFrame,
    base: usize,
    arity: usize,
    context_ptr: *mut JitExecutionContext<'_>,
    callable: &JitCallable,
) -> Option<JitSlot> {
    if arity > MAX_JIT_CALL_ARGS {
        return None;
    }
    for index in base..base + arity {
        let slot = unsafe { *frame.stack.add(index) };
        if matches!(slot.tag, TAG_UNINITIALIZED | TAG_ERROR) {
            return None;
        }
    }

    let root_len = unsafe { (&*frame.roots).len() };
    let mut child = unsafe { (&mut *context_ptr).call_frame.take() }
        .unwrap_or_else(|| Box::new(JitCallFrame::new(context_ptr.cast())));
    child.prepare(&callable.plan, context_ptr.cast());
    child.frame.roots = frame.roots;
    for (index, slot) in child.stack.get_mut(..arity)?.iter_mut().enumerate() {
        *slot = unsafe { *frame.stack.add(base + index) };
    }
    child.frame.stack_len = arity as u64;

    // The callee plan excludes Cog-to-Cog calls and effectful native imports.
    // Sharing roots therefore avoids cloning large BigInts for every pixel;
    // roots created by this call are discarded once its scalar result returns.
    let status = unsafe { (callable.entry)(&mut child.frame) };
    let result = {
        let context = unsafe { &*context_ptr };
        if matches!(
            decode_exit(&callable.plan, context.artifact, &child.frame, status),
            Ok(JitExit::Completed)
        ) && child.frame.stack_len == 1
        {
            child
                .stack
                .first()
                .copied()
                .filter(|slot| slot.tag == TAG_INT)
        } else {
            None
        }
    };
    unsafe {
        (&mut *frame.roots).truncate(root_len);
        (&mut *context_ptr).call_frame = Some(child);
    }
    result
}

fn decode_exit(
    plan: &FunctionPlan,
    artifact: &CompiledProgram,
    frame: &JitFrame,
    status: i32,
) -> Result<JitExit, ()> {
    let pc = usize::try_from(frame.pc).map_err(|_| ())?;
    let depth = usize::try_from(frame.stack_len).map_err(|_| ())?;
    let instruction = artifact.instructions.get(pc).ok_or(())?;
    if !(plan.entry..plan.end).contains(&pc)
        || plan.stack_depths.get(pc - plan.entry).copied().flatten() != Some(depth)
        || frame.propagation_pending > 1
    {
        return Err(());
    }

    match (status, frame.exit_reason) {
        (EXIT_COMPLETED, REASON_NORMAL) if matches!(instruction, Instruction::Return) => {
            Ok(JitExit::Completed)
        }
        (EXIT_RESUME, REASON_NORMAL) => Ok(JitExit::ResumeAt { pc }),
        (EXIT_RESUME, REASON_PROFILE_SLICE)
            if depth == 0
                && frame.backedge_budget == 0
                && frame.backedge_count > 0
                && plan.osr_entries.binary_search(&pc).is_ok() =>
        {
            Ok(JitExit::ResumeAt { pc })
        }
        (EXIT_FAULT, REASON_NORMAL)
            if matches!(
                instruction,
                Instruction::Load(_)
                    | Instruction::Add
                    | Instruction::Sub
                    | Instruction::Mul
                    | Instruction::Div
                    | Instruction::Mod
            ) =>
        {
            Ok(JitExit::FaultAt { pc })
        }
        _ => Err(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn restore_frame(
    plan: &FunctionPlan,
    frame: &JitFrame,
    jit_locals: &[JitSlot],
    jit_stack: &[JitSlot],
    roots: &[Value],
    variant_tags: &VariantTags,
    locals: &mut Vec<Option<Value>>,
    bp: usize,
    stack: &mut Vec<Value>,
    operand_base: usize,
) -> Result<(), ()> {
    let pc = usize::try_from(frame.pc).map_err(|_| ())?;
    let depth = usize::try_from(frame.stack_len).map_err(|_| ())?;
    if !(plan.entry..plan.end).contains(&pc)
        || jit_locals.len() != plan.local_count
        || depth > jit_stack.len()
        || plan.stack_depths.get(pc - plan.entry).copied().flatten() != Some(depth)
    {
        return Err(());
    }

    // Decode every slot before touching the interpreter frame. If native code
    // ever returns an invalid tag or root index, the caller can safely restart
    // this pure loop from its original OSR state.
    let mut restored_locals = Vec::new();
    restored_locals
        .try_reserve_exact(jit_locals.len())
        .map_err(|_| ())?;
    for slot in jit_locals.iter().copied() {
        restored_locals.push(from_slot(slot, roots, variant_tags)?);
    }
    let mut restored_stack = Vec::new();
    restored_stack.try_reserve_exact(depth).map_err(|_| ())?;
    for slot in jit_stack.iter().copied().take(depth) {
        restored_stack.push(from_slot(slot, roots, variant_tags)?.ok_or(())?);
    }

    let local_end = bp.checked_add(plan.local_count).ok_or(())?;
    let stack_end = operand_base.checked_add(depth).ok_or(())?;
    if locals.len() < local_end {
        locals
            .try_reserve(local_end - locals.len())
            .map_err(|_| ())?;
    }
    if stack.len() < stack_end {
        stack.try_reserve(stack_end - stack.len()).map_err(|_| ())?;
    }
    if locals.len() < local_end {
        locals.resize_with(local_end, || None);
    }
    for (index, value) in restored_locals.into_iter().enumerate() {
        locals[bp + index] = value;
    }
    stack.truncate(operand_base);
    stack.extend(restored_stack);
    Ok(())
}

fn load_profile(path: &Path, expected_key: &str) -> Option<ProfileFile> {
    let bytes = fs::read(path).ok()?;
    let mut profile: ProfileFile = serde_json::from_slice(&bytes).ok()?;
    if profile.key != expected_key {
        return None;
    }
    if profile
        .coefficients
        .as_ref()
        .is_some_and(|coefficients| !coefficients.is_valid())
    {
        profile.coefficients = None;
    }
    profile
        .functions
        .retain(|_, measurement| measurement.is_valid());
    Some(profile)
}

fn save_profile(path: &Path, profile: &ProfileFile) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let encoded = serde_json::to_vec(profile).map_err(io::Error::other)?;
    let (temporary, mut file) = (0..8)
        .find_map(|_| {
            let sequence = PROFILE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));
            match OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)
            {
                Ok(file) => Some(Ok((temporary, file))),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .transpose()?
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::AlreadyExists, "profile temp path collision")
        })?;
    let write_result = file.write_all(&encoded).and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

fn jit_reachable_pcs(
    plan: &FunctionPlan,
    artifact: &CompiledProgram,
    variant_tags: &VariantTags,
) -> Vec<bool> {
    let count = plan.end - plan.entry;
    let mut reachable = vec![false; count];
    let mut queue = VecDeque::new();
    reachable[0] = true;
    queue.push_back(plan.entry);
    for &entry in &plan.osr_entries {
        if (plan.entry..plan.end).contains(&entry) && !reachable[entry - plan.entry] {
            reachable[entry - plan.entry] = true;
            queue.push_back(entry);
        }
    }

    while let Some(pc) = queue.pop_front() {
        let Some(instruction) = artifact.instructions.get(pc) else {
            continue;
        };
        let next = (pc + 1 < plan.end).then_some(pc + 1);
        let successors: Vec<usize> = match instruction {
            Instruction::Push(value) if constant_slot(value).is_some() => {
                next.into_iter().collect()
            }
            Instruction::Load(_)
            | Instruction::LoadCapture(_)
            | Instruction::Store(_)
            | Instruction::Call(_, _)
            | Instruction::Add
            | Instruction::Sub
            | Instruction::Mul
            | Instruction::Div
            | Instruction::Mod
            | Instruction::Eq
            | Instruction::Ne
            | Instruction::Lt
            | Instruction::Le
            | Instruction::Gt
            | Instruction::Ge
            | Instruction::And
            | Instruction::Or
            | Instruction::Xor
            | Instruction::Not
            | Instruction::Dup
            | Instruction::Pop
            | Instruction::Swap
            | Instruction::StackTruncate(_)
            | Instruction::ConstrainArrayFrom => next.into_iter().collect(),
            Instruction::CallNative(index)
                if artifact.imports.get(*index).is_some_and(|import| {
                    matches!(
                        import.name(),
                        "bytes::get" | "bytes::set" | "str::to_bigint"
                    )
                }) =>
            {
                next.into_iter().collect()
            }
            Instruction::IsType(value_type)
                if type_tag(value_type).is_some()
                    || variant_tags.variant_tags(value_type).is_some() =>
            {
                next.into_iter().collect()
            }
            Instruction::Jump(target) => vec![*target],
            Instruction::JumpIfFalse(target)
            | Instruction::JumpIfTrue(target)
            | Instruction::JumpIfNoPropagation(target) => {
                let mut successors = vec![*target];
                successors.extend(next);
                successors
            }
            Instruction::Return | Instruction::ReturnPropagate => Vec::new(),
            _ => Vec::new(),
        };
        for successor in successors {
            if !(plan.entry..plan.end).contains(&successor) {
                continue;
            }
            let relative = successor - plan.entry;
            if plan.stack_depths[relative].is_some() && !reachable[relative] {
                reachable[relative] = true;
                queue.push_back(successor);
            }
        }
    }
    reachable
}

fn jit_instruction_is_linear(instruction: &Instruction, known_load_tag: Option<u64>) -> bool {
    match instruction {
        Instruction::Push(value) => constant_slot(value).is_some(),
        Instruction::Load(_) => known_load_tag.is_some(),
        Instruction::Store(_)
        | Instruction::Dup
        | Instruction::Pop
        | Instruction::StackTruncate(_)
        | Instruction::Swap => true,
        Instruction::IsType(value_type) => type_tag(value_type).is_some(),
        _ => false,
    }
}

fn jit_bytecode_block_starts(
    plan: &FunctionPlan,
    artifact: &CompiledProgram,
    reachable: &[bool],
    type_states: &[Option<JitTypeState>],
) -> std::collections::BTreeSet<usize> {
    let mut starts = std::collections::BTreeSet::from([plan.entry]);
    starts.extend(plan.osr_entries.iter().copied());
    for pc in plan.entry..plan.end {
        let relative = pc - plan.entry;
        if !reachable.get(relative).copied().unwrap_or(false) {
            continue;
        }
        let Some(instruction) = artifact.instructions.get(pc) else {
            continue;
        };
        let known_load_tag = match instruction {
            Instruction::Load(index) => type_states
                .get(relative)
                .and_then(Option::as_ref)
                .and_then(|state| state.locals.get(*index))
                .copied()
                .flatten(),
            _ => None,
        };
        match instruction {
            Instruction::Jump(target) => {
                starts.insert(*target);
            }
            Instruction::JumpIfFalse(target)
            | Instruction::JumpIfTrue(target)
            | Instruction::JumpIfNoPropagation(target) => {
                starts.insert(*target);
                if pc + 1 < plan.end {
                    starts.insert(pc + 1);
                }
            }
            _ => {}
        }
        if !jit_instruction_is_linear(instruction, known_load_tag)
            && pc + 1 < plan.end
            && reachable.get(relative + 1).copied().unwrap_or(false)
        {
            starts.insert(pc + 1);
        }
    }
    starts
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct JitTypeState {
    locals: Vec<Option<u64>>,
    stack: Vec<Option<u64>>,
}

impl JitTypeState {
    fn has_facts(&self) -> bool {
        self.locals.iter().any(Option::is_some) || self.stack.iter().any(Option::is_some)
    }
}

fn exact_stack_tag_pair(state: &JitTypeState, left: usize, right: usize) -> Option<(u64, u64)> {
    Some(((*state.stack.get(left)?)?, (*state.stack.get(right)?)?))
}

fn runtime_type_jit_tag(runtime_type: &RuntimeType) -> Option<u64> {
    Some(match runtime_type {
        RuntimeType::Int => TAG_INT,
        RuntimeType::UInt => TAG_UINT,
        RuntimeType::Float => TAG_FLOAT,
        RuntimeType::Bool => TAG_BOOL,
        RuntimeType::Char => TAG_CHAR,
        RuntimeType::Nil => TAG_NIL,
        _ => return None,
    })
}

fn jit_entry_type_state(plan: &FunctionPlan, artifact: &CompiledProgram) -> Option<JitTypeState> {
    let entry_depth = plan.stack_depths.first().copied().flatten()?;
    let mut stack = vec![None; entry_depth];
    if let Some(parameter_types) = artifact
        .callables
        .get(&plan.entry)
        .and_then(|signature| signature.parameter_types.as_ref())
    {
        for (slot, parameter_type) in stack.iter_mut().zip(parameter_types) {
            *slot = parameter_type.as_ref().and_then(runtime_type_jit_tag);
        }
    }
    Some(JitTypeState {
        locals: vec![None; plan.local_count],
        stack,
    })
}

fn jit_instruction_preserves_type_facts(instruction: &Instruction, state: &JitTypeState) -> bool {
    if matches!(instruction, Instruction::Add) {
        return exact_stack_tag_pair(
            state,
            state.stack.len().saturating_sub(2),
            state.stack.len().saturating_sub(1),
        )
        .is_some_and(|(left, right)| {
            matches!(left, TAG_INT | TAG_UINT | TAG_FLOAT)
                && matches!(right, TAG_INT | TAG_UINT | TAG_FLOAT)
        });
    }
    if matches!(instruction, Instruction::ConstrainArrayFrom) {
        return state
            .stack
            .get(state.stack.len().saturating_sub(2))
            .copied()
            .flatten()
            .is_some_and(|tag| tag != TAG_ARRAY);
    }
    matches!(
        instruction,
        Instruction::Push(_)
            | Instruction::Load(_)
            | Instruction::Store(_)
            | Instruction::Add
            | Instruction::Sub
            | Instruction::Mul
            | Instruction::Div
            | Instruction::Mod
            | Instruction::Eq
            | Instruction::Ne
            | Instruction::Lt
            | Instruction::Le
            | Instruction::Gt
            | Instruction::Ge
            | Instruction::And
            | Instruction::Or
            | Instruction::Xor
            | Instruction::Not
            | Instruction::Jump(_)
            | Instruction::JumpIfFalse(_)
            | Instruction::JumpIfTrue(_)
            | Instruction::JumpIfNoPropagation(_)
            | Instruction::IsType(_)
            | Instruction::Dup
            | Instruction::Pop
            | Instruction::Swap
            | Instruction::StackTruncate(_)
            | Instruction::ConstrainArrayFrom
    )
}

fn known_numeric_result_tag(
    left: Option<u64>,
    right: Option<u64>,
    instruction: &Instruction,
) -> Option<u64> {
    let left = left?;
    let right = right?;
    if left != right {
        return None;
    }
    match (instruction, left) {
        (
            Instruction::Add
            | Instruction::Sub
            | Instruction::Mul
            | Instruction::Div
            | Instruction::Mod,
            TAG_INT | TAG_UINT,
        )
        | (Instruction::Add | Instruction::Sub | Instruction::Mul | Instruction::Div, TAG_FLOAT) => {
            Some(left)
        }
        _ => None,
    }
}

fn known_comparison_result_tag(
    left: Option<u64>,
    right: Option<u64>,
    instruction: &Instruction,
) -> Option<u64> {
    let left = left?;
    if right? != left {
        return None;
    }
    match left {
        TAG_INT | TAG_UINT | TAG_FLOAT | TAG_CHAR => Some(TAG_BOOL),
        TAG_BOOL | TAG_NIL if matches!(instruction, Instruction::Eq | Instruction::Ne) => {
            Some(TAG_BOOL)
        }
        _ => None,
    }
}

fn jit_type_state_after(
    state: &JitTypeState,
    instruction: &Instruction,
    imports: &[crate::runtime::NativeImport],
) -> Option<JitTypeState> {
    if let Instruction::StackTruncate(target) = instruction {
        if *target > state.stack.len() {
            return None;
        }
        let mut next = state.clone();
        next.stack.truncate(*target);
        return Some(next);
    }
    let (required, produced) = instruction.stack_effect(imports).ok()?;
    if state.stack.len() < required {
        return None;
    }

    let mut next = state.clone();
    let retained = state.stack.len() - required;
    let popped = &state.stack[retained..];
    next.stack.truncate(retained);
    match instruction {
        Instruction::Push(value) => {
            next.stack.push(constant_slot(value).map(|slot| slot.tag));
        }
        Instruction::Load(index) => {
            next.stack.push(*state.locals.get(*index)?);
        }
        Instruction::Store(index) => {
            *next.locals.get_mut(*index)? = popped.last().copied().flatten();
        }
        Instruction::Dup => {
            let value = *popped.last()?;
            next.stack.extend([value, value]);
        }
        Instruction::Swap => {
            if popped.len() != 2 {
                return None;
            }
            next.stack.extend([popped[1], popped[0]]);
        }
        Instruction::Add
        | Instruction::Sub
        | Instruction::Mul
        | Instruction::Div
        | Instruction::Mod => {
            let left = popped.get(popped.len().checked_sub(2)?).copied().flatten();
            let right = popped.last().copied().flatten();
            next.stack
                .push(known_numeric_result_tag(left, right, instruction));
        }
        Instruction::Eq
        | Instruction::Ne
        | Instruction::Lt
        | Instruction::Le
        | Instruction::Gt
        | Instruction::Ge => {
            let left = popped.get(popped.len().checked_sub(2)?).copied().flatten();
            let right = popped.last().copied().flatten();
            next.stack
                .push(known_comparison_result_tag(left, right, instruction));
        }
        Instruction::IsType(value_type) if type_tag(value_type).is_some() => {
            next.stack.push(Some(TAG_BOOL));
        }
        Instruction::Not if popped.last() == Some(&Some(TAG_BOOL)) => {
            next.stack.push(Some(TAG_BOOL));
        }
        Instruction::And | Instruction::Or | Instruction::Xor
            if popped == [Some(TAG_BOOL), Some(TAG_BOOL)] =>
        {
            next.stack.push(Some(TAG_BOOL));
        }
        Instruction::ConstrainArrayFrom => {
            next.stack.push(popped.first().copied().flatten());
        }
        _ => next.stack.resize(retained + produced, None),
    }

    if !jit_instruction_preserves_type_facts(instruction, state) {
        next.locals.fill(None);
        next.stack.fill(None);
    }
    Some(next)
}

fn merge_jit_type_states(current: &mut JitTypeState, incoming: &JitTypeState) -> Option<bool> {
    if current.locals.len() != incoming.locals.len() || current.stack.len() != incoming.stack.len()
    {
        return None;
    }
    let mut changed = false;
    for (current, incoming) in current.locals.iter_mut().zip(&incoming.locals) {
        if current.is_some() && current != incoming {
            *current = None;
            changed = true;
        }
    }
    for (current, incoming) in current.stack.iter_mut().zip(&incoming.stack) {
        if current.is_some() && current != incoming {
            *current = None;
            changed = true;
        }
    }
    Some(changed)
}

fn infer_jit_type_states(
    plan: &FunctionPlan,
    artifact: &CompiledProgram,
) -> Vec<Option<JitTypeState>> {
    let mut states = vec![None; plan.end - plan.entry];
    let Some(entry_state) = jit_entry_type_state(plan, artifact) else {
        return states;
    };
    states[0] = Some(entry_state);
    let mut queue = VecDeque::from([plan.entry]);
    while let Some(pc) = queue.pop_front() {
        let relative = pc - plan.entry;
        let Some(state) = states.get(relative).and_then(Clone::clone) else {
            continue;
        };
        let Some(instruction) = artifact.instructions.get(pc) else {
            continue;
        };
        let Some(next_state) = jit_type_state_after(&state, instruction, &artifact.imports) else {
            continue;
        };
        let successors = match instruction {
            Instruction::Jump(target) => [Some(*target), None],
            Instruction::JumpIfFalse(target)
            | Instruction::JumpIfTrue(target)
            | Instruction::JumpIfNoPropagation(target) => [Some(*target), Some(pc + 1)],
            Instruction::Return
            | Instruction::ReturnPropagate
            | Instruction::Crash(_)
            | Instruction::Halt => [None, None],
            _ if pc + 1 < plan.end => [Some(pc + 1), None],
            _ => [None, None],
        };
        for successor in successors.into_iter().flatten() {
            let Some(successor_relative) = successor.checked_sub(plan.entry) else {
                continue;
            };
            let Some(expected_depth) = plan.stack_depths.get(successor_relative).copied().flatten()
            else {
                continue;
            };
            if next_state.stack.len() != expected_depth {
                continue;
            }
            let Some(target_state) = states.get_mut(successor_relative) else {
                continue;
            };
            let changed = match target_state {
                Some(current) => merge_jit_type_states(current, &next_state).unwrap_or(false),
                None => {
                    *target_state = Some(next_state.clone());
                    true
                }
            };
            if changed {
                queue.push_back(successor);
            }
        }
    }
    states
}

fn emit_jit_type_guard(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    state: &JitTypeState,
    target: cranelift_codegen::ir::Block,
) {
    let mut checks = Vec::new();
    for (index, expected) in state.locals.iter().copied().enumerate() {
        let Some(expected) = expected else {
            continue;
        };
        let (tag_variable, _) = local_state.variables[index];
        let tag = builder.use_var(tag_variable);
        checks.push(builder.ins().icmp_imm_s(IntCC::Equal, tag, expected as i64));
    }
    for (index, expected) in state.stack.iter().copied().enumerate() {
        let Some(expected) = expected else {
            continue;
        };
        let tag = load_stack_field(builder, local_state, index, 0);
        checks.push(builder.ins().icmp_imm_s(IntCC::Equal, tag, expected as i64));
    }
    if checks.is_empty() {
        builder.ins().jump(target, &[]);
        return;
    }
    let mut all_valid = checks[0];
    for check in checks.into_iter().skip(1) {
        all_valid = builder.ins().band(all_valid, check);
    }
    let valid = builder.create_block();
    let invalid = builder.create_block();
    builder.ins().brif(all_valid, valid, &[], invalid, &[]);
    builder.switch_to_block(invalid);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );
    builder.switch_to_block(valid);
    builder.ins().jump(target, &[]);
}

struct JitLocalState {
    variables: Vec<(Variable, Variable)>,
    stack_variables: Vec<(Variable, Variable)>,
    dirty: Variable,
    exit_block: cranelift_codegen::ir::Block,
}

#[derive(Clone, Copy)]
struct JitSite {
    pc: usize,
    stack_depth: usize,
}

fn lower_function(
    module: &mut JITModule,
    context: &mut cranelift_codegen::Context,
    builder_context: &mut FunctionBuilderContext,
    plan: &FunctionPlan,
    artifact: &CompiledProgram,
    variant_tags: &VariantTags,
    helpers: JitHelpers,
) -> Result<(), String> {
    let mut builder = FunctionBuilder::new(&mut context.func, builder_context);
    let helpers = JitHelperRefs {
        bigint_binary: module.declare_func_in_func(helpers.bigint_binary, builder.func),
        bigint_compare: module.declare_func_in_func(helpers.bigint_compare, builder.func),
        direct_call: module.declare_func_in_func(helpers.direct_call, builder.func),
        native_call: module.declare_func_in_func(helpers.native_call, builder.func),
    };
    let dispatch = builder.create_block();
    let exit_block = builder.create_block();
    builder.append_block_param(exit_block, types::I32);
    builder.append_block_param(exit_block, types::I64);
    builder.append_block_param(exit_block, types::I64);
    builder.append_block_param(exit_block, types::I64);
    builder.append_block_params_for_function_params(dispatch);
    let frame = builder.block_params(dispatch)[0];
    let local_state = JitLocalState {
        variables: (0..plan.local_count)
            .map(|_| {
                (
                    builder.declare_var(types::I64),
                    builder.declare_var(types::I64),
                )
            })
            .collect(),
        stack_variables: (0..plan.max_stack)
            .map(|_| {
                (
                    builder.declare_var(types::I64),
                    builder.declare_var(types::I64),
                )
            })
            .collect(),
        dirty: builder.declare_var(types::I8),
        exit_block,
    };
    let reachable = jit_reachable_pcs(plan, artifact, variant_tags);
    let type_states = infer_jit_type_states(plan, artifact);
    let block_starts = jit_bytecode_block_starts(plan, artifact, &reachable, &type_states);
    let mut blocks = HashMap::new();
    for start in block_starts.iter().copied() {
        let relative = start - plan.entry;
        if reachable.get(relative).copied().unwrap_or(false) {
            blocks.insert(start, builder.create_block());
        }
    }
    let mut block_for_pc = HashMap::new();
    let mut active_block = None;
    for (relative, is_reachable) in reachable.iter().copied().enumerate() {
        if !is_reachable {
            continue;
        }
        let pc = plan.entry + relative;
        if let Some(block) = blocks.get(&pc).copied() {
            active_block = Some(block);
        }
        if let Some(block) = active_block {
            block_for_pc.insert(pc, block);
        }
    }
    if std::env::var_os("COG_JIT_DUMP_IR").is_some() {
        eprintln!(
            "[jit] bytecode blocks function={} starts={block_starts:?} osr={:?}",
            plan.entry, plan.osr_entries
        );
        for (relative, state) in type_states.iter().enumerate() {
            if let Some(state) = state.as_ref() {
                eprintln!(
                    "[jit] inferred-types function={} pc={} instruction={:?} locals={:?} stack={:?}",
                    plan.entry,
                    plan.entry + relative,
                    artifact.instructions.get(plan.entry + relative),
                    state.locals,
                    state.stack
                );
            }
        }
    }

    builder.switch_to_block(dispatch);
    let start_pc = load_frame(
        &mut builder,
        frame,
        offset_of!(JitFrame, pc) as i32,
        types::I64,
    );
    let clean = builder.ins().iconst(types::I8, 0);
    builder.def_var(local_state.dirty, clean);
    for (index, (tag_variable, payload_variable)) in local_state.variables.iter().enumerate() {
        let tag = load_slot_field(&mut builder, frame, false, index, 0);
        let payload = load_slot_field(&mut builder, frame, false, index, 8);
        builder.def_var(*tag_variable, tag);
        builder.def_var(*payload_variable, payload);
    }
    let entry_depth = plan
        .stack_depths
        .first()
        .copied()
        .flatten()
        .ok_or("function entry has no verified stack depth")?;
    for (index, (tag_variable, payload_variable)) in local_state.stack_variables.iter().enumerate()
    {
        let (tag, payload) = if index < entry_depth {
            (
                load_slot_field(&mut builder, frame, true, index, 0),
                load_slot_field(&mut builder, frame, true, index, 8),
            )
        } else {
            (iconst_i64(&mut builder, 0), iconst_i64(&mut builder, 0))
        };
        builder.def_var(*tag_variable, tag);
        builder.def_var(*payload_variable, payload);
    }
    let entry_pc = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, start_pc, plan.entry as i64);
    let entry_block = block_for_pc
        .get(&plan.entry)
        .copied()
        .ok_or("function entry has no bytecode block")?;
    let next = builder.create_block();
    if let Some(entry_guard) = type_states
        .first()
        .and_then(Option::as_ref)
        .filter(|state| state.has_facts())
    {
        let guard = builder.create_block();
        builder.ins().brif(entry_pc, guard, &[], next, &[]);
        builder.switch_to_block(guard);
        emit_jit_type_guard(
            &mut builder,
            frame,
            &local_state,
            plan.entry,
            entry_depth,
            entry_guard,
            entry_block,
        );
    } else {
        builder.ins().brif(entry_pc, entry_block, &[], next, &[]);
    }
    builder.switch_to_block(next);
    for target in &plan.osr_entries {
        let target_block = block_for_pc
            .get(target)
            .copied()
            .ok_or("OSR point has no bytecode block")?;
        let is_target = builder
            .ins()
            .icmp_imm_s(IntCC::Equal, start_pc, *target as i64);
        let next = builder.create_block();
        let target_guard = type_states
            .get(target - plan.entry)
            .and_then(Option::as_ref)
            .filter(|state| state.has_facts());
        if let Some(target_guard) = target_guard {
            let guard = builder.create_block();
            builder.ins().brif(is_target, guard, &[], next, &[]);
            builder.switch_to_block(guard);
            let depth = plan.stack_depths[target - plan.entry]
                .ok_or("OSR point has no verified stack depth")?;
            emit_jit_type_guard(
                &mut builder,
                frame,
                &local_state,
                *target,
                depth,
                target_guard,
                target_block,
            );
        } else {
            builder.ins().brif(is_target, target_block, &[], next, &[]);
        }
        builder.switch_to_block(next);
    }
    let fallback = builder.ins().iconst(types::I32, i64::from(EXIT_RESUME));
    builder.ins().return_(&[fallback]);

    for pc in plan.entry..plan.end {
        let relative = pc - plan.entry;
        let Some(depth) = plan.stack_depths[relative].filter(|_| reachable[relative]) else {
            continue;
        };
        let type_state = type_states.get(relative).and_then(Option::as_ref);
        let bytecode_block = block_for_pc
            .get(&pc)
            .copied()
            .ok_or("reachable bytecode has no Cranelift block")?;
        if block_starts.contains(&pc) {
            builder.switch_to_block(bytecode_block);
        }
        let next = block_for_pc.get(&(pc + 1)).copied();
        match &artifact.instructions[pc] {
            Instruction::Push(value) => {
                if let Some(slot) = constant_slot(value) {
                    store_stack_const(&mut builder, &local_state, depth, slot);
                    continue_linear_block(&mut builder, bytecode_block, next, "push fallthrough")?;
                } else {
                    emit_exit(
                        &mut builder,
                        frame,
                        &local_state,
                        pc,
                        depth,
                        EXIT_RESUME,
                        REASON_NORMAL,
                    );
                }
            }
            Instruction::Load(index) => {
                let (tag_variable, payload_variable) = local_state.variables[*index];
                let payload = builder.use_var(payload_variable);
                if type_state
                    .and_then(|state| state.locals.get(*index))
                    .copied()
                    .flatten()
                    .is_some()
                {
                    // Entry and OSR guards establish the type. Keep the tag
                    // value live, though: mixed numeric comparisons need it
                    // to select the correct ordering path.
                    let tag = builder.use_var(tag_variable);
                    builder.def_var(local_state.stack_variables[depth].0, tag);
                    store_stack_field(&mut builder, &local_state, depth, 8, payload);
                    continue_linear_block(
                        &mut builder,
                        bytecode_block,
                        next,
                        "typed load fallthrough",
                    )?;
                } else {
                    let tag = builder.use_var(tag_variable);
                    let initialized =
                        builder
                            .ins()
                            .icmp_imm_s(IntCC::NotEqual, tag, TAG_UNINITIALIZED as i64);
                    let valid = builder.create_block();
                    let invalid = builder.create_block();
                    builder.ins().brif(initialized, valid, &[], invalid, &[]);
                    builder.switch_to_block(invalid);
                    emit_exit(
                        &mut builder,
                        frame,
                        &local_state,
                        pc,
                        depth,
                        EXIT_FAULT,
                        REASON_NORMAL,
                    );
                    builder.switch_to_block(valid);
                    store_stack_field(&mut builder, &local_state, depth, 0, tag);
                    store_stack_field(&mut builder, &local_state, depth, 8, payload);
                    jump_to(&mut builder, next, "load fallthrough")?;
                }
            }
            Instruction::LoadCapture(index) => {
                let capture_count = load_frame(
                    &mut builder,
                    frame,
                    offset_of!(JitFrame, capture_count) as i32,
                    types::I64,
                );
                let valid_capture = builder.ins().icmp_imm_s(
                    IntCC::UnsignedGreaterThan,
                    capture_count,
                    *index as i64,
                );
                let valid = builder.create_block();
                let invalid = builder.create_block();
                builder.ins().brif(valid_capture, valid, &[], invalid, &[]);
                builder.switch_to_block(invalid);
                emit_exit(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    EXIT_RESUME,
                    REASON_NORMAL,
                );
                builder.switch_to_block(valid);
                let capture_base = load_frame(
                    &mut builder,
                    frame,
                    offset_of!(JitFrame, captures) as i32,
                    types::I64,
                );
                let capture_address = builder
                    .ins()
                    .iadd_imm_s(capture_base, (*index * size_of::<JitSlot>()) as i64);
                let tag =
                    builder
                        .ins()
                        .load(types::I64, MemFlagsData::trusted(), capture_address, 0);
                let payload_address = builder.ins().iadd_imm_s(capture_address, 8);
                let payload =
                    builder
                        .ins()
                        .load(types::I64, MemFlagsData::trusted(), payload_address, 0);
                store_stack_field(&mut builder, &local_state, depth, 0, tag);
                store_stack_field(&mut builder, &local_state, depth, 8, payload);
                jump_to(&mut builder, next, "capture load fallthrough")?;
            }
            Instruction::Store(index) => {
                let (tag_variable, payload_variable) = local_state.variables[*index];
                let known_tag = type_state
                    .and_then(|state| state.stack.get(depth - 1))
                    .copied()
                    .flatten();
                let tag = match known_tag {
                    Some(tag) => iconst_i64(&mut builder, tag as i64),
                    None => load_stack_field(&mut builder, &local_state, depth - 1, 0),
                };
                let payload = load_stack_field(&mut builder, &local_state, depth - 1, 8);
                builder.def_var(tag_variable, tag);
                builder.def_var(payload_variable, payload);
                let dirty = builder.ins().iconst(types::I8, 1);
                builder.def_var(local_state.dirty, dirty);
                continue_linear_block(&mut builder, bytecode_block, next, "store fallthrough")?;
            }
            Instruction::Add
            | Instruction::Sub
            | Instruction::Mul
            | Instruction::Div
            | Instruction::Mod => {
                let known_tags = type_state
                    .and_then(|state| {
                        Some(exact_stack_tag_pair(
                            state,
                            depth.checked_sub(2)?,
                            depth.checked_sub(1)?,
                        ))
                    })
                    .flatten();
                emit_numeric_binary(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    next.ok_or("binary fallthrough is missing")?,
                    &artifact.instructions[pc],
                    known_tags,
                    helpers,
                );
            }
            Instruction::Eq
            | Instruction::Ne
            | Instruction::Lt
            | Instruction::Le
            | Instruction::Gt
            | Instruction::Ge => {
                let known_tags =
                    type_state.and_then(|state| exact_stack_tag_pair(state, depth - 2, depth - 1));
                emit_numeric_compare(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    next.ok_or("comparison fallthrough is missing")?,
                    &artifact.instructions[pc],
                    known_tags,
                    helpers,
                );
            }
            Instruction::Call(target, arity) => {
                emit_helper_call(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    *target,
                    *arity,
                    helpers.direct_call,
                    next.ok_or("call fallthrough is missing")?,
                )?;
            }
            Instruction::CallNative(index) => {
                let import = artifact.imports.get(*index);
                if import.is_some_and(|import| {
                    matches!(
                        import.name(),
                        "bytes::get" | "bytes::set" | "str::to_bigint"
                    )
                }) {
                    let arity = import.map_or(0, |import| import.parameters().len());
                    emit_native_call(
                        &mut builder,
                        frame,
                        &local_state,
                        pc,
                        depth,
                        *index,
                        arity,
                        helpers.native_call,
                        next.ok_or("native-call fallthrough is missing")?,
                    );
                } else {
                    emit_exit(
                        &mut builder,
                        frame,
                        &local_state,
                        pc,
                        depth,
                        EXIT_RESUME,
                        REASON_NORMAL,
                    );
                }
            }
            Instruction::Jump(target) => {
                let target_block = block_for_pc
                    .get(target)
                    .copied()
                    .ok_or("jump target has no bytecode block")?;
                emit_jump(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    *target,
                    plan.stack_depths[*target - plan.entry].ok_or("invalid jump stack")?,
                    target_block,
                );
            }
            Instruction::JumpIfNoPropagation(target) => {
                let target_block = block_for_pc
                    .get(target)
                    .copied()
                    .ok_or("propagation target has no bytecode block")?;
                let fallthrough = next.ok_or("propagation fallthrough is missing")?;
                let pending = load_frame(
                    &mut builder,
                    frame,
                    offset_of!(JitFrame, propagation_pending) as i32,
                    types::I64,
                );
                let no_pending = builder.ins().icmp_imm_s(IntCC::Equal, pending, 0);
                let take_target = builder.create_block();
                let clear_pending = builder.create_block();
                builder
                    .ins()
                    .brif(no_pending, take_target, &[], clear_pending, &[]);
                builder.switch_to_block(take_target);
                emit_jump(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    *target,
                    plan.stack_depths[*target - plan.entry].ok_or("invalid propagation stack")?,
                    target_block,
                );
                builder.switch_to_block(clear_pending);
                store_frame_i64(
                    &mut builder,
                    frame,
                    offset_of!(JitFrame, propagation_pending) as i32,
                    0,
                );
                builder.ins().jump(fallthrough, &[]);
            }
            Instruction::JumpIfFalse(target) | Instruction::JumpIfTrue(target) => {
                emit_conditional_jump(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    *target,
                    next,
                    &block_for_pc,
                    plan,
                    &artifact.instructions[pc],
                )?;
            }
            Instruction::Return => {
                emit_exit(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    EXIT_COMPLETED,
                    REASON_NORMAL,
                );
            }
            // Propagating returns update interpreter-only error state. Resume
            // before the instruction so the VM preserves those semantics.
            Instruction::ReturnPropagate => {
                emit_exit(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    EXIT_RESUME,
                    REASON_NORMAL,
                );
            }
            Instruction::IsType(value_type) => {
                if let Some(expected_tag) = type_tag(value_type) {
                    let actual_tag = load_stack_field(&mut builder, &local_state, depth - 1, 0);
                    let condition = if matches!(value_type, ValueType::BigInt) {
                        let rooted_bigint =
                            builder
                                .ins()
                                .icmp_imm_s(IntCC::Equal, actual_tag, TAG_BIGINT as i64);
                        let small_bigint = small_bigint_tag_test(&mut builder, actual_tag);
                        builder.ins().bor(rooted_bigint, small_bigint)
                    } else {
                        builder
                            .ins()
                            .icmp_imm_s(IntCC::Equal, actual_tag, expected_tag as i64)
                    };
                    let payload = builder.ins().uextend(types::I64, condition);
                    store_stack_const(
                        &mut builder,
                        &local_state,
                        depth - 1,
                        JitSlot {
                            tag: TAG_BOOL,
                            payload: 0,
                        },
                    );
                    store_stack_field(&mut builder, &local_state, depth - 1, 8, payload);
                    continue_linear_block(
                        &mut builder,
                        bytecode_block,
                        next,
                        "type-test fallthrough",
                    )?;
                } else if matches!(value_type, ValueType::Enum(_) | ValueType::Variant(_)) {
                    emit_variant_type_test(
                        &mut builder,
                        frame,
                        &local_state,
                        JitSite {
                            pc,
                            stack_depth: depth,
                        },
                        next,
                        value_type,
                        variant_tags,
                    )?;
                } else {
                    emit_exit(
                        &mut builder,
                        frame,
                        &local_state,
                        pc,
                        depth,
                        EXIT_RESUME,
                        REASON_NORMAL,
                    );
                }
            }
            Instruction::And | Instruction::Or | Instruction::Xor => {
                emit_bool_binary(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    next.ok_or("boolean operation fallthrough is missing")?,
                    &artifact.instructions[pc],
                );
            }
            Instruction::Not => {
                let tag = load_stack_field(&mut builder, &local_state, depth - 1, 0);
                let is_bool = builder.ins().icmp_imm_s(IntCC::Equal, tag, TAG_BOOL as i64);
                let valid = builder.create_block();
                let invalid = builder.create_block();
                builder.ins().brif(is_bool, valid, &[], invalid, &[]);
                builder.switch_to_block(invalid);
                emit_exit(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    EXIT_RESUME,
                    REASON_NORMAL,
                );
                builder.switch_to_block(valid);
                let payload = load_stack_field(&mut builder, &local_state, depth - 1, 8);
                let is_false = builder.ins().icmp_imm_s(IntCC::Equal, payload, 0);
                let payload = builder.ins().uextend(types::I64, is_false);
                store_stack_field(&mut builder, &local_state, depth - 1, 8, payload);
                jump_to(&mut builder, next, "not fallthrough")?;
            }
            Instruction::Dup => {
                copy_stack_slot(&mut builder, &local_state, depth - 1, depth);
                continue_linear_block(&mut builder, bytecode_block, next, "dup fallthrough")?;
            }
            Instruction::Pop => {
                continue_linear_block(&mut builder, bytecode_block, next, "pop fallthrough")?
            }
            Instruction::StackTruncate(_) => {
                continue_linear_block(
                    &mut builder,
                    bytecode_block,
                    next,
                    "stack truncate fallthrough",
                )?;
            }
            Instruction::Swap => {
                let a_tag = load_stack_field(&mut builder, &local_state, depth - 1, 0);
                let a_payload = load_stack_field(&mut builder, &local_state, depth - 1, 8);
                let b_tag = load_stack_field(&mut builder, &local_state, depth - 2, 0);
                let b_payload = load_stack_field(&mut builder, &local_state, depth - 2, 8);
                store_stack_field(&mut builder, &local_state, depth - 1, 0, b_tag);
                store_stack_field(&mut builder, &local_state, depth - 1, 8, b_payload);
                store_stack_field(&mut builder, &local_state, depth - 2, 0, a_tag);
                store_stack_field(&mut builder, &local_state, depth - 2, 8, a_payload);
                continue_linear_block(&mut builder, bytecode_block, next, "swap fallthrough")?;
            }
            Instruction::ConstrainArrayFrom => {
                // For scalars the VM operation preserves the assigned value.
                // Arrays may carry mutable type constraints, so let the VM own it.
                let tag = load_stack_field(&mut builder, &local_state, depth - 2, 0);
                let is_array = builder
                    .ins()
                    .icmp_imm_s(IntCC::Equal, tag, TAG_ARRAY as i64);
                let fallback = builder.create_block();
                let scalar = builder.create_block();
                builder.ins().brif(is_array, fallback, &[], scalar, &[]);
                builder.switch_to_block(fallback);
                emit_exit(
                    &mut builder,
                    frame,
                    &local_state,
                    pc,
                    depth,
                    EXIT_RESUME,
                    REASON_NORMAL,
                );
                builder.switch_to_block(scalar);
                jump_to(&mut builder, next, "array-context fallthrough")?;
            }
            _ => emit_exit(
                &mut builder,
                frame,
                &local_state,
                pc,
                depth,
                EXIT_RESUME,
                REASON_NORMAL,
            ),
        }
    }
    builder.switch_to_block(local_state.exit_block);
    let exit_params = builder.block_params(local_state.exit_block);
    let exit_status = exit_params[0];
    let exit_reason = exit_params[1];
    let exit_pc = exit_params[2];
    let exit_depth = exit_params[3];
    let dirty = builder.use_var(local_state.dirty);
    let flush = builder.create_block();
    let ready = builder.create_block();
    builder.ins().brif(dirty, flush, &[], ready, &[]);

    builder.switch_to_block(flush);
    for (index, (tag_variable, payload_variable)) in local_state.variables.iter().enumerate() {
        let tag = builder.use_var(*tag_variable);
        let payload = builder.use_var(*payload_variable);
        store_slot_field(&mut builder, frame, false, index, 0, tag);
        store_slot_field(&mut builder, frame, false, index, 8, payload);
    }
    builder.ins().jump(ready, &[]);

    builder.switch_to_block(ready);
    for (offset, value) in [
        (offset_of!(JitFrame, pc) as i32, exit_pc),
        (offset_of!(JitFrame, stack_len) as i32, exit_depth),
        (offset_of!(JitFrame, exit_reason) as i32, exit_reason),
    ] {
        let address = builder.ins().iadd_imm_s(frame, i64::from(offset));
        builder
            .ins()
            .store(MemFlagsData::trusted(), value, address, 0);
    }
    builder.ins().return_(&[exit_status]);
    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    Ok(())
}

fn emit_variant_type_test(
    builder: &mut FunctionBuilder,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    site: JitSite,
    next: Option<cranelift_codegen::ir::Block>,
    value_type: &ValueType,
    variant_tags: &VariantTags,
) -> Result<(), String> {
    let Some(expected_tags) = variant_tags.variant_tags(value_type) else {
        emit_exit(
            builder,
            frame,
            local_state,
            site.pc,
            site.stack_depth,
            EXIT_RESUME,
            REASON_NORMAL,
        );
        return Ok(());
    };
    let next = next.ok_or("variant type-test fallthrough is missing")?;
    let actual_tag = load_stack_field(builder, local_state, site.stack_depth - 1, 0);
    // Unknown variants and Error values may carry a nominal tag that is not
    // present in this artifact's registry. Resume the VM so it can compare the
    // full Cog tag instead of treating an absent native tag as a false match.
    let is_opaque = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, actual_tag, TAG_OPAQUE as i64);
    let is_error = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, actual_tag, TAG_ERROR as i64);
    let must_resume = builder.ins().bor(is_opaque, is_error);
    let fallback = builder.create_block();
    let inspect = builder.create_block();
    builder.ins().brif(must_resume, fallback, &[], inspect, &[]);

    builder.switch_to_block(fallback);
    emit_exit(
        builder,
        frame,
        local_state,
        site.pc,
        site.stack_depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );

    builder.switch_to_block(inspect);
    let mut matches = None;
    for expected_tag in expected_tags {
        let match_tag = builder.ins().icmp_imm_s(
            IntCC::Equal,
            actual_tag,
            i64::try_from(expected_tag).map_err(|_| "variant tag exceeds JIT immediate")?,
        );
        matches = Some(match matches {
            Some(previous) => builder.ins().bor(previous, match_tag),
            None => match_tag,
        });
    }
    let matches = matches.ok_or("variant test has no registered tags")?;
    let payload = builder.ins().uextend(types::I64, matches);
    store_stack_const(
        builder,
        local_state,
        site.stack_depth - 1,
        JitSlot {
            tag: TAG_BOOL,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, site.stack_depth - 1, 8, payload);
    jump_to(builder, Some(next), "variant type-test fallthrough")
}

fn type_tag(value_type: &ValueType) -> Option<u64> {
    Some(match value_type {
        ValueType::Int => TAG_INT,
        ValueType::UInt => TAG_UINT,
        ValueType::BigInt => TAG_BIGINT,
        ValueType::Float => TAG_FLOAT,
        ValueType::Bool => TAG_BOOL,
        ValueType::Char => TAG_CHAR,
        ValueType::Str => TAG_STR,
        ValueType::Nil => TAG_NIL,
        ValueType::Error => TAG_ERROR,
        ValueType::Enum(_) | ValueType::Variant(_) => return None,
    })
}

fn constant_slot(value: &Value) -> Option<JitSlot> {
    Some(match value {
        Value::Int(value) => JitSlot {
            tag: TAG_INT,
            payload: *value as u64,
        },
        Value::UInt(value) => JitSlot {
            tag: TAG_UINT,
            payload: *value,
        },
        Value::Float(value) => JitSlot {
            tag: TAG_FLOAT,
            payload: value.to_bits(),
        },
        Value::BigInt(value) => {
            if let Some(value) = value.to_i64() {
                JitSlot {
                    tag: TAG_BIGINT_JIT_SIGNED,
                    payload: value as u64,
                }
            } else {
                JitSlot {
                    tag: TAG_BIGINT_JIT_UNSIGNED,
                    payload: value.to_u64()?,
                }
            }
        }
        Value::Bool(value) => JitSlot {
            tag: TAG_BOOL,
            payload: u64::from(*value),
        },
        Value::Char(value) => JitSlot {
            tag: TAG_CHAR,
            payload: *value as u32 as u64,
        },
        Value::Nil => JitSlot {
            tag: TAG_NIL,
            payload: 0,
        },
        _ => return None,
    })
}

fn jump_to(
    builder: &mut FunctionBuilder<'_>,
    target: Option<cranelift_codegen::ir::Block>,
    context: &'static str,
) -> Result<(), String> {
    let target = target.ok_or(context)?;
    builder.ins().jump(target, &[]);
    Ok(())
}

fn continue_linear_block(
    builder: &mut FunctionBuilder<'_>,
    current: cranelift_codegen::ir::Block,
    target: Option<cranelift_codegen::ir::Block>,
    context: &'static str,
) -> Result<(), String> {
    let target = target.ok_or(context)?;
    if target != current {
        builder.ins().jump(target, &[]);
    }
    Ok(())
}

fn slot_address(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    stack: bool,
    index: usize,
) -> cranelift_codegen::ir::Value {
    let offset = if stack {
        offset_of!(JitFrame, stack)
    } else {
        offset_of!(JitFrame, locals)
    };
    let base = load_frame(builder, frame, offset as i32, types::I64);
    let byte_offset = index.saturating_mul(size_of::<JitSlot>()) as i64;
    builder.ins().iadd_imm_s(base, byte_offset)
}

fn load_slot_field(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    stack: bool,
    index: usize,
    field: i32,
) -> cranelift_codegen::ir::Value {
    let base = slot_address(builder, frame, stack, index);
    let address = if field == 0 {
        base
    } else {
        builder.ins().iadd_imm_s(base, i64::from(field))
    };
    builder
        .ins()
        .load(types::I64, MemFlagsData::trusted(), address, 0)
}

fn store_slot_field(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    stack: bool,
    index: usize,
    field: i32,
    value: cranelift_codegen::ir::Value,
) {
    let base = slot_address(builder, frame, stack, index);
    let address = if field == 0 {
        base
    } else {
        builder.ins().iadd_imm_s(base, i64::from(field))
    };
    builder
        .ins()
        .store(MemFlagsData::trusted(), value, address, 0);
}

fn load_stack_field(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    index: usize,
    field: i32,
) -> cranelift_codegen::ir::Value {
    let (tag, payload) = local_state.stack_variables[index];
    builder.use_var(if field == 0 { tag } else { payload })
}

fn store_stack_field(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    index: usize,
    field: i32,
    value: cranelift_codegen::ir::Value,
) {
    let (tag, payload) = local_state.stack_variables[index];
    builder.def_var(if field == 0 { tag } else { payload }, value);
}

fn store_stack_const(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    index: usize,
    slot: JitSlot,
) {
    let tag = iconst_i64(builder, slot.tag as i64);
    let payload = iconst_i64(builder, slot.payload as i64);
    store_stack_field(builder, local_state, index, 0, tag);
    store_stack_field(builder, local_state, index, 8, payload);
}

fn copy_stack_slot(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    source: usize,
    target: usize,
) {
    let tag = load_stack_field(builder, local_state, source, 0);
    let payload = load_stack_field(builder, local_state, source, 8);
    store_stack_field(builder, local_state, target, 0, tag);
    store_stack_field(builder, local_state, target, 8, payload);
}

fn materialize_stack(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    depth: usize,
) {
    for index in 0..depth {
        let tag = load_stack_field(builder, local_state, index, 0);
        let payload = load_stack_field(builder, local_state, index, 8);
        store_slot_field(builder, frame, true, index, 0, tag);
        store_slot_field(builder, frame, true, index, 8, payload);
    }
}

fn reload_stack(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    depth: usize,
) {
    for index in 0..depth {
        let tag = load_slot_field(builder, frame, true, index, 0);
        let payload = load_slot_field(builder, frame, true, index, 8);
        store_stack_field(builder, local_state, index, 0, tag);
        store_stack_field(builder, local_state, index, 8, payload);
    }
}

fn load_frame(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    offset: i32,
    ty: cranelift_codegen::ir::Type,
) -> cranelift_codegen::ir::Value {
    let address = builder.ins().iadd_imm_s(frame, i64::from(offset));
    builder.ins().load(ty, MemFlagsData::trusted(), address, 0)
}

fn store_frame_i64(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    offset: i32,
    value: i64,
) {
    let value = builder.ins().iconst(types::I64, value);
    let address = builder.ins().iadd_imm_s(frame, i64::from(offset));
    builder
        .ins()
        .store(MemFlagsData::trusted(), value, address, 0);
}

fn set_stack_len(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    depth: usize,
) {
    store_frame_i64(
        builder,
        frame,
        offset_of!(JitFrame, stack_len) as i32,
        depth as i64,
    );
}

fn emit_exit(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    stack_depth: usize,
    status: i32,
    reason: i64,
) {
    materialize_stack(builder, frame, local_state, stack_depth);
    let exit_reason = builder.ins().iconst(types::I64, reason);
    let result = builder.ins().iconst(types::I32, i64::from(status));
    let pc = builder.ins().iconst(types::I64, pc as i64);
    let stack_depth = builder.ins().iconst(types::I64, stack_depth as i64);
    builder.ins().jump(
        local_state.exit_block,
        &[
            BlockArg::Value(result),
            BlockArg::Value(exit_reason),
            BlockArg::Value(pc),
            BlockArg::Value(stack_depth),
        ],
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_numeric_binary(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
    known_tags: Option<(u64, u64)>,
    helpers: JitHelperRefs,
) {
    if let Some((left_tag, right_tag)) = known_tags {
        match (left_tag, right_tag, instruction) {
            (TAG_INT, TAG_INT, _) => emit_integer_binary_kind(
                builder,
                frame,
                local_state,
                pc,
                depth,
                next,
                instruction,
                TAG_INT,
                true,
            ),
            (TAG_UINT, TAG_UINT, _) => emit_integer_binary_kind(
                builder,
                frame,
                local_state,
                pc,
                depth,
                next,
                instruction,
                TAG_UINT,
                false,
            ),
            (
                TAG_FLOAT,
                TAG_FLOAT,
                Instruction::Add | Instruction::Sub | Instruction::Mul | Instruction::Div,
            ) => {
                emit_float_binary(builder, frame, local_state, pc, depth, next, instruction);
            }
            _ => emit_exit(
                builder,
                frame,
                local_state,
                pc,
                depth,
                EXIT_RESUME,
                REASON_NORMAL,
            ),
        }
        return;
    }
    let left_tag = load_stack_field(builder, local_state, depth - 2, 0);
    let right_tag = load_stack_field(builder, local_state, depth - 1, 0);
    let left_is_int = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_INT as i64);
    let right_is_int = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_INT as i64);
    let both_int = builder.ins().band(left_is_int, right_is_int);
    let left_is_uint = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_UINT as i64);
    let right_is_uint = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_UINT as i64);
    let both_uint = builder.ins().band(left_is_uint, right_is_uint);
    let left_is_float = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_FLOAT as i64);
    let right_is_float = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_FLOAT as i64);
    let both_float = builder.ins().band(left_is_float, right_is_float);
    let signed = builder.create_block();
    let check_unsigned = builder.create_block();
    let unsigned = builder.create_block();
    let check_float = builder.create_block();
    let float = (!matches!(instruction, Instruction::Mod)).then(|| builder.create_block());
    let fallback = builder.create_block();
    let bigint_check = matches!(
        instruction,
        Instruction::Add
            | Instruction::Sub
            | Instruction::Mul
            | Instruction::Div
            | Instruction::Mod
    )
    .then(|| builder.create_block());
    builder
        .ins()
        .brif(both_int, signed, &[], check_unsigned, &[]);
    builder.switch_to_block(check_unsigned);
    builder
        .ins()
        .brif(both_uint, unsigned, &[], check_float, &[]);
    builder.switch_to_block(check_float);
    if let Some(float) = float {
        builder.ins().brif(
            both_float,
            float,
            &[],
            bigint_check.unwrap_or(fallback),
            &[],
        );
    } else if let Some(bigint_check) = bigint_check {
        builder
            .ins()
            .brif(both_float, fallback, &[], bigint_check, &[]);
    } else {
        emit_exit(
            builder,
            frame,
            local_state,
            pc,
            depth,
            EXIT_RESUME,
            REASON_NORMAL,
        );
    }
    builder.switch_to_block(fallback);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );
    builder.switch_to_block(signed);
    emit_integer_binary_kind(
        builder,
        frame,
        local_state,
        pc,
        depth,
        next,
        instruction,
        TAG_INT,
        true,
    );
    builder.switch_to_block(unsigned);
    emit_integer_binary_kind(
        builder,
        frame,
        local_state,
        pc,
        depth,
        next,
        instruction,
        TAG_UINT,
        false,
    );
    if let Some(float) = float {
        builder.switch_to_block(float);
        emit_float_binary(builder, frame, local_state, pc, depth, next, instruction);
    }
    if let Some(bigint_check) = bigint_check {
        builder.switch_to_block(bigint_check);
        let left_bigint = bigint_tag_test(builder, left_tag);
        let right_bigint = bigint_tag_test(builder, right_tag);
        let both_bigint = builder.ins().band(left_bigint, right_bigint);
        let bigint = builder.create_block();
        builder.ins().brif(both_bigint, bigint, &[], fallback, &[]);
        builder.switch_to_block(bigint);
        let left_small = small_bigint_tag_test(builder, left_tag);
        let right_small = small_bigint_tag_test(builder, right_tag);
        let both_small = builder.ins().band(left_small, right_small);
        let inline = builder.create_block();
        let runtime = builder.create_block();
        builder.ins().brif(both_small, inline, &[], runtime, &[]);
        builder.switch_to_block(runtime);
        let depth_value = iconst_i64(builder, depth as i64);
        let operation_value = iconst_i32(builder, bigint_operation_code(instruction));
        emit_runtime_helper(
            builder,
            frame,
            local_state,
            pc,
            depth,
            next,
            helpers.bigint_binary,
            &[depth_value, operation_value],
            depth - 1,
        );
        builder.switch_to_block(inline);
        emit_bigint_binary(
            builder,
            frame,
            local_state,
            pc,
            depth,
            next,
            instruction,
            left_tag,
            right_tag,
            helpers.bigint_binary,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_helper_call(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    target: usize,
    arity: usize,
    helper: FuncRef,
    next: cranelift_codegen::ir::Block,
) -> Result<(), String> {
    let values = [
        iconst_i64(builder, pc as i64),
        iconst_i64(builder, depth as i64),
        iconst_i64(builder, target as i64),
        iconst_i64(builder, arity as i64),
    ];
    emit_runtime_helper(
        builder,
        frame,
        local_state,
        pc,
        depth,
        next,
        helper,
        &values,
        depth - arity + 1,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_native_call(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    import: usize,
    consumed: usize,
    helper: FuncRef,
    next: cranelift_codegen::ir::Block,
) {
    let values = [
        iconst_i64(builder, pc as i64),
        iconst_i64(builder, depth as i64),
        iconst_i64(builder, import as i64),
    ];
    emit_runtime_helper(
        builder,
        frame,
        local_state,
        pc,
        depth,
        next,
        helper,
        &values,
        depth - consumed + 1,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_runtime_helper(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    helper: FuncRef,
    arguments: &[cranelift_codegen::ir::Value],
    result_depth: usize,
) {
    materialize_stack(builder, frame, local_state, depth);
    set_stack_len(builder, frame, depth);
    let mut call_arguments = Vec::with_capacity(arguments.len() + 1);
    call_arguments.push(frame);
    call_arguments.extend_from_slice(arguments);
    let call = builder.ins().call(helper, &call_arguments);
    let status = builder.inst_results(call)[0];
    let succeeded = builder.ins().icmp_imm_s(IntCC::NotEqual, status, 0);
    let success = builder.create_block();
    let fallback = builder.create_block();
    builder.ins().brif(succeeded, success, &[], fallback, &[]);
    builder.switch_to_block(fallback);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );
    builder.switch_to_block(success);
    reload_stack(builder, frame, local_state, result_depth);
    set_stack_len(builder, frame, result_depth);
    builder.ins().jump(next, &[]);
}

fn iconst_i64(builder: &mut FunctionBuilder<'_>, value: i64) -> cranelift_codegen::ir::Value {
    builder.ins().iconst(types::I64, value)
}

fn iconst_i32(builder: &mut FunctionBuilder<'_>, value: i32) -> cranelift_codegen::ir::Value {
    builder.ins().iconst(types::I32, value as i64)
}

fn bigint_operation_code(instruction: &Instruction) -> i32 {
    match instruction {
        Instruction::Add => 0,
        Instruction::Sub => 1,
        Instruction::Mul => 2,
        Instruction::Div => 3,
        Instruction::Mod => 4,
        _ => unreachable!("BigInt arithmetic helper only handles numeric operators"),
    }
}

fn bigint_comparison_code(instruction: &Instruction) -> i32 {
    match instruction {
        Instruction::Eq => 0,
        Instruction::Ne => 1,
        Instruction::Lt => 2,
        Instruction::Le => 3,
        Instruction::Gt => 4,
        Instruction::Ge => 5,
        _ => unreachable!("BigInt comparison helper only handles comparisons"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_bigint_binary(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
    left_tag: cranelift_codegen::ir::Value,
    right_tag: cranelift_codegen::ir::Value,
    helper: FuncRef,
) {
    let left_signed = small_bigint_signed_tag_test(builder, left_tag);
    let right_signed = small_bigint_signed_tag_test(builder, right_tag);
    let both_signed = builder.ins().band(left_signed, right_signed);
    let left = load_stack_field(builder, local_state, depth - 2, 8);
    let right = load_stack_field(builder, local_state, depth - 1, 8);
    let fallback = builder.create_block();
    let signed = builder.create_block();
    let check_unsigned = builder.create_block();
    let unsigned = builder.create_block();
    let mixed = builder.create_block();
    builder
        .ins()
        .brif(both_signed, signed, &[], check_unsigned, &[]);

    builder.switch_to_block(check_unsigned);
    let left_unsigned = small_bigint_unsigned_tag_test(builder, left_tag);
    let right_unsigned = small_bigint_unsigned_tag_test(builder, right_tag);
    let both_unsigned = builder.ins().band(left_unsigned, right_unsigned);
    builder.ins().brif(both_unsigned, unsigned, &[], mixed, &[]);

    if matches!(instruction, Instruction::Div | Instruction::Mod) {
        builder.switch_to_block(signed);
        emit_bigint_divmod(
            builder,
            frame,
            local_state,
            pc,
            depth,
            next,
            instruction,
            left,
            right,
            true,
            fallback,
        );
        builder.switch_to_block(unsigned);
        emit_bigint_divmod(
            builder,
            frame,
            local_state,
            pc,
            depth,
            next,
            instruction,
            left,
            right,
            false,
            fallback,
        );
        builder.switch_to_block(mixed);
        builder.ins().jump(fallback, &[]);
        builder.switch_to_block(fallback);
        emit_exit(
            builder,
            frame,
            local_state,
            pc,
            depth,
            EXIT_RESUME,
            REASON_NORMAL,
        );
        return;
    }

    builder.switch_to_block(signed);
    let (result, overflow) = match instruction {
        Instruction::Add => {
            let result = builder.ins().iadd(left, right);
            let left_sign = builder.ins().bxor(left, result);
            let right_sign = builder.ins().bxor(right, result);
            let signs = builder.ins().band(left_sign, right_sign);
            let overflow = builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
            (result, overflow)
        }
        Instruction::Sub => {
            let operands = builder.ins().bxor(left, right);
            let result = builder.ins().isub(left, right);
            let left_sign = builder.ins().bxor(left, result);
            let signs = builder.ins().band(operands, left_sign);
            let overflow = builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
            (result, overflow)
        }
        Instruction::Mul => builder.ins().smul_overflow(left, right),
        _ => unreachable!("BigInt fast path handles arithmetic"),
    };
    let safe = builder.create_block();
    builder.ins().brif(overflow, fallback, &[], safe, &[]);
    builder.switch_to_block(safe);
    emit_bigint_result(
        builder,
        local_state,
        depth,
        next,
        result,
        TAG_BIGINT_JIT_SIGNED,
    );

    builder.switch_to_block(unsigned);
    let (result, overflow) = match instruction {
        Instruction::Add => builder.ins().uadd_overflow(left, right),
        Instruction::Sub => builder.ins().usub_overflow(left, right),
        Instruction::Mul => builder.ins().umul_overflow(left, right),
        _ => unreachable!("BigInt fast path handles arithmetic"),
    };
    if matches!(instruction, Instruction::Sub) {
        let no_underflow = builder.ins().icmp_imm_s(IntCC::Equal, overflow, 0);
        let result_is_negative = builder.ins().icmp_imm_s(IntCC::SignedLessThan, result, 0);
        let negative_fits = builder.ins().band(overflow, result_is_negative);
        let fits = builder.ins().bor(no_underflow, negative_fits);
        let safe = builder.create_block();
        builder.ins().brif(fits, safe, &[], fallback, &[]);
        builder.switch_to_block(safe);
        let signed_tag = builder
            .ins()
            .iconst(types::I64, TAG_BIGINT_JIT_SIGNED as i64);
        let unsigned_tag = builder
            .ins()
            .iconst(types::I64, TAG_BIGINT_JIT_UNSIGNED as i64);
        let result_tag = builder
            .ins()
            .select(negative_fits, signed_tag, unsigned_tag);
        emit_bigint_dynamic_result(builder, local_state, depth, next, result, result_tag);
    } else {
        let safe = builder.create_block();
        builder.ins().brif(overflow, fallback, &[], safe, &[]);
        builder.switch_to_block(safe);
        emit_bigint_result(
            builder,
            local_state,
            depth,
            next,
            result,
            TAG_BIGINT_JIT_UNSIGNED,
        );
    }

    builder.switch_to_block(mixed);
    emit_bigint_mixed_arithmetic(
        builder,
        local_state,
        depth,
        next,
        instruction,
        left,
        right,
        left_signed,
        right_signed,
        fallback,
    );

    builder.switch_to_block(fallback);
    let depth_value = iconst_i64(builder, depth as i64);
    let operation_value = iconst_i32(builder, bigint_operation_code(instruction));
    emit_runtime_helper(
        builder,
        frame,
        local_state,
        pc,
        depth,
        next,
        helper,
        &[depth_value, operation_value],
        depth - 1,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_bigint_mixed_arithmetic(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
    left: cranelift_codegen::ir::Value,
    right: cranelift_codegen::ir::Value,
    left_signed: cranelift_codegen::ir::Value,
    right_signed: cranelift_codegen::ir::Value,
    fallback: cranelift_codegen::ir::Block,
) {
    // Exactly one operand is signed i64 and the other is unsigned u64. Their
    // sum, difference, and product fit in i128, so range-checking the final
    // result before reducing it is exact.
    let signed_left = builder.ins().sextend(types::I128, left);
    let unsigned_left = builder.ins().uextend(types::I128, left);
    let left = builder
        .ins()
        .select(left_signed, signed_left, unsigned_left);
    let signed_right = builder.ins().sextend(types::I128, right);
    let unsigned_right = builder.ins().uextend(types::I128, right);
    let right = builder
        .ins()
        .select(right_signed, signed_right, unsigned_right);

    let result = match instruction {
        Instruction::Add => builder.ins().iadd(left, right),
        Instruction::Sub => builder.ins().isub(left, right),
        Instruction::Mul => builder.ins().imul(left, right),
        _ => unreachable!("mixed BigInt lowering handles add, subtract, and multiply"),
    };
    let payload = builder.ins().ireduce(types::I64, result);
    let signed_roundtrip = builder.ins().sextend(types::I128, payload);
    let signed_fits = builder.ins().icmp(IntCC::Equal, result, signed_roundtrip);
    let unsigned_roundtrip = builder.ins().uextend(types::I128, payload);
    let unsigned_fits = builder.ins().icmp(IntCC::Equal, result, unsigned_roundtrip);
    let zero_i64 = builder.ins().iconst(types::I64, 0);
    let zero = builder.ins().uextend(types::I128, zero_i64);
    let negative = builder.ins().icmp(IntCC::SignedLessThan, result, zero);
    let fits = builder.ins().select(negative, signed_fits, unsigned_fits);
    let safe = builder.create_block();
    builder.ins().brif(fits, safe, &[], fallback, &[]);
    builder.switch_to_block(safe);

    let signed_tag = builder
        .ins()
        .iconst(types::I64, TAG_BIGINT_JIT_SIGNED as i64);
    let unsigned_tag = builder
        .ins()
        .iconst(types::I64, TAG_BIGINT_JIT_UNSIGNED as i64);
    let result_tag = builder.ins().select(negative, signed_tag, unsigned_tag);
    emit_bigint_dynamic_result(builder, local_state, depth, next, payload, result_tag);
}

#[allow(clippy::too_many_arguments)]
fn emit_bigint_divmod(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
    left: cranelift_codegen::ir::Value,
    right: cranelift_codegen::ir::Value,
    signed: bool,
    fallback: cranelift_codegen::ir::Block,
) {
    let zero = builder.ins().icmp_imm_s(IntCC::Equal, right, 0);
    let fail = builder.create_block();
    let nonzero = builder.create_block();
    builder.ins().brif(zero, fail, &[], nonzero, &[]);
    builder.switch_to_block(fail);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_FAULT,
        REASON_NORMAL,
    );
    builder.switch_to_block(nonzero);

    let safe = builder.create_block();
    if signed {
        let min = builder.ins().icmp_imm_s(IntCC::Equal, left, i64::MIN);
        let minus_one = builder.ins().icmp_imm_s(IntCC::Equal, right, -1);
        let min_overflow = builder.ins().band(min, minus_one);
        builder.ins().brif(min_overflow, fallback, &[], safe, &[]);
    } else {
        builder.ins().jump(safe, &[]);
    }
    builder.switch_to_block(safe);
    let result = match (instruction, signed) {
        (Instruction::Div, true) => builder.ins().sdiv(left, right),
        (Instruction::Div, false) => builder.ins().udiv(left, right),
        (Instruction::Mod, true) => builder.ins().srem(left, right),
        (Instruction::Mod, false) => builder.ins().urem(left, right),
        _ => unreachable!("BigInt divmod lowering only handles div and mod"),
    };
    emit_bigint_result(
        builder,
        local_state,
        depth,
        next,
        result,
        if signed {
            TAG_BIGINT_JIT_SIGNED
        } else {
            TAG_BIGINT_JIT_UNSIGNED
        },
    );
}

fn emit_bigint_result(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    result: cranelift_codegen::ir::Value,
    result_tag: u64,
) {
    store_stack_const(
        builder,
        local_state,
        depth - 2,
        JitSlot {
            tag: result_tag,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, depth - 2, 8, result);
    builder.ins().jump(next, &[]);
}

fn emit_bigint_dynamic_result(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    result: cranelift_codegen::ir::Value,
    result_tag: cranelift_codegen::ir::Value,
) {
    store_stack_field(builder, local_state, depth - 2, 0, result_tag);
    store_stack_field(builder, local_state, depth - 2, 8, result);
    builder.ins().jump(next, &[]);
}

fn emit_float_binary(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
) {
    let left_bits = load_stack_field(builder, local_state, depth - 2, 8);
    let right_bits = load_stack_field(builder, local_state, depth - 1, 8);
    let left = builder
        .ins()
        .bitcast(types::F64, MemFlagsData::new(), left_bits);
    let right = builder
        .ins()
        .bitcast(types::F64, MemFlagsData::new(), right_bits);
    let result = match instruction {
        Instruction::Add => builder.ins().fadd(left, right),
        Instruction::Sub => builder.ins().fsub(left, right),
        Instruction::Mul => builder.ins().fmul(left, right),
        Instruction::Div => {
            let zero = builder.ins().f64const(Ieee64::with_bits(0));
            let is_zero = builder.ins().fcmp(FloatCC::Equal, right, zero);
            let fail = builder.create_block();
            let safe = builder.create_block();
            builder.ins().brif(is_zero, fail, &[], safe, &[]);
            builder.switch_to_block(fail);
            emit_exit(
                builder,
                frame,
                local_state,
                pc,
                depth,
                EXIT_FAULT,
                REASON_NORMAL,
            );
            builder.switch_to_block(safe);
            builder.ins().fdiv(left, right)
        }
        _ => unreachable!("float lowering supports add, sub, mul, and div"),
    };
    let result = builder
        .ins()
        .bitcast(types::I64, MemFlagsData::new(), result);
    store_stack_const(
        builder,
        local_state,
        depth - 2,
        JitSlot {
            tag: TAG_FLOAT,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, depth - 2, 8, result);
    builder.ins().jump(next, &[]);
}

#[allow(clippy::too_many_arguments)]
fn emit_integer_binary_kind(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
    result_tag: u64,
    signed: bool,
) {
    let left = load_stack_field(builder, local_state, depth - 2, 8);
    let right = load_stack_field(builder, local_state, depth - 1, 8);

    let (result, overflow) = match instruction {
        Instruction::Add => {
            let (result, overflow) = if signed {
                let result = builder.ins().iadd(left, right);
                let lhs_sign = builder.ins().bxor(left, result);
                let rhs_sign = builder.ins().bxor(right, result);
                let signs = builder.ins().band(lhs_sign, rhs_sign);
                let overflow = builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
                (result, overflow)
            } else {
                builder.ins().uadd_overflow(left, right)
            };
            (result, Some(overflow))
        }
        Instruction::Sub => {
            let (result, overflow) = if signed {
                let operands = builder.ins().bxor(left, right);
                let result = builder.ins().isub(left, right);
                let lhs_sign = builder.ins().bxor(left, result);
                let signs = builder.ins().band(operands, lhs_sign);
                let overflow = builder.ins().icmp_imm_s(IntCC::SignedLessThan, signs, 0);
                (result, overflow)
            } else {
                builder.ins().usub_overflow(left, right)
            };
            (result, Some(overflow))
        }
        Instruction::Mul if signed => {
            let (result, overflow) = builder.ins().smul_overflow(left, right);
            (result, Some(overflow))
        }
        Instruction::Mul => {
            let (result, overflow) = builder.ins().umul_overflow(left, right);
            (result, Some(overflow))
        }
        Instruction::Div | Instruction::Mod => {
            let zero = builder.ins().icmp_imm_s(IntCC::Equal, right, 0);
            let overflow = if signed {
                let min = builder.ins().icmp_imm_s(IntCC::Equal, left, i64::MIN);
                let minus_one = builder.ins().icmp_imm_s(IntCC::Equal, right, -1);
                let min_overflow = builder.ins().band(min, minus_one);
                builder.ins().bor(zero, min_overflow)
            } else {
                zero
            };
            let fail = builder.create_block();
            let safe = builder.create_block();
            builder.ins().brif(overflow, fail, &[], safe, &[]);
            builder.switch_to_block(fail);
            emit_exit(
                builder,
                frame,
                local_state,
                pc,
                depth,
                EXIT_FAULT,
                REASON_NORMAL,
            );
            builder.switch_to_block(safe);
            let result = match (instruction, signed) {
                (Instruction::Div, true) => builder.ins().sdiv(left, right),
                (Instruction::Div, false) => builder.ins().udiv(left, right),
                (Instruction::Mod, true) => builder.ins().srem(left, right),
                (Instruction::Mod, false) => builder.ins().urem(left, right),
                _ => unreachable!("integer division lowering only handles div and mod"),
            };
            store_stack_const(
                builder,
                local_state,
                depth - 2,
                JitSlot {
                    tag: result_tag,
                    payload: 0,
                },
            );
            store_stack_field(builder, local_state, depth - 2, 8, result);
            builder.ins().jump(next, &[]);
            return;
        }
        _ => unreachable!("integer lowering is used only for arithmetic"),
    };
    let fail = builder.create_block();
    let safe = builder.create_block();
    builder.ins().brif(
        overflow.expect("checked arithmetic overflow flag"),
        fail,
        &[],
        safe,
        &[],
    );
    builder.switch_to_block(fail);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_FAULT,
        REASON_NORMAL,
    );
    builder.switch_to_block(safe);
    store_stack_const(
        builder,
        local_state,
        depth - 2,
        JitSlot {
            tag: result_tag,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, depth - 2, 8, result);
    builder.ins().jump(next, &[]);
}

fn tag_range_test(
    builder: &mut FunctionBuilder<'_>,
    tag: cranelift_codegen::ir::Value,
    base: u64,
) -> cranelift_codegen::ir::Value {
    let lower = builder
        .ins()
        .icmp_imm_s(IntCC::UnsignedGreaterThanOrEqual, tag, base as i64);
    let upper = builder.ins().icmp_imm_s(
        IntCC::UnsignedLessThan,
        tag,
        base.saturating_add(MAX_JIT_ROOTS as u64) as i64,
    );
    builder.ins().band(lower, upper)
}

fn small_bigint_signed_tag_test(
    builder: &mut FunctionBuilder<'_>,
    tag: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let rooted = tag_range_test(builder, tag, TAG_BIGINT_SMALL_SIGNED_BASE);
    let computed = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, tag, TAG_BIGINT_JIT_SIGNED as i64);
    builder.ins().bor(rooted, computed)
}

fn small_bigint_unsigned_tag_test(
    builder: &mut FunctionBuilder<'_>,
    tag: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let rooted = tag_range_test(builder, tag, TAG_BIGINT_SMALL_UNSIGNED_BASE);
    let computed = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, tag, TAG_BIGINT_JIT_UNSIGNED as i64);
    builder.ins().bor(rooted, computed)
}

fn small_bigint_tag_test(
    builder: &mut FunctionBuilder<'_>,
    tag: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let signed = small_bigint_signed_tag_test(builder, tag);
    let unsigned = small_bigint_unsigned_tag_test(builder, tag);
    builder.ins().bor(signed, unsigned)
}

fn bigint_tag_test(
    builder: &mut FunctionBuilder<'_>,
    tag: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let rooted = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, tag, TAG_BIGINT as i64);
    let small = small_bigint_tag_test(builder, tag);
    builder.ins().bor(rooted, small)
}

fn emit_integer_comparison(
    builder: &mut FunctionBuilder<'_>,
    instruction: &Instruction,
    left: cranelift_codegen::ir::Value,
    right: cranelift_codegen::ir::Value,
    left_signed: cranelift_codegen::ir::Value,
    right_signed: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let signed_less = builder.ins().icmp(IntCC::SignedLessThan, left, right);
    let signed_less_equal = builder
        .ins()
        .icmp(IntCC::SignedLessThanOrEqual, left, right);
    let signed_greater = builder.ins().icmp(IntCC::SignedGreaterThan, left, right);
    let signed_greater_equal = builder
        .ins()
        .icmp(IntCC::SignedGreaterThanOrEqual, left, right);
    let unsigned_less = builder.ins().icmp(IntCC::UnsignedLessThan, left, right);
    let unsigned_less_equal = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, left, right);
    let unsigned_greater = builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right);
    let unsigned_greater_equal = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, left, right);
    let equal = builder.ins().icmp(IntCC::Equal, left, right);
    let not_equal = builder.ins().icmp(IntCC::NotEqual, left, right);
    let zero = builder.ins().iconst(types::I64, 0);
    let left_negative = builder.ins().icmp(IntCC::SignedLessThan, left, zero);
    let right_negative = builder.ins().icmp(IntCC::SignedLessThan, right, zero);
    let left_nonnegative = builder
        .ins()
        .icmp(IntCC::SignedGreaterThanOrEqual, left, zero);
    let right_nonnegative = builder
        .ins()
        .icmp(IntCC::SignedGreaterThanOrEqual, right, zero);
    let true_value = builder.ins().iconst(types::I8, 1);
    let false_value = builder.ins().iconst(types::I8, 0);

    let same_signedness = match instruction {
        Instruction::Eq => equal,
        Instruction::Ne => not_equal,
        Instruction::Lt => builder
            .ins()
            .select(left_signed, signed_less, unsigned_less),
        Instruction::Le => {
            builder
                .ins()
                .select(left_signed, signed_less_equal, unsigned_less_equal)
        }
        Instruction::Gt => builder
            .ins()
            .select(left_signed, signed_greater, unsigned_greater),
        Instruction::Ge => {
            builder
                .ins()
                .select(left_signed, signed_greater_equal, unsigned_greater_equal)
        }
        _ => unreachable!("integer comparison lowering handles only comparisons"),
    };

    let signed_left = match instruction {
        Instruction::Eq => builder.ins().band(left_nonnegative, equal),
        Instruction::Ne => builder.ins().bor(left_negative, not_equal),
        Instruction::Lt => builder
            .ins()
            .select(left_negative, true_value, unsigned_less),
        Instruction::Le => builder
            .ins()
            .select(left_negative, true_value, unsigned_less_equal),
        Instruction::Gt => builder
            .ins()
            .select(left_negative, false_value, unsigned_greater),
        Instruction::Ge => builder
            .ins()
            .select(left_negative, false_value, unsigned_greater_equal),
        _ => unreachable!("integer comparison lowering handles only comparisons"),
    };
    let unsigned_left = match instruction {
        Instruction::Eq => builder.ins().band(right_nonnegative, equal),
        Instruction::Ne => builder.ins().bor(right_negative, not_equal),
        Instruction::Lt => builder
            .ins()
            .select(right_negative, false_value, unsigned_less),
        Instruction::Le => builder
            .ins()
            .select(right_negative, false_value, unsigned_less_equal),
        Instruction::Gt => builder
            .ins()
            .select(right_negative, true_value, unsigned_greater),
        Instruction::Ge => builder
            .ins()
            .select(right_negative, true_value, unsigned_greater_equal),
        _ => unreachable!("integer comparison lowering handles only comparisons"),
    };
    let mixed = builder
        .ins()
        .icmp(IntCC::NotEqual, left_signed, right_signed);
    let mixed_result = builder
        .ins()
        .select(left_signed, signed_left, unsigned_left);
    builder.ins().select(mixed, mixed_result, same_signedness)
}

#[allow(clippy::too_many_arguments)]
fn emit_numeric_compare(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
    known_tags: Option<(u64, u64)>,
    helpers: JitHelperRefs,
) {
    if let Some((left_tag, right_tag)) = known_tags
        && left_tag == right_tag
        && matches!(
            left_tag,
            TAG_INT | TAG_UINT | TAG_FLOAT | TAG_CHAR | TAG_BOOL | TAG_NIL
        )
        && (!matches!(left_tag, TAG_BOOL | TAG_NIL)
            || matches!(instruction, Instruction::Eq | Instruction::Ne))
    {
        let left = load_stack_field(builder, local_state, depth - 2, 8);
        let right = load_stack_field(builder, local_state, depth - 1, 8);
        let result = match (left_tag, instruction) {
            (TAG_INT, Instruction::Eq) => builder.ins().icmp(IntCC::Equal, left, right),
            (TAG_INT, Instruction::Ne) => builder.ins().icmp(IntCC::NotEqual, left, right),
            (TAG_INT, Instruction::Lt) => builder.ins().icmp(IntCC::SignedLessThan, left, right),
            (TAG_INT, Instruction::Le) => {
                builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, left, right)
            }
            (TAG_INT, Instruction::Gt) => builder.ins().icmp(IntCC::SignedGreaterThan, left, right),
            (TAG_INT, Instruction::Ge) => {
                builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, left, right)
            }
            (TAG_UINT | TAG_CHAR, Instruction::Eq) => builder.ins().icmp(IntCC::Equal, left, right),
            (TAG_UINT | TAG_CHAR, Instruction::Ne) => {
                builder.ins().icmp(IntCC::NotEqual, left, right)
            }
            (TAG_UINT | TAG_CHAR, Instruction::Lt) => {
                builder.ins().icmp(IntCC::UnsignedLessThan, left, right)
            }
            (TAG_UINT | TAG_CHAR, Instruction::Le) => {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThanOrEqual, left, right)
            }
            (TAG_UINT | TAG_CHAR, Instruction::Gt) => {
                builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right)
            }
            (TAG_UINT | TAG_CHAR, Instruction::Ge) => {
                builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, left, right)
            }
            (TAG_FLOAT, Instruction::Eq) => {
                let left = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
                let right = builder
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                builder.ins().fcmp(FloatCC::Equal, left, right)
            }
            (TAG_FLOAT, Instruction::Ne) => {
                let left = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
                let right = builder
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                builder.ins().fcmp(FloatCC::NotEqual, left, right)
            }
            (TAG_FLOAT, Instruction::Lt) => {
                let left = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
                let right = builder
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                builder.ins().fcmp(FloatCC::LessThan, left, right)
            }
            (TAG_FLOAT, Instruction::Le) => {
                let left = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
                let right = builder
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                builder.ins().fcmp(FloatCC::LessThanOrEqual, left, right)
            }
            (TAG_FLOAT, Instruction::Gt) => {
                let left = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
                let right = builder
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                builder.ins().fcmp(FloatCC::GreaterThan, left, right)
            }
            (TAG_FLOAT, Instruction::Ge) => {
                let left = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
                let right = builder
                    .ins()
                    .bitcast(types::F64, MemFlagsData::new(), right);
                builder.ins().fcmp(FloatCC::GreaterThanOrEqual, left, right)
            }
            (TAG_BOOL, Instruction::Eq) => builder.ins().icmp(IntCC::Equal, left, right),
            (TAG_BOOL, Instruction::Ne) => builder.ins().icmp(IntCC::NotEqual, left, right),
            (TAG_NIL, Instruction::Eq) => builder.ins().iconst(types::I8, 1),
            (TAG_NIL, Instruction::Ne) => builder.ins().iconst(types::I8, 0),
            _ => unreachable!("filtered comparison type and operator"),
        };
        emit_compare_result(builder, local_state, depth, next, result);
        return;
    }
    let left_tag = load_stack_field(builder, local_state, depth - 2, 0);
    let right_tag = load_stack_field(builder, local_state, depth - 1, 0);
    let left_is_int = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_INT as i64);
    let right_is_int = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_INT as i64);
    let both_int = builder.ins().band(left_is_int, right_is_int);
    let left_is_uint = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_UINT as i64);
    let right_is_uint = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_UINT as i64);
    let both_uint = builder.ins().band(left_is_uint, right_is_uint);
    let int_uint = builder.ins().band(left_is_int, right_is_uint);
    let uint_int = builder.ins().band(left_is_uint, right_is_int);
    let mixed_int_uint = builder.ins().bor(int_uint, uint_int);
    let left_is_float = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_FLOAT as i64);
    let right_is_float = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_FLOAT as i64);
    let both_float = builder.ins().band(left_is_float, right_is_float);
    let int_float = builder.ins().band(left_is_int, right_is_float);
    let float_int = builder.ins().band(left_is_float, right_is_int);
    let mixed_int_float = builder.ins().bor(int_float, float_int);
    let uint_float = builder.ins().band(left_is_uint, right_is_float);
    let float_uint = builder.ins().band(left_is_float, right_is_uint);
    let mixed_uint_float = builder.ins().bor(uint_float, float_uint);
    let left_is_nil = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_NIL as i64);
    let right_is_nil = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_NIL as i64);
    let left_is_char = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_CHAR as i64);
    let right_is_char = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_CHAR as i64);
    let both_char = builder.ins().band(left_is_char, right_is_char);
    let left_is_bool = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_BOOL as i64);
    let right_is_bool = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_BOOL as i64);
    let both_bool = builder.ins().band(left_is_bool, right_is_bool);
    let bool_equality = matches!(instruction, Instruction::Eq | Instruction::Ne);
    let both_comparable_bool = if bool_equality {
        both_bool
    } else {
        builder.ins().iconst(types::I8, 0)
    };
    let integer = builder.ins().bor(both_int, both_uint);
    let integer = builder.ins().bor(integer, both_char);
    let integer = builder.ins().bor(integer, both_comparable_bool);
    let base_valid = builder.ins().bor(integer, both_float);
    let base_valid = builder.ins().bor(base_valid, mixed_int_uint);
    let mixed_float = builder.ins().bor(mixed_int_float, mixed_uint_float);
    let check_mixed = builder.create_block();
    let check_nil = builder.create_block();
    let mixed_float_block = builder.create_block();
    let regular_block = builder.create_block();
    let nil_block = builder.create_block();
    let nil_valid_block = builder.create_block();
    let check_bigint = builder.create_block();
    let bigint_block = builder.create_block();
    let fallback = builder.create_block();
    builder
        .ins()
        .brif(base_valid, regular_block, &[], check_mixed, &[]);

    builder.switch_to_block(check_mixed);
    builder
        .ins()
        .brif(mixed_float, mixed_float_block, &[], check_nil, &[]);

    builder.switch_to_block(check_nil);
    let nil_pair = if bool_equality {
        builder.ins().bor(left_is_nil, right_is_nil)
    } else {
        builder.ins().iconst(types::I8, 0)
    };
    builder
        .ins()
        .brif(nil_pair, nil_block, &[], check_bigint, &[]);

    builder.switch_to_block(check_bigint);
    let left_small_bigint = small_bigint_tag_test(builder, left_tag);
    let right_small_bigint = small_bigint_tag_test(builder, right_tag);
    let both_small_bigint = builder.ins().band(left_small_bigint, right_small_bigint);
    let left_bigint = bigint_tag_test(builder, left_tag);
    let right_bigint = bigint_tag_test(builder, right_tag);
    let both_bigint = builder.ins().band(left_bigint, right_bigint);
    let needs_bigint_helper = builder.ins().icmp_imm_s(IntCC::Equal, both_small_bigint, 0);
    let large_bigint_pair = builder.ins().band(both_bigint, needs_bigint_helper);
    let left_numeric = builder.ins().bor(left_is_int, left_is_uint);
    let left_numeric = builder.ins().bor(left_numeric, left_is_float);
    let right_numeric = builder.ins().bor(right_is_int, right_is_uint);
    let right_numeric = builder.ins().bor(right_numeric, right_is_float);
    let left_bigint_pair = builder.ins().band(left_small_bigint, right_numeric);
    let right_bigint_pair = builder.ins().band(right_small_bigint, left_numeric);
    let bigint_pair = builder.ins().bor(left_bigint_pair, right_bigint_pair);
    let bigint_pair = builder.ins().bor(bigint_pair, both_small_bigint);
    let bigint_helper_block = builder.create_block();
    let bigint_fast_check = builder.create_block();
    builder.ins().brif(
        large_bigint_pair,
        bigint_helper_block,
        &[],
        bigint_fast_check,
        &[],
    );
    builder.switch_to_block(bigint_helper_block);
    let depth_value = iconst_i64(builder, depth as i64);
    let operation_value = iconst_i32(builder, bigint_comparison_code(instruction));
    emit_runtime_helper(
        builder,
        frame,
        local_state,
        pc,
        depth,
        next,
        helpers.bigint_compare,
        &[depth_value, operation_value],
        depth - 1,
    );
    builder.switch_to_block(bigint_fast_check);
    builder
        .ins()
        .brif(bigint_pair, bigint_block, &[], fallback, &[]);

    builder.switch_to_block(fallback);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );

    builder.switch_to_block(nil_block);
    let left_uninitialized =
        builder
            .ins()
            .icmp_imm_s(IntCC::Equal, left_tag, TAG_UNINITIALIZED as i64);
    let right_uninitialized =
        builder
            .ins()
            .icmp_imm_s(IntCC::Equal, right_tag, TAG_UNINITIALIZED as i64);
    let uninitialized = builder.ins().bor(left_uninitialized, right_uninitialized);
    let left_error = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_ERROR as i64);
    let right_error = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_ERROR as i64);
    let error = builder.ins().bor(left_error, right_error);
    let invalid = builder.ins().bor(uninitialized, error);
    builder
        .ins()
        .brif(invalid, fallback, &[], nil_valid_block, &[]);

    builder.switch_to_block(nil_valid_block);
    let both_nil = builder.ins().band(left_is_nil, right_is_nil);
    let result = if matches!(instruction, Instruction::Eq) {
        both_nil
    } else {
        builder.ins().icmp_imm_s(IntCC::Equal, both_nil, 0)
    };
    emit_compare_result(builder, local_state, depth, next, result);

    builder.switch_to_block(bigint_block);
    let left_payload = load_stack_field(builder, local_state, depth - 2, 8);
    let right_payload = load_stack_field(builder, local_state, depth - 1, 8);
    let has_float = builder.ins().bor(left_is_float, right_is_float);
    let bigint_float_block = builder.create_block();
    let bigint_integer_block = builder.create_block();
    builder.ins().brif(
        has_float,
        bigint_float_block,
        &[],
        bigint_integer_block,
        &[],
    );

    builder.switch_to_block(bigint_float_block);
    let integer_payload = builder
        .ins()
        .select(left_is_float, right_payload, left_payload);
    let float_payload = builder
        .ins()
        .select(left_is_float, left_payload, right_payload);
    let float_value = builder
        .ins()
        .bitcast(types::F64, MemFlagsData::new(), float_payload);
    let left_small_signed = small_bigint_signed_tag_test(builder, left_tag);
    let right_small_signed = small_bigint_signed_tag_test(builder, right_tag);
    let left_signed = builder.ins().bor(left_is_int, left_small_signed);
    let right_signed = builder.ins().bor(right_is_int, right_small_signed);
    let is_signed = builder.ins().bor(left_signed, right_signed);
    let signed_block = builder.create_block();
    let unsigned_block = builder.create_block();
    builder
        .ins()
        .brif(is_signed, signed_block, &[], unsigned_block, &[]);

    builder.switch_to_block(signed_block);
    let (less, equal, greater) =
        emit_integer_float_ordering(builder, integer_payload, float_value, true);
    let result = emit_order_predicate(builder, instruction, less, equal, greater, left_is_float);
    emit_compare_result(builder, local_state, depth, next, result);

    builder.switch_to_block(unsigned_block);
    let (less, equal, greater) =
        emit_integer_float_ordering(builder, integer_payload, float_value, false);
    let result = emit_order_predicate(builder, instruction, less, equal, greater, left_is_float);
    emit_compare_result(builder, local_state, depth, next, result);

    builder.switch_to_block(bigint_integer_block);
    let left_small_signed = small_bigint_signed_tag_test(builder, left_tag);
    let right_small_signed = small_bigint_signed_tag_test(builder, right_tag);
    let left_signed = builder.ins().bor(left_is_int, left_small_signed);
    let right_signed = builder.ins().bor(right_is_int, right_small_signed);
    let result = emit_integer_comparison(
        builder,
        instruction,
        left_payload,
        right_payload,
        left_signed,
        right_signed,
    );
    emit_compare_result(builder, local_state, depth, next, result);

    builder.switch_to_block(mixed_float_block);
    let left_payload = load_stack_field(builder, local_state, depth - 2, 8);
    let right_payload = load_stack_field(builder, local_state, depth - 1, 8);
    let integer_payload = builder
        .ins()
        .select(left_is_float, right_payload, left_payload);
    let float_payload = builder
        .ins()
        .select(left_is_float, left_payload, right_payload);
    let float_value = builder
        .ins()
        .bitcast(types::F64, MemFlagsData::new(), float_payload);
    let is_signed = builder.ins().bor(left_is_int, right_is_int);
    let signed_block = builder.create_block();
    let unsigned_block = builder.create_block();
    builder
        .ins()
        .brif(is_signed, signed_block, &[], unsigned_block, &[]);

    builder.switch_to_block(signed_block);
    let (less, equal, greater) =
        emit_integer_float_ordering(builder, integer_payload, float_value, true);
    let result = emit_order_predicate(builder, instruction, less, equal, greater, left_is_float);
    emit_compare_result(builder, local_state, depth, next, result);

    builder.switch_to_block(unsigned_block);
    let (less, equal, greater) =
        emit_integer_float_ordering(builder, integer_payload, float_value, false);
    let result = emit_order_predicate(builder, instruction, less, equal, greater, left_is_float);
    emit_compare_result(builder, local_state, depth, next, result);

    builder.switch_to_block(regular_block);
    let left = load_stack_field(builder, local_state, depth - 2, 8);
    let right = load_stack_field(builder, local_state, depth - 1, 8);
    let integer_condition = match instruction {
        Instruction::Eq => builder.ins().icmp(IntCC::Equal, left, right),
        Instruction::Ne => builder.ins().icmp(IntCC::NotEqual, left, right),
        Instruction::Lt => {
            let signed = builder.ins().icmp(IntCC::SignedLessThan, left, right);
            let unsigned = builder.ins().icmp(IntCC::UnsignedLessThan, left, right);
            let unsigned_type = builder.ins().bor(both_uint, both_char);
            builder.ins().select(unsigned_type, unsigned, signed)
        }
        Instruction::Le => {
            let signed = builder
                .ins()
                .icmp(IntCC::SignedLessThanOrEqual, left, right);
            let unsigned = builder
                .ins()
                .icmp(IntCC::UnsignedLessThanOrEqual, left, right);
            let unsigned_type = builder.ins().bor(both_uint, both_char);
            builder.ins().select(unsigned_type, unsigned, signed)
        }
        Instruction::Gt => {
            let signed = builder.ins().icmp(IntCC::SignedGreaterThan, left, right);
            let unsigned = builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right);
            let unsigned_type = builder.ins().bor(both_uint, both_char);
            builder.ins().select(unsigned_type, unsigned, signed)
        }
        Instruction::Ge => {
            let signed = builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, left, right);
            let unsigned = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, left, right);
            let unsigned_type = builder.ins().bor(both_uint, both_char);
            builder.ins().select(unsigned_type, unsigned, signed)
        }
        _ => unreachable!("integer comparison lowering handles only comparisons"),
    };
    let zero = builder.ins().iconst(types::I64, 0);
    let left_negative = builder.ins().icmp(IntCC::SignedLessThan, left, zero);
    let right_negative = builder.ins().icmp(IntCC::SignedLessThan, right, zero);
    let equal = builder.ins().icmp(IntCC::Equal, left, right);
    let not_equal = builder.ins().icmp(IntCC::NotEqual, left, right);
    let less = builder.ins().icmp(IntCC::UnsignedLessThan, left, right);
    let less_equal = builder
        .ins()
        .icmp(IntCC::UnsignedLessThanOrEqual, left, right);
    let greater = builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right);
    let greater_equal = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, left, right);
    let true_value = builder.ins().iconst(types::I8, 1);
    let false_value = builder.ins().iconst(types::I8, 0);
    let left_nonnegative = builder
        .ins()
        .icmp(IntCC::SignedGreaterThanOrEqual, left, zero);
    let right_nonnegative = builder
        .ins()
        .icmp(IntCC::SignedGreaterThanOrEqual, right, zero);
    let int_uint_condition = match instruction {
        Instruction::Eq => builder.ins().band(left_nonnegative, equal),
        Instruction::Ne => builder.ins().bor(left_negative, not_equal),
        Instruction::Lt => builder.ins().select(left_negative, true_value, less),
        Instruction::Le => builder.ins().select(left_negative, true_value, less_equal),
        Instruction::Gt => builder.ins().select(left_negative, false_value, greater),
        Instruction::Ge => builder
            .ins()
            .select(left_negative, false_value, greater_equal),
        _ => unreachable!("integer comparison lowering handles only comparisons"),
    };
    let uint_int_condition = match instruction {
        Instruction::Eq => builder.ins().band(right_nonnegative, equal),
        Instruction::Ne => builder.ins().bor(right_negative, not_equal),
        Instruction::Lt => builder.ins().select(right_negative, false_value, less),
        Instruction::Le => builder
            .ins()
            .select(right_negative, false_value, less_equal),
        Instruction::Gt => builder.ins().select(right_negative, true_value, greater),
        Instruction::Ge => builder
            .ins()
            .select(right_negative, true_value, greater_equal),
        _ => unreachable!("integer comparison lowering handles only comparisons"),
    };
    let mixed_condition = builder
        .ins()
        .select(int_uint, int_uint_condition, uint_int_condition);
    let integer_condition =
        builder
            .ins()
            .select(mixed_int_uint, mixed_condition, integer_condition);
    let left_float = builder.ins().bitcast(types::F64, MemFlagsData::new(), left);
    let right_float = builder
        .ins()
        .bitcast(types::F64, MemFlagsData::new(), right);
    let float_condition = match instruction {
        Instruction::Eq => builder.ins().fcmp(FloatCC::Equal, left_float, right_float),
        Instruction::Ne => builder
            .ins()
            .fcmp(FloatCC::NotEqual, left_float, right_float),
        Instruction::Lt => builder
            .ins()
            .fcmp(FloatCC::LessThan, left_float, right_float),
        Instruction::Le => builder
            .ins()
            .fcmp(FloatCC::LessThanOrEqual, left_float, right_float),
        Instruction::Gt => builder
            .ins()
            .fcmp(FloatCC::GreaterThan, left_float, right_float),
        Instruction::Ge => builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, left_float, right_float),
        _ => unreachable!("float comparison lowering handles only comparisons"),
    };
    let condition = builder
        .ins()
        .select(both_float, float_condition, integer_condition);
    let payload = builder.ins().uextend(types::I64, condition);
    store_stack_const(
        builder,
        local_state,
        depth - 2,
        JitSlot {
            tag: TAG_BOOL,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, depth - 2, 8, payload);
    builder.ins().jump(next, &[]);
}

/// Writes a boolean comparison result and consumes the two input stack slots.
fn emit_compare_result(
    builder: &mut FunctionBuilder<'_>,
    local_state: &JitLocalState,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    result: cranelift_codegen::ir::Value,
) {
    let payload = builder.ins().uextend(types::I64, result);
    store_stack_const(
        builder,
        local_state,
        depth - 2,
        JitSlot {
            tag: TAG_BOOL,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, depth - 2, 8, payload);
    builder.ins().jump(next, &[]);
}

/// Mirrors `compare_int_float` and `compare_uint_float` in `value.rs` without
/// rounding the integer to f64. Saturating conversions are safe for every f64;
/// the in-range predicates ensure their saturated values only affect the
/// fallback ordering that is discarded for out-of-range inputs.
fn emit_integer_float_ordering(
    builder: &mut FunctionBuilder<'_>,
    integer: cranelift_codegen::ir::Value,
    float: cranelift_codegen::ir::Value,
    signed: bool,
) -> (
    cranelift_codegen::ir::Value,
    cranelift_codegen::ir::Value,
    cranelift_codegen::ir::Value,
) {
    let zero = builder.ins().f64const(Ieee64::with_bits(0));
    let ordered = builder.ins().fcmp(FloatCC::Equal, float, float);
    let (
        lower_bound,
        upper_bound,
        below,
        above,
        integer_less,
        integer_equal,
        integer_greater,
        truncated_float,
    ) = if signed {
        let lower_bits = Ieee64::with_bits(0xc3e0_0000_0000_0000);
        let upper_bits = Ieee64::with_bits(0x43e0_0000_0000_0000);
        let lower = builder.ins().f64const(lower_bits);
        let upper = builder.ins().f64const(upper_bits);
        let below = builder.ins().fcmp(FloatCC::LessThan, float, lower);
        let above = builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, float, upper);
        let truncated = builder.ins().fcvt_to_sint_sat(types::I64, float);
        let integer_less = builder
            .ins()
            .icmp(IntCC::SignedLessThan, integer, truncated);
        let integer_equal = builder.ins().icmp(IntCC::Equal, integer, truncated);
        let integer_greater = builder
            .ins()
            .icmp(IntCC::SignedGreaterThan, integer, truncated);
        let truncated_float = builder.ins().fcvt_from_sint(types::F64, truncated);
        (
            lower,
            upper,
            below,
            above,
            integer_less,
            integer_equal,
            integer_greater,
            truncated_float,
        )
    } else {
        let lower = zero;
        let upper = builder
            .ins()
            .f64const(Ieee64::with_bits(0x43f0_0000_0000_0000));
        let below = builder.ins().fcmp(FloatCC::LessThan, float, lower);
        let above = builder
            .ins()
            .fcmp(FloatCC::GreaterThanOrEqual, float, upper);
        let truncated = builder.ins().fcvt_to_uint_sat(types::I64, float);
        let integer_less = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, integer, truncated);
        let integer_equal = builder.ins().icmp(IntCC::Equal, integer, truncated);
        let integer_greater = builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThan, integer, truncated);
        let truncated_float = builder.ins().fcvt_from_uint(types::F64, truncated);
        (
            lower,
            upper,
            below,
            above,
            integer_less,
            integer_equal,
            integer_greater,
            truncated_float,
        )
    };
    let in_range = builder
        .ins()
        .fcmp(FloatCC::GreaterThanOrEqual, float, lower_bound);
    let below_upper = builder.ins().fcmp(FloatCC::LessThan, float, upper_bound);
    let in_range = builder.ins().band(in_range, below_upper);
    let in_range = builder.ins().band(in_range, ordered);
    let truncated_less = builder
        .ins()
        .fcmp(FloatCC::LessThan, truncated_float, float);
    let truncated_equal = builder.ins().fcmp(FloatCC::Equal, truncated_float, float);
    let truncated_greater = builder
        .ins()
        .fcmp(FloatCC::GreaterThan, truncated_float, float);
    let inner_less = builder
        .ins()
        .select(integer_equal, truncated_less, integer_less);
    let inner_greater = builder
        .ins()
        .select(integer_equal, truncated_greater, integer_greater);
    let in_range_less = builder.ins().band(in_range, inner_less);
    let less = builder.ins().bor(above, in_range_less);
    let in_range_greater = builder.ins().band(in_range, inner_greater);
    let greater = builder.ins().bor(below, in_range_greater);
    let trunc_match = builder.ins().band(integer_equal, truncated_equal);
    let equal = builder.ins().band(in_range, trunc_match);
    (less, equal, greater)
}

fn emit_order_predicate(
    builder: &mut FunctionBuilder<'_>,
    instruction: &Instruction,
    mut less: cranelift_codegen::ir::Value,
    equal: cranelift_codegen::ir::Value,
    mut greater: cranelift_codegen::ir::Value,
    reverse: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let swapped_less = builder.ins().select(reverse, greater, less);
    let swapped_greater = builder.ins().select(reverse, less, greater);
    less = swapped_less;
    greater = swapped_greater;
    match instruction {
        Instruction::Eq => equal,
        Instruction::Ne => builder.ins().icmp_imm_s(IntCC::Equal, equal, 0),
        Instruction::Lt => less,
        Instruction::Le => builder.ins().bor(less, equal),
        Instruction::Gt => greater,
        Instruction::Ge => builder.ins().bor(greater, equal),
        _ => unreachable!("ordering predicate accepts only comparisons"),
    }
}

fn emit_bool_binary(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    next: cranelift_codegen::ir::Block,
    instruction: &Instruction,
) {
    let left_tag = load_stack_field(builder, local_state, depth - 2, 0);
    let right_tag = load_stack_field(builder, local_state, depth - 1, 0);
    let left_bool = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, left_tag, TAG_BOOL as i64);
    let right_bool = builder
        .ins()
        .icmp_imm_s(IntCC::Equal, right_tag, TAG_BOOL as i64);
    let both_bool = builder.ins().band(left_bool, right_bool);
    let valid = builder.create_block();
    let fallback = builder.create_block();
    builder.ins().brif(both_bool, valid, &[], fallback, &[]);
    builder.switch_to_block(fallback);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );
    builder.switch_to_block(valid);
    let left = load_stack_field(builder, local_state, depth - 2, 8);
    let right = load_stack_field(builder, local_state, depth - 1, 8);
    let result = match instruction {
        Instruction::And => builder.ins().band(left, right),
        Instruction::Or => builder.ins().bor(left, right),
        Instruction::Xor => builder.ins().bxor(left, right),
        _ => unreachable!("emit_bool_binary is used only for boolean operations"),
    };
    store_stack_const(
        builder,
        local_state,
        depth - 2,
        JitSlot {
            tag: TAG_BOOL,
            payload: 0,
        },
    );
    store_stack_field(builder, local_state, depth - 2, 8, result);
    builder.ins().jump(next, &[]);
}

fn emit_jump(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    target: usize,
    target_depth: usize,
    target_block: cranelift_codegen::ir::Block,
) {
    if target > pc {
        builder.ins().jump(target_block, &[]);
        return;
    }

    let counter_address = builder
        .ins()
        .iadd_imm_s(frame, offset_of!(JitFrame, backedge_count) as i64);
    let count = builder
        .ins()
        .load(types::I64, MemFlagsData::trusted(), counter_address, 0);
    let next_count = builder.ins().iadd_imm_s(count, 1);
    builder
        .ins()
        .store(MemFlagsData::trusted(), next_count, counter_address, 0);

    let budget = load_frame(
        builder,
        frame,
        offset_of!(JitFrame, backedge_budget) as i32,
        types::I64,
    );
    let bounded = builder.ins().icmp_imm_s(IntCC::NotEqual, budget, 0);
    let profile_path = builder.create_block();
    let unbounded_path = builder.create_block();
    builder
        .ins()
        .brif(bounded, profile_path, &[], unbounded_path, &[]);
    builder.switch_to_block(unbounded_path);
    builder.ins().jump(target_block, &[]);

    builder.switch_to_block(profile_path);
    let one = builder.ins().iconst(types::I64, 1);
    let remaining = builder.ins().isub(budget, one);
    let budget_address = builder
        .ins()
        .iadd_imm_s(frame, offset_of!(JitFrame, backedge_budget) as i64);
    builder
        .ins()
        .store(MemFlagsData::trusted(), remaining, budget_address, 0);
    let exhausted = builder.ins().icmp_imm_s(IntCC::Equal, remaining, 0);
    let slice_end = builder.create_block();
    let keep_running = builder.create_block();
    builder
        .ins()
        .brif(exhausted, slice_end, &[], keep_running, &[]);
    builder.switch_to_block(slice_end);
    emit_exit(
        builder,
        frame,
        local_state,
        target,
        target_depth,
        EXIT_RESUME,
        REASON_PROFILE_SLICE,
    );
    builder.switch_to_block(keep_running);
    builder.ins().jump(target_block, &[]);
}

#[allow(clippy::too_many_arguments)]
fn emit_conditional_jump(
    builder: &mut FunctionBuilder<'_>,
    frame: cranelift_codegen::ir::Value,
    local_state: &JitLocalState,
    pc: usize,
    depth: usize,
    target: usize,
    next: Option<cranelift_codegen::ir::Block>,
    blocks: &HashMap<usize, cranelift_codegen::ir::Block>,
    plan: &FunctionPlan,
    instruction: &Instruction,
) -> Result<(), String> {
    let next = next.ok_or("conditional fallthrough is missing")?;
    let tag = load_stack_field(builder, local_state, depth - 1, 0);
    let is_bool = builder.ins().icmp_imm_s(IntCC::Equal, tag, TAG_BOOL as i64);
    let valid = builder.create_block();
    let fallback = builder.create_block();
    builder.ins().brif(is_bool, valid, &[], fallback, &[]);
    builder.switch_to_block(fallback);
    emit_exit(
        builder,
        frame,
        local_state,
        pc,
        depth,
        EXIT_RESUME,
        REASON_NORMAL,
    );

    builder.switch_to_block(valid);
    let payload = load_stack_field(builder, local_state, depth - 1, 8);
    let take_target = match instruction {
        Instruction::JumpIfFalse(_) => builder.ins().icmp_imm_s(IntCC::Equal, payload, 0),
        Instruction::JumpIfTrue(_) => builder.ins().icmp_imm_s(IntCC::NotEqual, payload, 0),
        _ => unreachable!("conditional jump instruction expected"),
    };
    let target_block = blocks
        .get(&target)
        .copied()
        .ok_or("conditional target has no bytecode block")?;
    let taken = builder.create_block();
    let fallthrough = builder.create_block();
    builder
        .ins()
        .brif(take_target, taken, &[], fallthrough, &[]);

    builder.switch_to_block(taken);
    emit_jump(
        builder,
        frame,
        local_state,
        pc,
        target,
        plan.stack_depths[target - plan.entry].ok_or("invalid conditional stack depth")?,
        target_block,
    );

    builder.switch_to_block(fallthrough);
    builder.ins().jump(next, &[]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jit_bytecode_blocks_coalesce_linear_instructions() {
        let artifact = CompiledProgram {
            instructions: vec![
                Instruction::Push(Value::Int(1)),
                Instruction::Store(0),
                Instruction::Add,
                Instruction::Push(Value::Int(2)),
                Instruction::Store(0),
                Instruction::JumpIfFalse(8),
                Instruction::Pop,
                Instruction::Jump(9),
                Instruction::Push(Value::Int(3)),
                Instruction::Return,
            ],
            ..CompiledProgram::default()
        };
        let plan = FunctionPlan {
            entry: 0,
            end: artifact.instructions.len(),
            instruction_count: artifact.instructions.len(),
            block_count: 4,
            edge_count: 0,
            phi_estimate: 0,
            local_count: 1,
            max_stack: 1,
            stack_depths: vec![Some(0); artifact.instructions.len()],
            osr_entries: vec![1],
            has_backedges: false,
            callable_from_jit: false,
        };
        let reachable = vec![true; artifact.instructions.len()];
        let type_states = infer_jit_type_states(&plan, &artifact);

        assert_eq!(
            jit_bytecode_block_starts(&plan, &artifact, &reachable, &type_states)
                .into_iter()
                .collect::<Vec<_>>(),
            [0, 1, 3, 6, 8, 9]
        );
    }

    #[test]
    fn jit_type_dataflow_tracks_checked_integer_loop_state() {
        let artifact = CompiledProgram {
            instructions: vec![
                Instruction::Store(0),
                Instruction::Load(0),
                Instruction::Push(Value::Int(1)),
                Instruction::Add,
                Instruction::Store(0),
                Instruction::Jump(1),
            ],
            callables: BTreeMap::from([(
                0,
                crate::backend::CallableSignature {
                    parameters: 1,
                    captures: 0,
                    asynchronous: false,
                    parameter_types: Some(vec![Some(RuntimeType::Int)]),
                    result_type: None,
                },
            )]),
            ..CompiledProgram::default()
        };
        let plan = FunctionPlan {
            entry: 0,
            end: artifact.instructions.len(),
            instruction_count: artifact.instructions.len(),
            block_count: 2,
            edge_count: 1,
            phi_estimate: 0,
            local_count: 1,
            max_stack: 2,
            stack_depths: vec![Some(1), Some(0), Some(1), Some(2), Some(1), Some(0)],
            osr_entries: vec![1],
            has_backedges: true,
            callable_from_jit: false,
        };

        let states = infer_jit_type_states(&plan, &artifact);
        let entry = states[0].as_ref().expect("entry state is reachable");
        assert_eq!(entry.stack, [Some(TAG_INT)]);
        let loop_state = states[1].as_ref().expect("loop state is reachable");
        assert_eq!(loop_state.locals, [Some(TAG_INT)]);
        let add_state = states[3].as_ref().expect("integer addition is reachable");
        assert_eq!(
            exact_stack_tag_pair(add_state, 0, 1),
            Some((TAG_INT, TAG_INT))
        );
        assert_eq!(
            states[1].as_ref().map(|state| state.locals[0]),
            Some(Some(TAG_INT))
        );
    }

    #[test]
    fn enum_type_tests_use_registered_tags_and_fall_back_past_the_code_cap() {
        let artifact_with_variants =
            |count: usize| {
                let mut artifact = CompiledProgram::default();
                for index in 0..count {
                    artifact.instructions.push(Instruction::Push(Value::TypeTag(
                        std::rc::Rc::from(format!("Signal::V{index}")),
                    )));
                }
                artifact
                    .instructions
                    .push(Instruction::IsType(ValueType::Enum(std::rc::Rc::from(
                        "Signal",
                    ))));
                artifact
            };

        let supported = artifact_with_variants(MAX_JIT_ENUM_VARIANTS);
        let supported_tags = VariantTags::from_program(&supported);
        assert_eq!(
            supported_tags
                .variant_tags(&ValueType::Enum(std::rc::Rc::from("Signal")))
                .map(|tags| tags.len()),
            Some(MAX_JIT_ENUM_VARIANTS)
        );

        let too_large = artifact_with_variants(MAX_JIT_ENUM_VARIANTS + 1);
        let too_large_tags = VariantTags::from_program(&too_large);
        assert!(
            too_large_tags
                .variant_tags(&ValueType::Enum(std::rc::Rc::from("Signal")))
                .is_none()
        );
        assert!(
            VariantTags::default()
                .variant_tags(&ValueType::Variant(std::rc::Rc::from("External::Value")))
                .is_none()
        );
    }

    fn candidate(profile: FunctionProfile, windows_this_run: u32) -> Candidate {
        let now = Instant::now();
        Candidate {
            profile,
            window_edges: 0,
            windows_this_run,
            window_started: now,
            last_backedge_at: (windows_this_run > 0).then_some(now),
            last_selection_window: 0,
            retry_after_backedges: 0,
            retry_after_calls: 0,
            last_selection_call: 0,
            compiled: None,
            disabled: false,
        }
    }

    #[test]
    fn adaptive_ape_uses_shared_calibration_without_stale_candidate_blocking() {
        let mut candidates = BTreeMap::new();
        // These calibrated records are historical: no candidate is active in
        // the current run, but their measurements remain useful to the model.
        for entry in 1..=MAX_BOOTSTRAP_PROBES {
            candidates.insert(
                entry,
                candidate(
                    FunctionProfile {
                        vm_ns_per_backedge: 10_000,
                        jit_ns_per_backedge: 5_000,
                        total_backedges: WINDOW_BACKEDGES * u64::from(REQUIRED_WINDOWS),
                        hot_series_ns: MAX_HORIZON_NS,
                        active_windows: REQUIRED_WINDOWS,
                        calibrated: true,
                        ..FunctionProfile::default()
                    },
                    0,
                ),
            );
        }
        // This candidate has no private JIT measurements. Once bootstrap is
        // exhausted, it must use the conservative shared speedup estimate.
        candidates.insert(
            MAX_BOOTSTRAP_PROBES + 1,
            candidate(
                FunctionProfile {
                    vm_ns_per_backedge: 10_000,
                    total_backedges: WINDOW_BACKEDGES * u64::from(REQUIRED_WINDOWS),
                    hot_series_ns: MAX_HORIZON_NS,
                    active_windows: REQUIRED_WINDOWS,
                    ..FunctionProfile::default()
                },
                REQUIRED_WINDOWS,
            ),
        );
        let entry = MAX_BOOTSTRAP_PROBES + 1;
        let plan = FunctionPlan {
            entry,
            end: entry + 16,
            instruction_count: 8,
            block_count: 2,
            edge_count: 1,
            phi_estimate: 0,
            local_count: 2,
            max_stack: 1,
            stack_depths: Vec::new(),
            osr_entries: vec![entry + 4],
            has_backedges: true,
            callable_from_jit: false,
        };
        let stale_entry = entry + 100;
        let mut stale_candidate = candidate(
            FunctionProfile {
                compile_ns: 1,
                vm_ns_per_backedge: 1_000_000,
                jit_ns_per_backedge: 1,
                total_backedges: WINDOW_BACKEDGES * u64::from(REQUIRED_WINDOWS),
                total_calls: 0,
                hot_series_ns: MAX_HORIZON_NS,
                active_windows: REQUIRED_WINDOWS,
                calibrated: true,
            },
            REQUIRED_WINDOWS,
        );
        stale_candidate.last_backedge_at =
            Some(Instant::now() - std::time::Duration::from_nanos(MAX_ACTIVITY_GAP_NS + 1));
        let stale_plan = FunctionPlan {
            entry: stale_entry,
            end: stale_entry + (plan.end - plan.entry),
            osr_entries: vec![stale_entry + 4],
            ..plan.clone()
        };
        candidates.insert(stale_entry, stale_candidate);
        let (compiler, _) = JitCompiler::new().expect("native JIT should initialize");
        let mut state = JitState {
            plans: BTreeMap::from([(entry, Arc::new(plan)), (stale_entry, Arc::new(stale_plan))]),
            variant_tags: VariantTags::default(),
            candidates,
            compiler,
            coefficients: Coefficients::seeded(),
            profile_key: String::new(),
            profile_path: None,
            bootstrap_probes: MAX_BOOTSTRAP_PROBES,
            compile_tokens_ns: TOKEN_CAPACITY_NS as i64,
            token_refill_at: Instant::now(),
        };

        let decision = state.candidate_decision(entry);
        assert!(matches!(
            decision,
            ApeDecision::Compile { priority, .. } if priority > 0.0
        ));
        assert_eq!(state.best_candidate(Instant::now()), Some(entry));
        state
            .candidates
            .get_mut(&stale_entry)
            .expect("stale candidate is present")
            .last_backedge_at = Some(Instant::now());
        assert_eq!(state.best_candidate(Instant::now()), Some(stale_entry));
    }

    #[test]
    fn adaptive_ape_charges_estimated_compile_cost_before_first_probe() {
        let entry = 100;
        let mut candidates = BTreeMap::new();
        for calibrated_entry in 1..=MAX_BOOTSTRAP_PROBES {
            candidates.insert(
                calibrated_entry,
                candidate(
                    FunctionProfile {
                        vm_ns_per_backedge: 10_000,
                        jit_ns_per_backedge: 5_000,
                        active_windows: REQUIRED_WINDOWS,
                        calibrated: true,
                        ..FunctionProfile::default()
                    },
                    0,
                ),
            );
        }
        candidates.insert(
            entry,
            candidate(
                FunctionProfile {
                    vm_ns_per_backedge: 10_000,
                    total_backedges: WINDOW_BACKEDGES * u64::from(REQUIRED_WINDOWS),
                    hot_series_ns: 40_000,
                    active_windows: REQUIRED_WINDOWS,
                    ..FunctionProfile::default()
                },
                REQUIRED_WINDOWS,
            ),
        );
        let plan = FunctionPlan {
            entry,
            end: entry + 16,
            instruction_count: 8,
            block_count: 2,
            edge_count: 1,
            phi_estimate: 0,
            local_count: 2,
            max_stack: 1,
            stack_depths: Vec::new(),
            osr_entries: vec![entry + 4],
            has_backedges: true,
            callable_from_jit: false,
        };
        let (compiler, _) = JitCompiler::new().expect("native JIT should initialize");
        let state = JitState {
            plans: BTreeMap::from([(entry, Arc::new(plan))]),
            variant_tags: VariantTags::default(),
            candidates,
            compiler,
            coefficients: Coefficients::seeded(),
            profile_key: String::new(),
            profile_path: None,
            bootstrap_probes: MAX_BOOTSTRAP_PROBES,
            compile_tokens_ns: TOKEN_CAPACITY_NS as i64,
            token_refill_at: Instant::now(),
        };

        assert!(matches!(
            state.candidate_decision(entry),
            ApeDecision::Profile("not-profitable")
        ));
    }

    #[test]
    fn compile_budget_records_oversized_compilation_as_refillable_debt() {
        let mut available = TOKEN_CAPACITY_NS as i64;
        let reserved = reserve_compile_budget(&mut available, TOKEN_CAPACITY_NS * 2)
            .expect("a full bucket should admit one oversized compile");
        assert_eq!(reserved, TOKEN_CAPACITY_NS);
        assert_eq!(available, 0);

        settle_compile_budget(&mut available, reserved, TOKEN_CAPACITY_NS * 2);
        assert_eq!(available, -(TOKEN_CAPACITY_NS as i64));
        assert_eq!(
            reserve_compile_budget(&mut available, 1),
            Err(-(TOKEN_CAPACITY_NS as i64))
        );
    }

    #[test]
    fn run_local_heat_is_not_written_to_profiles() {
        let profile = FunctionProfile {
            compile_ns: 10,
            vm_ns_per_backedge: 20,
            jit_ns_per_backedge: 5,
            total_backedges: 900,
            total_calls: 7,
            hot_series_ns: MAX_HORIZON_NS,
            active_windows: REQUIRED_WINDOWS,
            calibrated: true,
        };
        let encoded = serde_json::to_value(profile).expect("profile serializes");
        assert!(encoded.get("total_backedges").is_none());
        assert!(encoded.get("total_calls").is_none());
        assert!(encoded.get("hot_series_ns").is_none());
        assert!(encoded.get("active_windows").is_none());
        let restored: FunctionProfile =
            serde_json::from_value(encoded).expect("profile deserializes");
        assert_eq!(restored.total_backedges, 0);
        assert_eq!(restored.total_calls, 0);
        assert_eq!(restored.hot_series_ns, 0);
        assert_eq!(restored.active_windows, 0);
    }

    #[test]
    fn a_long_idle_gap_starts_a_new_hot_series() {
        let now = Instant::now();
        let mut candidate = candidate(
            FunctionProfile {
                total_backedges: 800,
                hot_series_ns: MAX_HORIZON_NS,
                active_windows: REQUIRED_WINDOWS,
                ..FunctionProfile::default()
            },
            REQUIRED_WINDOWS,
        );
        candidate.last_backedge_at =
            Some(now - std::time::Duration::from_nanos(MAX_ACTIVITY_GAP_NS + 1));

        assert!(!update_heat(&mut candidate, now));
        assert_eq!(candidate.profile.total_backedges, 801);
        assert_eq!(candidate.profile.hot_series_ns, 0);
        assert_eq!(candidate.profile.active_windows, 0);
        assert_eq!(candidate.windows_this_run, 0);
    }

    #[test]
    fn native_exit_decoder_rejects_unknown_status_and_wrong_boundary() {
        let artifact = CompiledProgram {
            instructions: vec![Instruction::Return],
            ..CompiledProgram::default()
        };
        let plan = FunctionPlan {
            entry: 0,
            end: 1,
            instruction_count: 1,
            block_count: 1,
            edge_count: 0,
            phi_estimate: 0,
            local_count: 0,
            max_stack: 1,
            stack_depths: vec![Some(1)],
            osr_entries: Vec::new(),
            has_backedges: false,
            callable_from_jit: false,
        };
        let frame = JitFrame {
            locals: std::ptr::null_mut(),
            stack: std::ptr::null_mut(),
            captures: std::ptr::null_mut(),
            capture_count: 0,
            roots: std::ptr::null_mut(),
            runtime_context: std::ptr::null_mut(),
            pc: 0,
            stack_len: 1,
            propagation_pending: 0,
            backedge_budget: 0,
            backedge_count: 0,
            exit_reason: REASON_NORMAL,
        };

        assert_eq!(
            decode_exit(&plan, &artifact, &frame, EXIT_COMPLETED),
            Ok(JitExit::Completed)
        );
        assert!(decode_exit(&plan, &artifact, &frame, 99).is_err());

        let mut wrong_reason = frame;
        wrong_reason.exit_reason = REASON_PROFILE_SLICE;
        assert!(decode_exit(&plan, &artifact, &wrong_reason, EXIT_COMPLETED).is_err());
    }

    #[test]
    fn invalid_persisted_cost_data_is_rejected() {
        let mut coefficients = Coefficients::seeded();
        assert!(coefficients.is_valid());
        coefficients.instructions = f64::NAN;
        assert!(!coefficients.is_valid());

        assert!(
            FunctionProfile {
                compile_ns: 10,
                vm_ns_per_backedge: 20,
                jit_ns_per_backedge: 5,
                calibrated: true,
                ..FunctionProfile::default()
            }
            .is_valid()
        );
        assert!(
            !FunctionProfile {
                calibrated: true,
                ..FunctionProfile::default()
            }
            .is_valid()
        );
    }
}
