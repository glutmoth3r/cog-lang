//! Generational handles and a tracing registry for runtime graph values.
//!
//! Values keep ordinary `Rc` aliasing at the Rust boundary. The registry stores
//! weak object references, traces their Cog-value edges, and breaks unreachable
//! graph cycles before the last strong references are released. Registry growth
//! is fallible; allocation of the `Rc` payload still follows the host allocator's
//! infallible API.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::{Rc, Weak};

use crate::error::RuntimeError;
use crate::value::{Value, WeakTaskHandle};

pub(crate) trait HeapObject: Any {
    fn visit_values(&self, visit: &mut dyn FnMut(&Value));
    fn clear_values(&self);

    fn has_heap_children(&self) -> Result<bool, RuntimeError> {
        let mut has_children = false;
        let mut traversal_error = None;
        self.visit_values(&mut |value| {
            if traversal_error.is_none()
                && let Err(error) = value.visit_heap_objects(&mut |_| has_children = true)
            {
                traversal_error = Some(error);
            }
        });
        traversal_error.map_or(Ok(has_children), Err)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ObjectId {
    slot: usize,
    generation: u64,
}

#[derive(Clone)]
pub(crate) struct ObjectKey {
    registry: Rc<RefCell<Registry>>,
    id: ObjectId,
}

pub(crate) struct TaskRootSet<'a> {
    pub id: usize,
    pub lease: &'a WeakTaskHandle,
    pub strong_count: usize,
    pub values: Vec<&'a Value>,
}

struct TaskGraphNode {
    task_id: Option<usize>,
    object: Option<ObjectKey>,
    strong_count: usize,
    incoming: usize,
    edges: Vec<usize>,
}

struct Slot {
    generation: u64,
    active_index: usize,
    object: Option<Weak<dyn HeapObject>>,
}

#[derive(Default)]
struct Registry {
    slots: Vec<Slot>,
    active_slots: Vec<usize>,
    free: Vec<usize>,
    allocations_since_collection: usize,
    collection_threshold: usize,
}

impl Registry {
    fn retire(&mut self, id: ObjectId) {
        let slot = &mut self.slots[id.slot];
        if slot.generation != id.generation || slot.object.is_none() {
            return;
        }
        slot.object = None;
        let active_index = slot.active_index;
        self.active_slots.swap_remove(active_index);
        if let Some(&moved) = self.active_slots.get(active_index) {
            self.slots[moved].active_index = active_index;
        }
        let slot = &mut self.slots[id.slot];
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            // A failed free-list growth only prevents reuse of this dead slot.
            if self.free.try_reserve(1).is_ok() {
                self.free.push(id.slot);
            }
        }
    }
}

thread_local! {
    static REGISTRY: Rc<RefCell<Registry>> = Rc::default();
    static COLLECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    // Mirrors registry counters so each VM step avoids borrowing REGISTRY.
    static COLLECTION_DUE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ACTIVE_OBJECT_LIMIT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    static SNAPSHOT_SLOT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Temporarily enforce the strictest active managed-object limit on this
/// thread. Nested executions share the thread-local heap, so an inner context
/// must not relax an outer execution's configured cap.
pub(crate) struct ObjectLimitGuard(Option<Option<usize>>);

#[inline]
pub(crate) fn limit_object_allocations(limit: Option<usize>) -> ObjectLimitGuard {
    let previous = ACTIVE_OBJECT_LIMIT.with(|active| {
        let current = active.get();
        let effective = match (current, limit) {
            (Some(current), Some(limit)) => Some(current.min(limit)),
            (Some(current), None) => Some(current),
            (None, limit) => limit,
        };
        if current == effective {
            return None;
        }
        active.set(effective);
        Some(current)
    });
    ObjectLimitGuard(previous)
}

impl Drop for ObjectLimitGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            ACTIVE_OBJECT_LIMIT.with(|active| active.set(previous));
        }
    }
}

struct CollectionFlag;

impl Drop for CollectionFlag {
    fn drop(&mut self) {
        COLLECTING.with(|collecting| collecting.set(false));
    }
}

/// A typed, generational reference to an object managed by the current thread's
/// VM heap. Cloning a handle preserves ordinary Cog aliasing.
pub struct HeapRef<T: 'static> {
    object: Rc<T>,
    registry: Rc<RefCell<Registry>>,
    id: ObjectId,
    drop_hook: fn(&HeapRef<T>),
    marker: PhantomData<T>,
}

impl<T: 'static> Clone for HeapRef<T> {
    fn clone(&self) -> Self {
        Self {
            object: self.object.clone(),
            registry: self.registry.clone(),
            id: self.id,
            drop_hook: self.drop_hook,
            marker: PhantomData,
        }
    }
}

impl<T: 'static> std::fmt::Debug for HeapRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeapRef")
            .field("slot", &self.id.slot)
            .field("generation", &self.id.generation)
            .finish_non_exhaustive()
    }
}

#[allow(private_bounds)]
impl<T: HeapObject> HeapRef<T> {
    /// Allocate a runtime-managed object.
    ///
    /// # Panics
    /// Panics if the active managed-object limit is reached or the heap registry
    /// cannot grow. Use [`Self::try_new`] to handle these failures. Most
    /// embedders should obtain values from Cog or the SDK constructors instead
    /// of building graph nodes by hand.
    pub fn new(value: T) -> Self {
        Self::try_new(value).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Allocate a runtime-managed object, returning an error if the active
    /// managed-object limit is reached or the heap registry cannot grow. The
    /// Rust standard library does not expose fallible `Rc` allocation, so this
    /// does not convert host-wide out-of-memory into an error.
    pub fn try_new(value: T) -> Result<Self, RuntimeError> {
        REGISTRY.with(|registry| Self::try_new_in(registry.clone(), value))
    }

    fn try_new_in(registry: Rc<RefCell<Registry>>, value: T) -> Result<Self, RuntimeError> {
        let live_objects = || {
            registry
                .try_borrow()
                .map(|state| state.active_slots.len())
                .map_err(|_| RuntimeError::InvalidArgument {
                    op: "VM heap allocation",
                })
        };
        if let Some(limit) = ACTIVE_OBJECT_LIMIT.with(std::cell::Cell::get)
            && live_objects()? >= limit
        {
            // Reclaim unreachable cycles before rejecting. The allocation is
            // not attempted until a live registry slot is available.
            collect(&[])?;
            if live_objects()? >= limit {
                return Err(RuntimeError::ResourceLimit {
                    resource: "VM heap objects",
                });
            }
        }

        let mut registry_state =
            registry
                .try_borrow_mut()
                .map_err(|_| RuntimeError::InvalidArgument {
                    op: "VM heap allocation",
                })?;
        if registry_state.free.is_empty() {
            registry_state
                .slots
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "VM heap registry",
                })?;
        }
        registry_state
            .active_slots
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "VM heap registry",
            })?;

        // Reserve registry storage before allocating the Rc so a failed slot
        // growth cannot leave an unregistered managed object behind.
        let object = Rc::new(value);
        let erased: Rc<dyn HeapObject> = object.clone();
        let weak = Rc::downgrade(&erased);
        drop(erased);

        registry_state.allocations_since_collection = registry_state
            .allocations_since_collection
            .saturating_add(1);
        let collection_due = registry_state.allocations_since_collection
            >= registry_state.collection_threshold.max(256);
        COLLECTION_DUE.with(|due| due.set(collection_due));
        let active_index = registry_state.active_slots.len();
        let id = if let Some(slot) = registry_state.free.pop() {
            let generation = registry_state.slots[slot].generation;
            registry_state.slots[slot].object = Some(weak);
            registry_state.slots[slot].active_index = active_index;
            ObjectId { slot, generation }
        } else {
            let slot = registry_state.slots.len();
            registry_state.slots.push(Slot {
                generation: 0,
                active_index,
                object: Some(weak),
            });
            ObjectId {
                slot,
                generation: 0,
            }
        };
        registry_state.active_slots.push(id.slot);
        drop(registry_state);

        Ok(Self {
            object,
            registry,
            id,
            drop_hook: collect_on_drop::<T>,
            marker: PhantomData,
        })
    }

    #[cfg(test)]
    pub(crate) fn id(&self) -> ObjectId {
        self.id
    }

    pub(crate) fn key(&self) -> ObjectKey {
        ObjectKey {
            registry: self.registry.clone(),
            id: self.id,
        }
    }

    pub(crate) fn strong_count(&self) -> usize {
        Rc::strong_count(&self.object)
    }

    pub(crate) fn identity(&self) -> usize {
        Rc::as_ptr(&self.object) as usize
    }
}

impl<T: 'static> Deref for HeapRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.object
    }
}

impl<T: 'static> Drop for HeapRef<T> {
    fn drop(&mut self) {
        (self.drop_hook)(self);
    }
}

// Pin immediate children before clearing a uniquely owned node. Releasing the
// pins through this work list prevents recursive destruction without tracing
// unrelated live objects. Shared nodes (including cycles) stay for tracing GC.
type PendingDrop = (Rc<RefCell<Registry>>, ObjectId, Rc<dyn HeapObject>);

fn pin_children(object: &dyn HeapObject, work: &mut Vec<PendingDrop>) -> Result<(), RuntimeError> {
    let mut result = Ok(());
    object.visit_values(&mut |value| {
        if result.is_err() {
            return;
        }
        let traversal = value.visit_heap_objects(&mut |child| {
            if result.is_err() {
                return;
            }
            if work.try_reserve(1).is_err() {
                result = Err(RuntimeError::ResourceLimit {
                    resource: "VM heap cleanup",
                });
                return;
            }
            let object = {
                let state = child.registry.borrow();
                state
                    .slots
                    .get(child.id.slot)
                    .filter(|slot| slot.generation == child.id.generation)
                    .and_then(|slot| slot.object.as_ref())
                    .and_then(Weak::upgrade)
            };
            if let Some(object) = object {
                work.push((child.registry, child.id, object));
            }
        });
        if let Err(error) = traversal {
            result = Err(error);
        }
    });
    result
}

fn collect_on_drop<T: HeapObject>(value: &HeapRef<T>) {
    if COLLECTING.with(std::cell::Cell::get) || Rc::strong_count(&value.object) != 1 {
        return;
    }
    match value.object.has_heap_children() {
        Ok(false) => {
            value.registry.borrow_mut().retire(value.id);
            return;
        }
        Ok(true) => {}
        Err(_) => {
            std::mem::forget(value.object.clone());
            return;
        }
    }
    COLLECTING.with(|collecting| collecting.set(true));
    let _flag = CollectionFlag;
    let mut work = Vec::new();
    if pin_children(&*value.object, &mut work).is_err() {
        // Preserve the graph if cleanup storage is unavailable; ordinary Drop
        // could overflow the host stack. Never release partially pinned chains.
        std::mem::forget(work);
        std::mem::forget(value.object.clone());
        return;
    }
    value.object.clear_values();
    value.registry.borrow_mut().retire(value.id);
    while let Some((registry, id, object)) = work.pop() {
        if Rc::strong_count(&object) != 1 {
            continue;
        }
        if pin_children(&*object, &mut work).is_err() {
            std::mem::forget(work);
            std::mem::forget(object);
            return;
        }
        object.clear_values();
        registry.borrow_mut().retire(id);
    }
}

pub(crate) fn collection_due() -> bool {
    COLLECTION_DUE.with(std::cell::Cell::get)
}

pub(crate) fn live_objects() -> usize {
    REGISTRY.with(|registry| registry.borrow().active_slots.len())
}

/// Return cold task rows that have no path from a live handle or an externally
/// held heap object. Task and heap references share one graph so cycles crossing
/// the scheduler table and Cog aggregates are collected together.
pub(crate) fn unreachable_task_roots(
    roots: &[TaskRootSet<'_>],
) -> Result<Vec<usize>, RuntimeError> {
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(roots.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    let mut task_indices = HashMap::new();
    task_indices
        .try_reserve(roots.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    for root in roots {
        if task_indices.contains_key(&root.id) {
            return Err(RuntimeError::InvalidArgument {
                op: "async task roots",
            });
        }
        let index = nodes.len();
        nodes.push(TaskGraphNode {
            task_id: Some(root.id),
            object: None,
            strong_count: root.strong_count,
            incoming: 0,
            edges: Vec::new(),
        });
        task_indices.insert(root.id, (index, root.lease));
    }

    let mut object_indices = HashMap::new();
    object_indices
        .try_reserve(roots.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    for root in roots {
        let from = task_indices[&root.id].0;
        for value in &root.values {
            add_task_value_edges(from, value, &task_indices, &mut nodes)?;
            add_heap_value_edges(from, value, &mut nodes, &mut object_indices)?;
        }
    }

    let mut index = roots.len();
    while index < nodes.len() {
        let Some(key) = nodes[index].object.clone() else {
            index += 1;
            continue;
        };
        if let Some(object) = upgrade_key(&key) {
            let mut failure = None;
            object.visit_values(&mut |value| {
                if failure.is_some() {
                    return;
                }
                if let Err(error) = add_task_value_edges(index, value, &task_indices, &mut nodes) {
                    failure = Some(error);
                    return;
                }
                if let Err(error) =
                    add_heap_value_edges(index, value, &mut nodes, &mut object_indices)
                {
                    failure = Some(error);
                }
            });
            if let Some(error) = failure {
                return Err(error);
            }
        }
        index += 1;
    }

    for node in &mut nodes {
        if let Some(key) = &node.object {
            node.strong_count = key_strong_count(key);
        }
    }
    let mut marked = Vec::new();
    marked
        .try_reserve_exact(nodes.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    marked.resize(nodes.len(), false);
    let mut work = Vec::new();
    work.try_reserve_exact(nodes.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    for (index, node) in nodes.iter().enumerate() {
        if node.strong_count > node.incoming {
            marked[index] = true;
            work.push(index);
        }
    }
    while let Some(index) = work.pop() {
        for &child in &nodes[index].edges {
            if !marked[child] {
                marked[child] = true;
                work.push(child);
            }
        }
    }

    let mut unreachable = Vec::new();
    unreachable
        .try_reserve(roots.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    for (index, node) in nodes.iter().enumerate() {
        if !marked[index]
            && let Some(id) = node.task_id
        {
            unreachable.push(id);
        }
    }
    Ok(unreachable)
}

fn add_task_value_edges(
    from: usize,
    value: &Value,
    task_indices: &HashMap<usize, (usize, &WeakTaskHandle)>,
    nodes: &mut [TaskGraphNode],
) -> Result<(), RuntimeError> {
    let mut values = Vec::new();
    values
        .try_reserve(1)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    values.push(value);
    let mut errors = Vec::new();
    while !values.is_empty() || !errors.is_empty() {
        while let Some(value) = values.pop() {
            match value {
                Value::Task(task) => {
                    if let Some(&(target, lease)) = task_indices.get(&task.id())
                        && lease.matches(task)
                    {
                        add_task_graph_edge(from, target, nodes)?;
                    }
                }
                Value::Error(error) => {
                    errors
                        .try_reserve(1)
                        .map_err(|_| RuntimeError::ResourceLimit {
                            resource: "async task roots",
                        })?;
                    errors.push(error.as_ref());
                }
                _ => {}
            }
        }
        let Some(error) = errors.pop() else {
            continue;
        };
        match error {
            RuntimeError::WithSuppressed {
                primary,
                suppressed,
            } => {
                errors
                    .try_reserve(suppressed.len().saturating_add(1))
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                errors.push(primary);
                errors.extend(suppressed.iter());
            }
            RuntimeError::Crashed { value, .. } => {
                values
                    .try_reserve(1)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.push(value);
            }
            _ => {}
        }
    }
    Ok(())
}

fn add_heap_value_edges(
    from: usize,
    value: &Value,
    nodes: &mut Vec<TaskGraphNode>,
    object_indices: &mut HashMap<(usize, ObjectId), usize>,
) -> Result<(), RuntimeError> {
    let mut failure = None;
    let traversal = value.visit_heap_objects(&mut |key| {
        if failure.is_some() {
            return;
        }
        let identity = (Rc::as_ptr(&key.registry) as usize, key.id);
        let target = if let Some(index) = object_indices.get(&identity).copied() {
            index
        } else {
            if nodes.try_reserve(1).is_err() || object_indices.try_reserve(1).is_err() {
                failure = Some(RuntimeError::ResourceLimit {
                    resource: "async task roots",
                });
                return;
            }
            let index = nodes.len();
            nodes.push(TaskGraphNode {
                task_id: None,
                object: Some(key),
                strong_count: 0,
                incoming: 0,
                edges: Vec::new(),
            });
            object_indices.insert(identity, index);
            index
        };
        if let Err(error) = add_task_graph_edge(from, target, nodes) {
            failure = Some(error);
        }
    });
    traversal?;
    failure.map_or(Ok(()), Err)
}

fn add_task_graph_edge(
    from: usize,
    to: usize,
    nodes: &mut [TaskGraphNode],
) -> Result<(), RuntimeError> {
    nodes[from]
        .edges
        .try_reserve(1)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    nodes[to].incoming = nodes[to]
        .incoming
        .checked_add(1)
        .ok_or(RuntimeError::ResourceLimit {
            resource: "async task roots",
        })?;
    nodes[from].edges.push(to);
    Ok(())
}

fn upgrade_key(key: &ObjectKey) -> Option<Rc<dyn HeapObject>> {
    let state = key.registry.borrow();
    state
        .slots
        .get(key.id.slot)
        .filter(|slot| slot.generation == key.id.generation)
        .and_then(|slot| slot.object.as_ref())
        .and_then(Weak::upgrade)
}

fn key_strong_count(key: &ObjectKey) -> usize {
    let state = key.registry.borrow();
    state
        .slots
        .get(key.id.slot)
        .filter(|slot| slot.generation == key.id.generation)
        .and_then(|slot| slot.object.as_ref())
        .map_or(0, Weak::strong_count)
}

/// Collect unreachable managed cycles on the current thread's Cog heap.
/// Strongly held `Value` and `HeapRef` aliases remain roots.
pub fn collect_cycles() -> Result<usize, RuntimeError> {
    collect(&[])
}

/// Trace all objects, clear unreachable graph edges, then retire their slots.
/// `discarded` contains execution roots that are about to leave scope; their
/// references do not keep objects alive during this final collection.
pub(crate) fn collect(discarded: &[Value]) -> Result<usize, RuntimeError> {
    collect_with_excluded(discarded, &[])
}

fn collect_with_excluded(
    discarded_values: &[Value],
    discarded_ids: &[ObjectId],
) -> Result<usize, RuntimeError> {
    if COLLECTING.with(|collecting| collecting.replace(true)) {
        return Ok(0);
    }
    let _collection_flag = CollectionFlag;
    let registry = REGISTRY.with(Rc::clone);
    let snapshot = {
        let state = registry.borrow();
        let mut snapshot = Vec::new();
        snapshot
            .try_reserve_exact(state.active_slots.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "VM heap",
            })?;
        for &slot in &state.active_slots {
            let entry = &state.slots[slot];
            #[cfg(test)]
            SNAPSHOT_SLOT_VISITS.with(|visits| visits.set(visits.get() + 1));
            if let Some(weak) = &entry.object {
                snapshot.push((
                    ObjectId {
                        slot,
                        generation: entry.generation,
                    },
                    weak.clone(),
                ));
            }
        }
        snapshot
    };
    // Active positions stay stable until tracing finishes. Generation checks
    // still reject stale handles; retired slot capacity is not scanned or
    // copied into collection scratch storage.
    let snapshot_index = |id: ObjectId| {
        let state = registry.borrow();
        let index = state.slots.get(id.slot)?.active_index;
        snapshot
            .get(index)
            .filter(|(current, _)| *current == id)
            .map(|_| index)
    };

    let mut incoming = Vec::new();
    incoming
        .try_reserve_exact(snapshot.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    incoming.resize(snapshot.len(), 0usize);
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(snapshot.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    edges.resize_with(snapshot.len(), Vec::new);
    let mut edge_allocation_failed = false;
    for (index, (_, weak)) in snapshot.iter().enumerate() {
        let Some(object) = weak.upgrade() else {
            continue;
        };

        object.visit_values(&mut |value| {
            if edge_allocation_failed {
                return;
            }
            let result = value.visit_heap_objects(&mut |child| {
                if Rc::ptr_eq(&registry, &child.registry)
                    && let Some(child_index) = snapshot_index(child.id)
                {
                    if edges[index].try_reserve(1).is_err() {
                        edge_allocation_failed = true;
                        return;
                    }
                    incoming[child_index] = incoming[child_index].saturating_add(1);
                    edges[index].push(child_index);
                }
            });
            if result.is_err() {
                edge_allocation_failed = true;
            }
        });
    }
    if edge_allocation_failed {
        return Err(RuntimeError::ResourceLimit {
            resource: "VM heap",
        });
    }

    let mut discarded_counts = Vec::new();
    discarded_counts
        .try_reserve_exact(snapshot.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    discarded_counts.resize(snapshot.len(), 0usize);
    for value in discarded_values {
        value.visit_heap_objects(&mut |object| {
            if Rc::ptr_eq(&registry, &object.registry)
                && let Some(index) = snapshot_index(object.id)
            {
                discarded_counts[index] = discarded_counts[index].saturating_add(1);
            }
        })?;
    }
    for id in discarded_ids {
        if let Some(index) = snapshot_index(*id) {
            discarded_counts[index] = discarded_counts[index].saturating_add(1);
        }
    }

    let mut marked = Vec::new();
    marked
        .try_reserve_exact(snapshot.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    marked.resize(snapshot.len(), false);
    let mut work = Vec::new();
    work.try_reserve_exact(snapshot.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    for (index, (_, weak)) in snapshot.iter().enumerate() {
        let strong = weak.strong_count();
        let accounted = incoming[index].saturating_add(discarded_counts[index]);
        if strong > accounted {
            marked[index] = true;
            work.push(index);
        }
    }
    while let Some(index) = work.pop() {
        for &child in &edges[index] {
            if !marked[child] {
                marked[child] = true;
                work.push(child);
            }
        }
    }

    let unreachable = marked
        .iter()
        .zip(&snapshot)
        .filter(|(marked, (_, weak))| !**marked && weak.strong_count() != 0)
        .count();
    registry
        .borrow_mut()
        .free
        .try_reserve(snapshot.len())
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    let mut objects = Vec::new();
    objects
        .try_reserve_exact(unreachable)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM heap",
        })?;
    for (index, (id, weak)) in snapshot.iter().enumerate() {
        if !marked[index]
            && let Some(object) = weak.upgrade()
        {
            objects.push((*id, object));
        }
    }

    // Keep every unreachable object alive while all edges are cleared. This
    // makes destruction iterative even for very deep chains.
    for (_, object) in &objects {
        object.clear_values();
    }
    drop(objects);

    let mut state = registry.borrow_mut();
    for (id, weak) in snapshot {
        if weak.strong_count() != 0 {
            continue;
        }
        let Some(slot) = state.slots.get_mut(id.slot) else {
            continue;
        };
        if slot.generation != id.generation
            || slot
                .object
                .as_ref()
                .is_none_or(|current| current.strong_count() != 0)
        {
            continue;
        }
        state.retire(id);
    }
    state.allocations_since_collection = 0;
    state.collection_threshold = state.active_slots.len().max(256);
    COLLECTION_DUE.with(|due| due.set(false));
    Ok(unreachable)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::value::{ArrayValue, ClosureCaptures, RecordValue};

    #[test]
    fn foreign_task_with_reused_id_does_not_hide_a_live_task_root() {
        use crate::value::TaskHandle;
        use std::cell::Cell;
        use std::sync::mpsc;

        let (local_sender, _) = mpsc::channel();
        let (foreign_sender, _) = mpsc::channel();
        let local = TaskHandle::new(2, local_sender, Rc::new(Cell::new(false)));
        let foreign = TaskHandle::new(2, foreign_sender, Rc::new(Cell::new(false)));
        let foreign_value = Value::Task(foreign);
        let lease = local.downgrade();
        let roots = [TaskRootSet {
            id: 2,
            lease: &lease,
            strong_count: lease.strong_count(),
            values: vec![&foreign_value],
        }];
        assert!(unreachable_task_roots(&roots).unwrap().is_empty());

        let local_value = Value::Task(local.clone());
        let roots = [TaskRootSet {
            id: 2,
            lease: &lease,
            strong_count: lease.strong_count(),
            values: vec![&local_value],
        }];
        assert!(unreachable_task_roots(&roots).unwrap().is_empty());
        drop(local);
        let roots = [TaskRootSet {
            id: 2,
            lease: &lease,
            strong_count: lease.strong_count(),
            values: vec![&local_value],
        }];
        assert_eq!(unreachable_task_roots(&roots).unwrap(), vec![2]);
    }

    #[test]
    fn collection_work_ignores_retired_slots_after_a_heap_peak() {
        collect_cycles().unwrap();
        let mut roots = (0..4096)
            .map(|index| {
                HeapRef::new(RecordValue::new(HashMap::from([(
                    "index".into(),
                    Value::Int(index),
                )])))
            })
            .collect::<Vec<_>>();
        let last = roots.pop().unwrap();
        let first = roots.swap_remove(0);
        drop(roots);
        assert_eq!(live_objects(), 2);

        SNAPSHOT_SLOT_VISITS.with(|visits| visits.set(0));
        assert_eq!(collect_cycles().unwrap(), 0);
        assert_eq!(SNAPSHOT_SLOT_VISITS.with(|visits| visits.get()), 2);
        assert!(matches!(
            first.try_get_field("index").unwrap(),
            Some(Value::Int(0))
        ));
        assert!(matches!(
            last.try_get_field("index").unwrap(),
            Some(Value::Int(4095))
        ));

        first
            .try_set_field("peer", Value::Record(last.clone()))
            .unwrap();
        last.try_set_field("peer", Value::Record(first.clone()))
            .unwrap();
        drop(first);
        drop(last);
        SNAPSHOT_SLOT_VISITS.with(|visits| visits.set(0));
        assert_eq!(collect_cycles().unwrap(), 2);
        assert_eq!(SNAPSHOT_SLOT_VISITS.with(|visits| visits.get()), 2);
        assert_eq!(live_objects(), 0);
    }

    #[test]
    fn collection_intervals_grow_with_the_surviving_heap() {
        collect_cycles().unwrap();
        let mut roots = Vec::new();
        let mut collections = 0;
        for _ in 0..16_384 {
            roots.push(HeapRef::new(RefCell::new(Vec::<u8>::new())));
            if collection_due() {
                collect_cycles().unwrap();
                collections += 1;
            }
        }
        assert!(
            collections <= 8,
            "{collections} complete traces for a growing live heap"
        );
        assert_eq!(live_objects(), roots.len());
        drop(roots);
        assert_eq!(live_objects(), 0);
    }

    #[test]
    fn temporary_container_cleanup_does_not_trace_unrelated_or_shared_graphs() {
        struct Counted(Rc<std::cell::Cell<usize>>);
        impl HeapObject for Counted {
            fn visit_values(&self, _: &mut dyn FnMut(&Value)) {
                self.0.set(self.0.get() + 1);
            }
            fn clear_values(&self) {}
        }
        let visits = Rc::new(std::cell::Cell::new(0));
        let roots = (0..2048)
            .map(|_| HeapRef::new(Counted(visits.clone())))
            .collect::<Vec<_>>();
        let shared = HeapRef::new(ArrayValue::new(vec![Value::Int(42)]).unwrap());
        for _ in 0..256 {
            let child = HeapRef::new(ArrayValue::new(vec![Value::Array(shared.clone())]).unwrap());
            drop(HeapRef::new(RecordValue::new(HashMap::from([(
                "child".into(),
                Value::Array(child),
            )]))));
            assert_eq!(live_objects(), roots.len() + 1);
        }
        assert_eq!(
            visits.get(),
            0,
            "temporary drops traced the entire registry"
        );
        assert_eq!(shared.borrow().len(), 1);
        drop(roots);
        drop(shared);
        assert_eq!(live_objects(), 0);
    }

    #[test]
    fn forgotten_cleanup_roots_remain_in_the_live_object_budget() {
        let before = live_objects();
        let value = HeapRef::new(RecordValue::new(HashMap::new()));
        std::mem::forget(value);
        assert_eq!(live_objects(), before + 1);
    }

    #[test]
    fn object_quota_rejects_before_registering_an_over_limit_object() {
        let before = live_objects();
        let _limit = limit_object_allocations(Some(before + 1));
        let first = HeapRef::try_new(RecordValue::new(HashMap::new())).unwrap();
        assert_eq!(live_objects(), before + 1);

        let error = HeapRef::try_new(RecordValue::new(HashMap::new())).unwrap_err();
        assert!(matches!(
            error,
            RuntimeError::ResourceLimit {
                resource: "VM heap objects"
            }
        ));
        assert_eq!(live_objects(), before + 1);

        drop(first);
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn nested_object_limits_cannot_relax_an_outer_limit() {
        let before = live_objects();
        let _outer = limit_object_allocations(Some(before + 1));
        let value = HeapRef::try_new(RecordValue::new(HashMap::new())).unwrap();
        {
            let _inner = limit_object_allocations(None);
            assert!(matches!(
                HeapRef::try_new(RecordValue::new(HashMap::new())),
                Err(RuntimeError::ResourceLimit {
                    resource: "VM heap objects"
                })
            ));
        }
        drop(value);
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn object_quota_collects_unreachable_cycles_before_reusing_the_slot() {
        let before = live_objects();
        let _limit = limit_object_allocations(Some(before + 1));
        let cell = HeapRef::try_new(RefCell::new(Value::Nil)).unwrap();
        *cell.borrow_mut() = Value::Cell(cell.clone());
        drop(cell);
        assert_eq!(live_objects(), before + 1);

        let next = HeapRef::try_new(RecordValue::new(HashMap::new())).unwrap();
        assert_eq!(live_objects(), before + 1);
        drop(next);
        assert_eq!(live_objects(), before);
    }

    #[test]
    fn collection_breaks_cycles_and_reuses_slots_with_a_new_generation() {
        let value = HeapRef::new(RecordValue::new(HashMap::new()));
        value
            .try_set_field("self", Value::Record(value.clone()))
            .unwrap();
        let before = value.id();
        assert_eq!(collect(&[]).unwrap(), 0);
        drop(value);

        assert_eq!(collect(&[]).unwrap(), 1);
        let next = HeapRef::new(ArrayValue::new(Vec::new()).unwrap());
        assert_eq!(next.id().slot, before.slot);
        assert_ne!(next.id().generation, before.generation);
    }

    #[test]
    fn live_external_alias_is_a_root() {
        let array = HeapRef::new(ArrayValue::new(vec![Value::Int(1)]).unwrap());
        let alias = array.clone();
        assert_eq!(collect(&[]).unwrap(), 0);
        assert_eq!(alias.borrow().len(), 1);
    }

    #[test]
    fn collection_traces_cycles_through_closure_captures() {
        let cell = HeapRef::new(RefCell::new(Value::Nil));
        let captures = HeapRef::new(ClosureCaptures::new(vec![Value::Cell(cell.clone())]));
        *cell.borrow_mut() = Value::Closure(7, captures.clone());
        drop(cell);
        drop(captures);
        assert_eq!(collect(&[]).unwrap(), 2);
    }

    #[test]
    fn collection_traces_reference_values_through_closure_captures() {
        let record = HeapRef::new(RecordValue::new(HashMap::new()));
        let captures = HeapRef::new(ClosureCaptures::new(vec![Value::Record(record.clone())]));
        record
            .try_set_field("callback", Value::Closure(7, captures.clone()))
            .unwrap();
        drop(record);
        drop(captures);
        assert_eq!(collect(&[]).unwrap(), 2);
    }

    #[test]
    fn external_root_keeps_entire_ring_alive_across_graph_sizes() {
        for count in 1..=128 {
            let cells = (0..count)
                .map(|_| HeapRef::new(RefCell::new(Value::Nil)))
                .collect::<Vec<_>>();
            for index in 0..count {
                *cells[index].borrow_mut() = Value::Cell(cells[(index + 1) % count].clone());
            }

            let root = cells[count / 2].clone();
            drop(cells);
            assert_eq!(collect(&[]).unwrap(), 0, "ring size {count}");
            assert!(matches!(*root.borrow(), Value::Cell(_)));

            drop(root);
            assert_eq!(collect(&[]).unwrap(), count, "ring size {count}");
            assert_eq!(live_objects(), 0, "ring size {count}");
        }
    }

    #[test]
    fn duplicated_edges_and_an_external_alias_are_counted_as_roots() {
        let record = HeapRef::new(RecordValue::new(HashMap::new()));
        let cell = HeapRef::new(RefCell::new(Value::Nil));
        record
            .try_set_field("first", Value::Cell(cell.clone()))
            .unwrap();
        record
            .try_set_field("second", Value::Cell(cell.clone()))
            .unwrap();
        *cell.borrow_mut() = Value::Record(record.clone());

        let root = cell.clone();
        drop(record);
        drop(cell);
        assert_eq!(collect(&[]).unwrap(), 0);
        assert!(matches!(*root.borrow(), Value::Record(_)));

        drop(root);
        assert_eq!(collect(&[]).unwrap(), 2);
        assert_eq!(live_objects(), 0);
    }

    #[test]
    fn dropping_a_deep_record_chain_is_iterative() {
        let mut value = Value::Int(0);
        for _ in 0..100_000 {
            value = Value::Record(HeapRef::new(RecordValue::new(HashMap::from([(
                "next".into(),
                value,
            )]))));
        }
        drop(value);
        assert_eq!(live_objects(), 0);
    }
}
