use crate::error::RuntimeError;
use crate::heap;
use crate::heap::{HeapRef, TaskRootSet};
use crate::instruction::Instruction;
use crate::jit::{JitExit, JitPlanState, JitState};
use crate::template::ArrayTemplate;
use crate::value::{
    ArrayValue, BoundFunction, ClosureCaptures, RecordValue, RuntimeType, TaskHandle, Value,
    ValueFields, WeakTaskHandle,
};
use num_bigint::BigInt;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::{CallableSignature, CompiledProgram};
use crate::runtime::RuntimeContext;
use crate::runtime::VmCapabilities;
use crate::runtime::async_io::{
    AsyncFileOperation, AsyncFsJob, AsyncFsOperation, AsyncIoCompletion, AsyncIoFailure,
    AsyncIoJob, AsyncIoPool, AsyncIoResult, AsyncStdioInputJob, AsyncStdioOutputJob,
    AsyncStdioPool, AsyncTcpOperation,
};
use crate::runtime::{
    AsyncIoReadStream, AsyncIoWriteStream, AsyncWakeHandle, FileOpenMode, NativeAsyncOperation,
    NativeImport, checked_size, file_write_buffer,
};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

const MAX_ASYNC_TASKS: usize = 1_000_000;
const MAX_STACK_EFFECT_CACHE_INSTRUCTIONS: usize = 4096;

#[derive(Clone, Default)]
pub(crate) struct CaptureValues(Option<HeapRef<ClosureCaptures>>);

impl CaptureValues {
    pub(crate) fn get(&self, index: usize) -> Option<Value> {
        self.0.as_ref()?.get(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |captures| captures.len())
    }

    fn into_root(self) -> Option<Value> {
        self.0.map(|captures| Value::Closure(0, captures))
    }
}

type CallFrame = (usize, usize, CaptureValues, usize, usize, usize);

struct VmContinuation {
    pc: usize,
    stack: Vec<Value>,
    locals: Vec<Option<Value>>,
    bp: usize,
    call_stack: Vec<CallFrame>,
    operand_base: usize,
    captures: CaptureValues,
    error_value_pc: Option<usize>,
    halt_error_pc: Option<usize>,
    propagation_pending: bool,
    callable_start: usize,
    callable_end: usize,
}

enum VmExecution {
    Complete(Result<Value, RuntimeError>),
    Suspended(VmContinuation),
}

/// Immutable metadata derived from a program. SDK programs retain this plan
/// across invocations; an interpreter run shares it with every async task.
pub(crate) struct VmExecutionPlan {
    stack_effect_cache: Option<Vec<(usize, usize)>>,
    callables: Option<Vec<CallablePlanEntry>>,
    needs_stdin: bool,
    needs_stdout: bool,
    has_async: bool,
    jit: RefCell<JitPlanState>,
}

struct CallablePlanEntry {
    entry: usize,
    signature: CallableSignature,
}

impl VmExecutionPlan {
    fn new(artifact: &CompiledProgram) -> Self {
        let program = &artifact.instructions;
        let mut has_back_edge = false;
        let mut needs_stdin = false;
        let mut needs_stdout = false;
        let mut has_async = false;
        for (pc, instruction) in program.iter().enumerate() {
            has_back_edge |= matches!(
                instruction,
                Instruction::Jump(target)
                    | Instruction::JumpIfFalse(target)
                    | Instruction::JumpIfTrue(target)
                    | Instruction::JumpIfNoPropagation(target)
                    if *target <= pc
            );
            needs_stdin |= matches!(
                instruction,
                Instruction::AsyncIoRead
                    | Instruction::AsyncIoReadExact
                    | Instruction::AsyncIoReadAll
            );
            needs_stdout |= matches!(
                instruction,
                Instruction::AsyncIoWrite
                    | Instruction::AsyncIoWriteAll
                    | Instruction::AsyncIoFlush
            );
            has_async |= instruction_requires_scheduler(instruction, &artifact.imports);
        }

        let stack_effect_cache =
            prepare_stack_effect_cache(program, &artifact.imports, has_back_edge);
        let cached_parameter_count = artifact
            .callables
            .values()
            .map(|signature| signature.parameter_types.as_ref().map_or(0, Vec::len))
            .fold(0usize, |sum, count| sum.saturating_add(count));
        let callables =
            if (8..=256).contains(&artifact.callables.len()) && cached_parameter_count <= 4096 {
                let mut entries = Vec::new();
                if entries.try_reserve_exact(artifact.callables.len()).is_ok() {
                    entries.extend(artifact.callables.iter().map(|(&entry, signature)| {
                        CallablePlanEntry {
                            entry,
                            signature: signature.clone(),
                        }
                    }));
                    Some(entries)
                } else {
                    None
                }
            } else {
                None
            };

        Self {
            stack_effect_cache,
            callables,
            needs_stdin,
            needs_stdout,
            has_async,
            jit: RefCell::new(JitPlanState::NotInitialized),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_jit_osr(
        &self,
        artifact: &CompiledProgram,
        entry: usize,
        target: usize,
        pc: &mut usize,
        locals: &mut Vec<Option<Value>>,
        bp: usize,
        stack: &mut Vec<Value>,
        operand_base: usize,
        captures: &CaptureValues,
        propagation_pending: &mut bool,
        context: &RuntimeContext<'_>,
    ) -> Option<JitExit> {
        if !context.jit_enabled || context.limits.any_active() {
            return None;
        }

        let mut state = self.jit.borrow_mut();
        if matches!(*state, JitPlanState::NotInitialized) {
            *state = match JitState::new(artifact, context.jit_profile_path.as_deref()) {
                Ok(jit) => JitPlanState::Ready(Box::new(jit)),
                Err(error) => {
                    if context.jit_debug {
                        eprintln!("[jit] unavailable: {error}");
                    }
                    JitPlanState::Unavailable
                }
            };
        }
        let JitPlanState::Ready(jit) = &mut *state else {
            return None;
        };
        jit.try_osr(
            artifact,
            entry,
            target,
            pc,
            locals,
            bp,
            stack,
            operand_base,
            captures,
            propagation_pending,
            context.jit_debug,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_jit_call(
        &self,
        artifact: &CompiledProgram,
        entry: usize,
        pc: &mut usize,
        locals: &mut Vec<Option<Value>>,
        bp: usize,
        stack: &mut Vec<Value>,
        operand_base: usize,
        captures: &CaptureValues,
        propagation_pending: &mut bool,
        context: &RuntimeContext<'_>,
    ) -> Option<JitExit> {
        if !context.jit_enabled || context.limits.any_active() {
            return None;
        }

        let mut state = self.jit.borrow_mut();
        if matches!(*state, JitPlanState::NotInitialized) {
            *state = match JitState::new(artifact, context.jit_profile_path.as_deref()) {
                Ok(jit) => JitPlanState::Ready(Box::new(jit)),
                Err(error) => {
                    if context.jit_debug {
                        eprintln!("[jit] unavailable: {error}");
                    }
                    JitPlanState::Unavailable
                }
            };
        }
        let JitPlanState::Ready(jit) = &mut *state else {
            return None;
        };
        jit.try_call(
            artifact,
            entry,
            pc,
            locals,
            bp,
            stack,
            operand_base,
            captures,
            propagation_pending,
            context.jit_debug,
        )
    }

    fn next_callable_boundary(&self, artifact: &CompiledProgram, entry: usize) -> usize {
        self.callables.as_ref().map_or_else(
            || next_callable_boundary(artifact, entry).unwrap_or(artifact.instructions.len()),
            |entries| {
                let index = entries.partition_point(|candidate| candidate.entry <= entry);
                entries
                    .get(index)
                    .map_or(artifact.instructions.len(), |entry| entry.entry)
            },
        )
    }

    fn callable_region_start(&self, artifact: &CompiledProgram, pc: usize) -> usize {
        self.callables.as_ref().map_or_else(
            || callable_region_start(artifact, pc),
            |entries| {
                entries
                    .partition_point(|entry| entry.entry <= pc)
                    .checked_sub(1)
                    .map_or(0, |index| entries[index].entry)
            },
        )
    }

    fn callable_signature<'a>(
        &'a self,
        artifact: &'a CompiledProgram,
        entry: usize,
    ) -> Option<&'a CallableSignature> {
        self.callables.as_ref().map_or_else(
            || artifact.callables.get(&entry),
            |callables| {
                callables
                    .binary_search_by_key(&entry, |callable| callable.entry)
                    .ok()
                    .map(|index| &callables[index].signature)
            },
        )
    }
}

pub(crate) fn prepare_execution_plan(artifact: &CompiledProgram) -> Rc<VmExecutionPlan> {
    Rc::new(VmExecutionPlan::new(artifact))
}

fn instruction_requires_scheduler(instruction: &Instruction, imports: &[NativeImport]) -> bool {
    match instruction {
        Instruction::CallNative(index) => imports.get(*index).is_some_and(NativeImport::is_async),
        Instruction::CallAsync(_, _)
        | Instruction::Await
        | Instruction::Spawn
        | Instruction::Cancel
        | Instruction::CheckCancel
        | Instruction::YieldNow
        | Instruction::TimeoutAsync
        | Instruction::Select
        | Instruction::Select2
        | Instruction::SleepAsync
        | Instruction::ReadBytesAsync
        | Instruction::ProcessRunAsync
        | Instruction::ProcessRunTimeoutAsync
        | Instruction::ProcessRunInputAsync
        | Instruction::ProcessRunInputTimeoutAsync
        | Instruction::ProcessRunInputTimeoutOutputLimitAsync
        | Instruction::NetworkExchangeAsync
        | Instruction::TcpListenAsync
        | Instruction::TcpAcceptAsync
        | Instruction::TcpConnectAsync
        | Instruction::TcpReadAsync
        | Instruction::TcpReadExactAsync
        | Instruction::TcpReadAllAsync
        | Instruction::TcpWriteAsync
        | Instruction::TcpWriteAllAsync
        | Instruction::TcpShutdownWriteAsync
        | Instruction::FsMetadataAsync
        | Instruction::FsReadDirAsync
        | Instruction::FsWriteBytesAsync
        | Instruction::FsCreateDirAsync
        | Instruction::FsCreateDirAllAsync
        | Instruction::FsRemoveFileAsync
        | Instruction::FsRemoveDirAsync
        | Instruction::FsReadTextAsync
        | Instruction::FsWriteTextAsync
        | Instruction::FsReadLinesAsync
        | Instruction::FsAppendBytesAsync
        | Instruction::FsAppendTextAsync
        | Instruction::FsCopyAsync
        | Instruction::FsRenameAsync
        | Instruction::AsyncIoOpen
        | Instruction::AsyncIoRead
        | Instruction::AsyncIoReadExact
        | Instruction::AsyncIoReadAll
        | Instruction::AsyncIoWrite
        | Instruction::AsyncIoWriteAll
        | Instruction::AsyncIoFlush
        | Instruction::AsyncIoSeek => true,
        _ => false,
    }
}

fn await_outcome(outcome: Result<Value, RuntimeError>) -> Result<Value, RuntimeError> {
    match outcome {
        Ok(value) => Ok(value),
        Err(error) if error.is_recoverable() => Ok(Value::Error(Box::new(error))),
        Err(error) => Err(error),
    }
}

enum AsyncTaskState {
    Cold,
    Queued,
    Running,
    WaitingTask(usize),
    WaitingTimeout {
        target: usize,
        deadline: Instant,
    },
    WaitingSelect {
        targets: Rc<Vec<(usize, usize)>>,
    },
    Joining,
    WaitingUntil(Instant),
    WaitingIo,
    WaitingPlugin,
    Complete {
        outcome: Result<Value, RuntimeError>,
        completed_at: u64,
    },
}

enum AsyncTaskTarget {
    Function {
        entry: usize,
        arguments: Vec<Value>,
    },
    Sleep(Value),
    YieldNow,
    Timeout {
        task: Value,
        seconds: Value,
    },
    Select2 {
        first: Value,
        second: Value,
    },
    Select(Value),
    Ready(Result<Value, RuntimeError>),
    Filesystem {
        operation: AsyncFsOperation,
        arguments: Vec<Value>,
    },
    FileHandle {
        operation: AsyncFileOperation,
        arguments: Vec<Value>,
    },
    ProcessRun {
        program: Value,
        arguments: Value,
        input: Option<Value>,
        timeout: Option<Value>,
        output_limit: Option<Value>,
    },
    NetworkExchange {
        arguments: Vec<Value>,
    },
    Tcp {
        operation: AsyncTcpOperation,
        arguments: Vec<Value>,
    },
    Native {
        import: NativeImport,
        arguments: Vec<Value>,
    },
    NativePlugin {
        operation: Box<dyn NativeAsyncOperation>,
    },
    Consumed,
}

impl AsyncTaskTarget {
    fn append_captured_values<'a>(
        &'a self,
        values: &mut Vec<&'a Value>,
    ) -> Result<(), RuntimeError> {
        match self {
            Self::Function { arguments, .. }
            | Self::Filesystem { arguments, .. }
            | Self::FileHandle { arguments, .. }
            | Self::NetworkExchange { arguments }
            | Self::Tcp { arguments, .. }
            | Self::Native { arguments, .. } => {
                values
                    .try_reserve(arguments.len())
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.extend(arguments);
            }
            Self::Sleep(seconds) => {
                values
                    .try_reserve(1)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.push(seconds);
            }
            Self::Timeout { task, seconds } => {
                values
                    .try_reserve(2)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.extend([task, seconds]);
            }
            Self::Select2 { first, second } => {
                values
                    .try_reserve(2)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.extend([first, second]);
            }
            Self::Select(tasks) => {
                values
                    .try_reserve(1)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.push(tasks);
            }
            Self::ProcessRun {
                program,
                arguments,
                input,
                timeout,
                output_limit,
            } => {
                values
                    .try_reserve(
                        2 + usize::from(input.is_some())
                            + usize::from(timeout.is_some())
                            + usize::from(output_limit.is_some()),
                    )
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async task roots",
                    })?;
                values.extend([program, arguments]);
                if let Some(input) = input {
                    values.push(input);
                }
                if let Some(timeout) = timeout {
                    values.push(timeout);
                }
                if let Some(output_limit) = output_limit {
                    values.push(output_limit);
                }
            }
            Self::YieldNow | Self::Ready(_) | Self::NativePlugin { .. } | Self::Consumed => {}
        }
        Ok(())
    }
}

enum WaiterWake {
    Await,
    Timeout(Instant),
    Select {
        index: i64,
        targets: Rc<Vec<(usize, usize)>>,
    },
}

enum WaiterTarget {
    One(usize),
    Many(Rc<Vec<(usize, usize)>>),
}

struct AsyncTask {
    lease: WeakTaskHandle,
    target: AsyncTaskTarget,
    state: AsyncTaskState,
    children: Vec<usize>,
    parent: Option<usize>,
    observed: bool,
    joined: bool,
    abandoned: bool,
    cancel_requested: bool,
    continuation: Option<VmContinuation>,
    resume_outcome: Option<Result<Value, RuntimeError>>,
    pending_await: Option<TaskHandle>,
    pending_select: Option<Vec<TaskHandle>>,
    join_outcome: Option<Result<Value, RuntimeError>>,
    plugin_wake: Option<AsyncWakeHandle>,
    io_cancel: Option<Arc<AtomicBool>>,
}

enum AsyncIoSubmission {
    Worker(AsyncIoJob),
    Stdin(AsyncStdioInputJob),
    Stdout(AsyncStdioOutputJob),
}

struct AsyncScheduler {
    execution_plan: Option<Rc<VmExecutionPlan>>,
    tasks: HashMap<usize, AsyncTask>,
    waiters: HashMap<usize, Vec<usize>>,
    ready: VecDeque<usize>,
    active: Vec<usize>,
    timers: BinaryHeap<Reverse<(Instant, usize)>>,
    io_pool: Option<AsyncIoPool>,
    stdio_pool: Option<AsyncStdioPool>,
    io_completion_sender: Option<Sender<AsyncIoCompletion>>,
    io_completions: Option<Receiver<AsyncIoCompletion>>,
    pending_io: usize,
    file_open_reservations: HashMap<usize, usize>,
    pending_plugin_operations: usize,
    next_id: usize,
    completion_order: u64,
    abandoned_sender: Option<Sender<usize>>,
    abandoned_receiver: Option<Receiver<usize>>,
    // TaskHandle is !Send, so this same-thread flag can guard the empty-queue fast path.
    abandoned_pending: Option<Rc<std::cell::Cell<bool>>>,
    plugin_wake_sender: Option<std::sync::mpsc::SyncSender<usize>>,
    plugin_wake_receiver: Option<Receiver<usize>>,
}

impl AsyncScheduler {
    fn task_id(&self, handle: &TaskHandle) -> Result<usize, RuntimeError> {
        let id = handle.id();
        if self
            .tasks
            .get(&id)
            .is_some_and(|task| task.lease.matches(handle))
        {
            Ok(id)
        } else {
            Err(RuntimeError::InvalidArgument { op: "async task" })
        }
    }

    fn new(
        execution_plan: Option<Rc<VmExecutionPlan>>,
        io_completion_channel: Option<(Sender<AsyncIoCompletion>, Receiver<AsyncIoCompletion>)>,
        stdio_pool: Option<AsyncStdioPool>,
    ) -> Self {
        let (io_completion_sender, io_completions) = io_completion_channel
            .map_or((None, None), |(sender, receiver)| {
                (Some(sender), Some(receiver))
            });
        Self {
            execution_plan,
            tasks: HashMap::new(),
            waiters: HashMap::new(),
            ready: VecDeque::new(),
            active: Vec::new(),
            timers: BinaryHeap::new(),
            io_pool: None,
            stdio_pool,
            io_completion_sender,
            io_completions,
            pending_io: 0,
            file_open_reservations: HashMap::new(),
            pending_plugin_operations: 0,
            next_id: 1,
            completion_order: 0,
            abandoned_sender: None,
            abandoned_receiver: None,
            abandoned_pending: None,
            plugin_wake_sender: None,
            plugin_wake_receiver: None,
        }
    }

    fn ensure_io_channel(&mut self) -> Sender<AsyncIoCompletion> {
        if self.io_completion_sender.is_none() {
            let (sender, receiver) = mpsc::channel();
            self.io_completion_sender = Some(sender);
            self.io_completions = Some(receiver);
        }
        self.io_completion_sender
            .as_ref()
            .expect("I/O channel initialized above")
            .clone()
    }

    fn ensure_abandoned_channel(&mut self) -> (Sender<usize>, Rc<std::cell::Cell<bool>>) {
        if self.abandoned_sender.is_none() {
            let (sender, receiver) = mpsc::channel();
            self.abandoned_sender = Some(sender);
            self.abandoned_receiver = Some(receiver);
            self.abandoned_pending = Some(Rc::new(std::cell::Cell::new(false)));
        }
        (
            self.abandoned_sender
                .as_ref()
                .expect("abandoned-task channel initialized above")
                .clone(),
            Rc::clone(
                self.abandoned_pending
                    .as_ref()
                    .expect("abandoned-task flag initialized above"),
            ),
        )
    }

    fn create_task(
        &mut self,
        target: usize,
        arguments: Vec<Value>,
    ) -> Result<TaskHandle, RuntimeError> {
        self.create_target(AsyncTaskTarget::Function {
            entry: target,
            arguments,
        })
    }

    fn create_target(&mut self, target: AsyncTaskTarget) -> Result<TaskHandle, RuntimeError> {
        if self.tasks.len() >= MAX_ASYNC_TASKS {
            return Err(RuntimeError::ResourceLimit {
                resource: "async tasks",
            });
        }
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or(RuntimeError::ResourceLimit {
            resource: "async tasks",
        })?;
        self.tasks
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async tasks",
            })?;
        let (abandoned_sender, abandoned_pending) = self.ensure_abandoned_channel();
        let handle = TaskHandle::new(id, abandoned_sender, abandoned_pending);
        let plugin_wake = if matches!(&target, AsyncTaskTarget::Native { .. }) {
            if self.plugin_wake_sender.is_none() {
                let (sender, receiver) = mpsc::sync_channel(4096);
                self.plugin_wake_sender = Some(sender);
                self.plugin_wake_receiver = Some(receiver);
            }
            let sender = self
                .plugin_wake_sender
                .as_ref()
                .ok_or(RuntimeError::InvalidArgument {
                    op: "async plugin wakeup channel",
                })?;
            Some(AsyncWakeHandle::new(id, sender.clone()))
        } else {
            None
        };
        self.tasks.insert(
            id,
            AsyncTask {
                lease: handle.downgrade(),
                target,
                state: AsyncTaskState::Cold,
                children: Vec::new(),
                parent: None,
                observed: false,
                joined: false,
                abandoned: false,
                cancel_requested: false,
                continuation: None,
                resume_outcome: None,
                pending_await: None,
                pending_select: None,
                join_outcome: None,
                plugin_wake,
                io_cancel: None,
            },
        );
        Ok(handle)
    }

    fn complete_task(
        &mut self,
        id: usize,
        outcome: Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let parent = self
            .tasks
            .get(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .parent;
        self.complete_task_raw(id, outcome)?;
        if let Some(parent) = parent {
            self.finish_ready_joins(parent)?;
        }
        Ok(())
    }

    fn complete_task_raw(
        &mut self,
        id: usize,
        mut outcome: Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let completed_at = self.next_completion_order()?;
        if self
            .tasks
            .get(&id)
            .is_some_and(|task| matches!(task.state, AsyncTaskState::WaitingPlugin))
        {
            self.pending_plugin_operations = self.pending_plugin_operations.checked_sub(1).ok_or(
                RuntimeError::InvalidArgument {
                    op: "async plugin operation count",
                },
            )?;
        }
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        if std::mem::take(&mut task.cancel_requested) {
            outcome = match outcome {
                Ok(Value::Error(primary)) => Ok(Value::Error(Box::new(
                    (*primary).clone().with_suppressed(RuntimeError::Cancelled),
                ))),
                Ok(_) => Ok(Value::Error(Box::new(RuntimeError::Cancelled))),
                Err(primary) => Err(primary.with_suppressed(RuntimeError::Cancelled)),
            };
        }
        task.target = AsyncTaskTarget::Consumed;
        task.plugin_wake = None;
        task.io_cancel = None;
        task.continuation = None;
        task.resume_outcome = None;
        task.pending_await = None;
        task.pending_select = None;
        task.state = AsyncTaskState::Complete {
            outcome: outcome.clone(),
            completed_at,
        };
        self.wake_waiters(id, outcome)?;
        self.release_abandoned_task(id);
        Ok(())
    }

    fn wake_waiters(
        &mut self,
        target: usize,
        outcome: Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let Some(waiters) = self.waiters.get(&target) else {
            return Ok(());
        };
        let active_waiters = waiters
            .iter()
            .filter(|waiter| {
                self.tasks
                    .get(waiter)
                    .is_some_and(|task| match &task.state {
                        AsyncTaskState::WaitingTask(awaited)
                        | AsyncTaskState::WaitingTimeout {
                            target: awaited, ..
                        } => *awaited == target,
                        AsyncTaskState::WaitingSelect { targets } => {
                            targets.iter().any(|(candidate, _)| *candidate == target)
                        }
                        _ => false,
                    })
            })
            .count();
        self.ready
            .try_reserve(active_waiters)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async ready queue",
            })?;
        let waiters = self
            .waiters
            .remove(&target)
            .expect("waiters were checked above");
        let now = Instant::now();
        for waiter in waiters {
            let waiter_state = self.tasks.get(&waiter).and_then(|task| match &task.state {
                AsyncTaskState::WaitingTask(awaited) if *awaited == target => {
                    Some(WaiterWake::Await)
                }
                AsyncTaskState::WaitingTimeout {
                    target: awaited,
                    deadline,
                } if *awaited == target => Some(WaiterWake::Timeout(*deadline)),
                AsyncTaskState::WaitingSelect { targets } => targets
                    .iter()
                    .find(|(candidate, _)| *candidate == target)
                    .map(|(_, index)| WaiterWake::Select {
                        index: *index as i64,
                        targets: Rc::clone(targets),
                    }),
                _ => None,
            });
            match waiter_state {
                Some(WaiterWake::Await) => {
                    let Some(task) = self.tasks.get_mut(&waiter) else {
                        continue;
                    };
                    task.state = AsyncTaskState::Queued;
                    task.resume_outcome = Some(outcome.clone());
                    task.pending_await = None;
                    self.ready.push_back(waiter);
                }
                Some(WaiterWake::Timeout(deadline)) => {
                    let source_won = now < deadline;
                    let outcome = if source_won {
                        if let Some(source) = self.tasks.get_mut(&target) {
                            source.observed = true;
                        }
                        outcome.clone()
                    } else {
                        Err(RuntimeError::Timeout)
                    };
                    let Some(task) = self.tasks.get_mut(&waiter) else {
                        continue;
                    };
                    task.state = AsyncTaskState::Queued;
                    task.target = AsyncTaskTarget::Ready(outcome);
                    task.pending_await = None;
                    self.ready.push_back(waiter);
                }
                Some(WaiterWake::Select { index, targets }) => {
                    for other in targets
                        .iter()
                        .map(|(other, _)| *other)
                        .filter(|other| *other != target)
                    {
                        self.remove_waiter(other, waiter);
                    }
                    let Some(task) = self.tasks.get_mut(&waiter) else {
                        continue;
                    };
                    task.state = AsyncTaskState::Queued;
                    task.target = AsyncTaskTarget::Ready(Ok(Value::Int(index)));
                    task.pending_await = None;
                    task.pending_select = None;
                    self.ready.push_back(waiter);
                }
                None => {}
            }
        }
        Ok(())
    }

    fn finish_task_body(
        &mut self,
        id: usize,
        mut outcome: Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let children = self.copy_children(id)?;
        if matches!(&outcome, Err(error) if !error.is_recoverable()) {
            // Fatal VM failures cannot be handled by this scope. Stop its spawned
            // work before joining so a long-running child does not delay the error.
            for child in &children {
                if !self.task_complete_or_released(*child) {
                    let _ = self.cancel_task(*child)?;
                }
            }
        }
        let task = self
            .tasks
            .get(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        let children_complete = task
            .children
            .iter()
            .all(|child| self.task_complete_or_released(*child));
        if !children_complete {
            let task = self
                .tasks
                .get_mut(&id)
                .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
            task.join_outcome = Some(outcome);
            task.state = AsyncTaskState::Joining;
            return Ok(());
        }
        self.join_unobserved_errors(id, &mut outcome)?;
        self.mark_children_joined(&children);
        self.complete_task(id, outcome)
    }

    fn finish_ready_joins(&mut self, mut parent: usize) -> Result<(), RuntimeError> {
        loop {
            let ready = self.tasks.get(&parent).is_some_and(|task| {
                matches!(task.state, AsyncTaskState::Joining)
                    && task
                        .children
                        .iter()
                        .all(|child| self.task_complete_or_released(*child))
            });
            if !ready {
                return Ok(());
            }
            let children = self.copy_children(parent)?;
            let (mut outcome, grandparent) = {
                let task = self
                    .tasks
                    .get_mut(&parent)
                    .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
                (
                    task.join_outcome
                        .take()
                        .ok_or(RuntimeError::InvalidArgument {
                            op: "async join state",
                        })?,
                    task.parent,
                )
            };
            self.join_unobserved_errors(parent, &mut outcome)?;
            self.mark_children_joined(&children);
            self.complete_task_raw(parent, outcome)?;
            let Some(next) = grandparent else {
                return Ok(());
            };
            parent = next;
        }
    }

    fn copy_children(&self, id: usize) -> Result<Vec<usize>, RuntimeError> {
        let children = &self
            .tasks
            .get(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .children;
        let mut copied = Vec::new();
        copied
            .try_reserve_exact(children.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async child tasks",
            })?;
        copied.extend_from_slice(children);
        Ok(copied)
    }

    fn task_complete_or_released(&self, id: usize) -> bool {
        self.tasks
            .get(&id)
            .is_none_or(|task| matches!(task.state, AsyncTaskState::Complete { .. }))
    }

    fn mark_children_joined(&mut self, children: &[usize]) {
        for child in children {
            if let Some(task) = self.tasks.get_mut(child) {
                task.joined = true;
            }
            self.release_abandoned_task(*child);
        }
    }

    fn reap_abandoned_tasks(&mut self) {
        let Some(abandoned_pending) = self.abandoned_pending.as_ref() else {
            return;
        };
        if !abandoned_pending.replace(false) {
            return;
        }
        while let Some(id) = self
            .abandoned_receiver
            .as_ref()
            .and_then(|receiver| receiver.try_recv().ok())
        {
            if let Some(task) = self.tasks.get_mut(&id) {
                task.abandoned = true;
            }
            self.release_abandoned_task(id);
        }
    }

    fn reap_unreachable_cold_tasks(&mut self) -> Result<bool, RuntimeError> {
        self.reap_abandoned_tasks();
        let mut roots = Vec::new();
        roots
            .try_reserve(self.tasks.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async task roots",
            })?;
        for (&id, task) in &self.tasks {
            if matches!(task.state, AsyncTaskState::Cold) {
                let mut values = Vec::new();
                task.target.append_captured_values(&mut values)?;
                roots.push(TaskRootSet {
                    id,
                    lease: &task.lease,
                    strong_count: task.lease.strong_count(),
                    values,
                });
            }
        }
        let unreachable = heap::unreachable_task_roots(&roots)?;
        drop(roots);

        let mut removed = false;
        for id in unreachable {
            if self
                .tasks
                .get(&id)
                .is_some_and(|task| matches!(task.state, AsyncTaskState::Cold))
            {
                drop(self.tasks.remove(&id));
                removed = true;
            }
        }
        self.reap_abandoned_tasks();
        Ok(removed)
    }

    fn release_abandoned_task(&mut self, id: usize) {
        let releasable = self.tasks.get(&id).is_some_and(|task| {
            if !task.abandoned {
                return false;
            }
            match task.state {
                AsyncTaskState::Cold => true,
                AsyncTaskState::Complete { .. } => {
                    task.parent.is_none() || task.observed || task.joined
                }
                AsyncTaskState::Queued
                | AsyncTaskState::Running
                | AsyncTaskState::WaitingTask(_)
                | AsyncTaskState::WaitingTimeout { .. }
                | AsyncTaskState::WaitingSelect { .. }
                | AsyncTaskState::Joining
                | AsyncTaskState::WaitingUntil(_)
                | AsyncTaskState::WaitingIo
                | AsyncTaskState::WaitingPlugin => false,
            }
        });
        if releasable {
            self.tasks.remove(&id);
        }
    }

    fn enqueue(&mut self, id: usize) -> Result<(), RuntimeError> {
        let is_cold = self
            .tasks
            .get(&id)
            .map(|task| matches!(task.state, AsyncTaskState::Cold))
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        if is_cold {
            self.ready
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async ready queue",
                })?;
            self.tasks
                .get_mut(&id)
                .expect("task was checked above")
                .state = AsyncTaskState::Queued;
            self.ready.push_back(id);
        }
        Ok(())
    }

    fn spawn(&mut self, id: usize) -> Result<(), RuntimeError> {
        let parent = self
            .active
            .last()
            .copied()
            .ok_or(RuntimeError::InvalidArgument {
                op: "spawn outside async task",
            })?;
        if parent == id {
            return Err(RuntimeError::InvalidArgument {
                op: "spawn task cycle",
            });
        }
        let task_parent = self
            .tasks
            .get(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .parent;
        if task_parent.is_some_and(|owner| owner != parent) {
            return Err(RuntimeError::InvalidArgument {
                op: "task already has a parent",
            });
        }
        let known_child = self
            .tasks
            .get(&parent)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .children
            .contains(&id);
        if !known_child {
            self.tasks
                .get_mut(&parent)
                .expect("active task was checked above")
                .children
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async child tasks",
                })?;
        }
        self.enqueue(id)?;
        if !known_child {
            self.tasks
                .get_mut(&parent)
                .expect("active task was checked above")
                .children
                .push(id);
            self.tasks
                .get_mut(&id)
                .expect("task was checked above")
                .parent = Some(parent);
        }
        Ok(())
    }

    fn cancel_task(&mut self, id: usize) -> Result<bool, RuntimeError> {
        let can_cancel = self
            .tasks
            .get(&id)
            .map(|task| self.task_cancellable(task))
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        if !can_cancel {
            return Ok(false);
        }
        let mut tasks_to_cancel = Vec::new();
        tasks_to_cancel
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async task cancellation",
            })?;
        tasks_to_cancel.push(id);
        let mut visited = HashSet::new();
        let mut index = 0;
        while index < tasks_to_cancel.len() {
            let current = tasks_to_cancel[index];
            index += 1;
            if visited.contains(&current) {
                continue;
            }
            visited
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async task cancellation",
                })?;
            visited.insert(current);
            let Some(task) = self.tasks.get(&current) else {
                continue;
            };
            if matches!(task.state, AsyncTaskState::Complete { .. }) {
                continue;
            }
            tasks_to_cancel
                .try_reserve(task.children.len())
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async task cancellation",
                })?;
            tasks_to_cancel.extend(
                task.children
                    .iter()
                    .copied()
                    .filter(|child| self.tasks.contains_key(child)),
            );
        }
        for task in tasks_to_cancel {
            if !self.request_task_cancellation(task)? && task == id {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn request_task_cancellation(&mut self, id: usize) -> Result<bool, RuntimeError> {
        let (complete_now, request, waiting_on) = self
            .tasks
            .get(&id)
            .map(|task| {
                let plugin_active = matches!(&task.target, AsyncTaskTarget::NativePlugin { .. });
                let complete_now = matches!(task.state, AsyncTaskState::Cold)
                    || matches!(task.state, AsyncTaskState::Queued) && task.continuation.is_none()
                    || matches!(task.state, AsyncTaskState::WaitingPlugin)
                    || plugin_active
                    || matches!(
                        task.state,
                        AsyncTaskState::WaitingUntil(_)
                            | AsyncTaskState::WaitingTimeout { .. }
                            | AsyncTaskState::WaitingSelect { .. }
                    );
                let request = self.task_cancellable(task) && !complete_now;
                let waiting_on = match &task.state {
                    AsyncTaskState::WaitingTask(target)
                    | AsyncTaskState::WaitingTimeout { target, .. } => {
                        Some(WaiterTarget::One(*target))
                    }
                    AsyncTaskState::WaitingSelect { targets } => {
                        Some(WaiterTarget::Many(Rc::clone(targets)))
                    }
                    _ => None,
                };
                (complete_now, request, waiting_on)
            })
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        if complete_now {
            self.ready.retain(|queued| *queued != id);
            self.timers.retain(|Reverse((_, timer_id))| *timer_id != id);
            if let Some(targets) = &waiting_on {
                self.remove_waiter_target(targets, id);
            }
            let target = self
                .tasks
                .get_mut(&id)
                .map(|task| std::mem::replace(&mut task.target, AsyncTaskTarget::Consumed));
            if let Some(AsyncTaskTarget::NativePlugin { mut operation }) = target {
                operation.cancel();
            }
            self.complete_task(id, Ok(Value::Error(Box::new(RuntimeError::Cancelled))))?;
            return Ok(true);
        }
        if request {
            let task = self.tasks.get_mut(&id).expect("task was checked above");
            task.cancel_requested = true;
            if let Some(cancelled) = &task.io_cancel {
                cancelled.store(true, Ordering::Release);
            }
            if let Some(targets) = &waiting_on {
                self.remove_waiter_target(targets, id);
            }
            if waiting_on.is_some() {
                self.ready
                    .try_reserve(1)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async ready queue",
                    })?;
                let task = self.tasks.get_mut(&id).expect("task was checked above");
                task.state = AsyncTaskState::Queued;
                task.resume_outcome = Some(Ok(Value::Error(Box::new(RuntimeError::Cancelled))));
                task.pending_await = None;
                task.pending_select = None;
                self.ready.push_back(id);
            }
        }
        Ok(request)
    }

    fn task_cancellable(&self, task: &AsyncTask) -> bool {
        match task.state {
            AsyncTaskState::Cold
            | AsyncTaskState::WaitingUntil(_)
            | AsyncTaskState::WaitingTask(_)
            | AsyncTaskState::WaitingTimeout { .. }
            | AsyncTaskState::WaitingSelect { .. }
            | AsyncTaskState::Joining
            | AsyncTaskState::Running
            | AsyncTaskState::WaitingIo
            | AsyncTaskState::WaitingPlugin => !task.cancel_requested,
            AsyncTaskState::Queued => !task.cancel_requested,
            AsyncTaskState::Complete { .. } => false,
        }
    }

    fn take_cancel_request(&mut self, id: usize) -> Result<bool, RuntimeError> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        Ok(std::mem::take(&mut task.cancel_requested))
    }

    fn suspend_on(
        &mut self,
        waiter: usize,
        awaited: TaskHandle,
    ) -> Result<Option<Value>, RuntimeError> {
        let target = self.task_id(&awaited)?;
        let (completed, is_running) = match self
            .tasks
            .get(&target)
            .map(|task| &task.state)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
        {
            AsyncTaskState::Complete { outcome, .. } => (Some(outcome.clone()), false),
            AsyncTaskState::Running => (None, true),
            _ => (None, false),
        };
        if let Some(outcome) = completed {
            self.tasks
                .get_mut(&target)
                .expect("awaited task was checked above")
                .observed = true;
            return await_outcome(outcome).map(Some);
        }
        if is_running {
            return Err(RuntimeError::InvalidArgument {
                op: "async task cycle",
            });
        }
        if self.wait_path_contains(waiter, target)? {
            return Err(RuntimeError::InvalidArgument {
                op: "async task cycle",
            });
        }
        self.tasks
            .get(&waiter)
            .filter(|task| matches!(task.state, AsyncTaskState::Running))
            .ok_or(RuntimeError::InvalidArgument {
                op: "async task state",
            })?;
        if !self.waiters.contains_key(&target) {
            self.waiters
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async waiters",
                })?;
        }
        let waiters = self.waiters.entry(target).or_default();
        waiters
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async waiters",
            })?;
        self.enqueue(target)?;
        let waiters = self
            .waiters
            .get_mut(&target)
            .expect("waiter list was reserved above");
        waiters.push(waiter);
        self.tasks
            .get_mut(&target)
            .expect("awaited task was checked above")
            .observed = true;
        let waiting = self
            .tasks
            .get_mut(&waiter)
            .expect("active task was checked above");
        waiting.state = AsyncTaskState::WaitingTask(target);
        waiting.pending_await = Some(awaited);
        Ok(None)
    }

    fn wait_path_contains(&self, waiter: usize, target: usize) -> Result<bool, RuntimeError> {
        let mut pending = Vec::new();
        pending
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async wait graph",
            })?;
        pending.push(target);
        let mut visited = HashSet::new();
        while let Some(current) = pending.pop() {
            if current == waiter {
                return Ok(true);
            }
            if !visited.contains(&current) {
                visited
                    .try_reserve(1)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "async wait graph",
                    })?;
                visited.insert(current);
                let Some(task) = self.tasks.get(&current) else {
                    continue;
                };
                match task.state {
                    AsyncTaskState::WaitingTask(next)
                    | AsyncTaskState::WaitingTimeout { target: next, .. } => {
                        pending
                            .try_reserve(1)
                            .map_err(|_| RuntimeError::ResourceLimit {
                                resource: "async wait graph",
                            })?;
                        pending.push(next);
                    }
                    AsyncTaskState::WaitingSelect { ref targets } => {
                        pending.try_reserve(targets.len()).map_err(|_| {
                            RuntimeError::ResourceLimit {
                                resource: "async wait graph",
                            }
                        })?;
                        pending.extend(targets.iter().map(|(target, _)| *target));
                    }
                    AsyncTaskState::Joining => {
                        let children = task.children.iter().filter(|child| {
                            self.tasks.get(child).is_some_and(|child_task| {
                                !matches!(child_task.state, AsyncTaskState::Complete { .. })
                            })
                        });
                        let count = children.clone().count();
                        pending
                            .try_reserve(count)
                            .map_err(|_| RuntimeError::ResourceLimit {
                                resource: "async wait graph",
                            })?;
                        pending.extend(children.copied());
                    }
                    _ => {}
                }
            }
        }
        Ok(false)
    }

    fn await_task(
        &mut self,
        task_handle: TaskHandle,
        artifact: &CompiledProgram,
        context: &mut RuntimeContext<'_>,
        owner: &Rc<()>,
        globals: &mut Vec<Option<Value>>,
        remaining_steps: &mut Option<u64>,
    ) -> Result<Value, RuntimeError> {
        let id = self.task_id(&task_handle)?;
        let is_running = self
            .tasks
            .get(&id)
            .map(|task| matches!(task.state, AsyncTaskState::Running))
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        if is_running {
            return Err(RuntimeError::InvalidArgument {
                op: "async task cycle",
            });
        }
        self.tasks
            .get_mut(&id)
            .expect("task was checked above")
            .observed = true;
        self.enqueue(id)?;
        loop {
            let completed = self
                .tasks
                .get(&id)
                .is_some_and(|task| matches!(task.state, AsyncTaskState::Complete { .. }));
            if completed {
                break;
            }
            let Some(next) = self.next_ready_or_wait(context)? else {
                let completed = self
                    .tasks
                    .get(&id)
                    .is_some_and(|task| matches!(task.state, AsyncTaskState::Complete { .. }));
                if completed {
                    continue;
                }
                return Err(RuntimeError::InvalidArgument {
                    op: "async executor deadlock",
                });
            };
            self.run_queued_task(next, artifact, context, owner, globals, remaining_steps)?;
        }
        let outcome = match &self
            .tasks
            .get(&id)
            .expect("awaited task remains in table")
            .state
        {
            AsyncTaskState::Complete { outcome, .. } => outcome.clone(),
            _ => Err(RuntimeError::InvalidArgument {
                op: "async task state",
            }),
        };
        await_outcome(outcome)
    }

    fn run_queued_task(
        &mut self,
        id: usize,
        artifact: &CompiledProgram,
        context: &mut RuntimeContext<'_>,
        owner: &Rc<()>,
        globals: &mut Vec<Option<Value>>,
        remaining_steps: &mut Option<u64>,
    ) -> Result<(), RuntimeError> {
        let (target, continuation, resume_outcome) = {
            let task = self
                .tasks
                .get_mut(&id)
                .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
            if !matches!(task.state, AsyncTaskState::Queued) {
                return Ok(());
            }
            task.state = AsyncTaskState::Running;
            (
                std::mem::replace(&mut task.target, AsyncTaskTarget::Consumed),
                task.continuation.take(),
                task.resume_outcome.take(),
            )
        };
        let start = match target {
            AsyncTaskTarget::Function { entry, arguments } => Some((entry, arguments)),
            AsyncTaskTarget::YieldNow => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.complete_task(id, Ok(Value::Nil))?;
                return Ok(());
            }
            AsyncTaskTarget::Timeout { task, seconds } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_timeout(id, task, seconds, context)?;
                return Ok(());
            }
            AsyncTaskTarget::Select2 { first, second } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_select2(id, first, second)?;
                return Ok(());
            }
            AsyncTaskTarget::Select(tasks) => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_select(id, tasks)?;
                return Ok(());
            }
            AsyncTaskTarget::Ready(outcome) => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.complete_task(id, outcome)?;
                return Ok(());
            }
            AsyncTaskTarget::Sleep(seconds) => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_sleep(id, seconds, context)?;
                return Ok(());
            }
            AsyncTaskTarget::Filesystem {
                operation,
                arguments,
            } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_filesystem(id, operation, arguments, context)?;
                return Ok(());
            }
            AsyncTaskTarget::FileHandle {
                operation,
                arguments,
            } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_file_handle(id, operation, arguments, context)?;
                return Ok(());
            }
            AsyncTaskTarget::ProcessRun {
                program,
                arguments,
                input,
                timeout,
                output_limit,
            } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_process_run(
                    id,
                    program,
                    arguments,
                    input,
                    timeout,
                    output_limit,
                    context,
                )?;
                return Ok(());
            }
            AsyncTaskTarget::NetworkExchange { arguments } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_network_exchange(id, arguments, context)?;
                return Ok(());
            }
            AsyncTaskTarget::Tcp {
                operation,
                arguments,
            } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.start_tcp_operation(id, operation, arguments, context)?;
                return Ok(());
            }
            AsyncTaskTarget::Native { import, arguments } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                let wake = self
                    .tasks
                    .get(&id)
                    .and_then(|task| task.plugin_wake.clone())
                    .ok_or(RuntimeError::InvalidArgument {
                        op: "async plugin wakeup state",
                    })?;
                match import.start_async(context, &arguments, wake) {
                    Ok(operation) => self.poll_plugin_operation(id, operation, context)?,
                    Err(error) => self.complete_task(id, Err(error))?,
                }
                return Ok(());
            }
            AsyncTaskTarget::NativePlugin { operation } => {
                if continuation.is_some() {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    });
                }
                self.poll_plugin_operation(id, operation, context)?;
                return Ok(());
            }
            AsyncTaskTarget::Consumed if continuation.is_some() => None,
            AsyncTaskTarget::Consumed => {
                return Err(RuntimeError::InvalidArgument {
                    op: "async task state",
                });
            }
        };
        if !self.active.is_empty() {
            self.complete_task(
                id,
                Err(RuntimeError::InvalidArgument {
                    op: "async scheduler reentry",
                }),
            )?;
            return Ok(());
        }
        if self.active.try_reserve(1).is_err() {
            self.complete_task(
                id,
                Err(RuntimeError::ResourceLimit {
                    resource: "async active task",
                }),
            )?;
            return Ok(());
        }
        self.active.push(id);
        let execution_plan =
            self.execution_plan
                .as_ref()
                .map(Rc::clone)
                .ok_or(RuntimeError::InvalidArgument {
                    op: "async execution plan",
                })?;
        let execution = execute_vm(
            artifact,
            context,
            owner,
            &execution_plan,
            self,
            globals,
            remaining_steps,
            start,
            Some(id),
            continuation,
            resume_outcome,
        );
        let popped = self.active.pop();
        if popped != Some(id) {
            return Err(RuntimeError::InvalidArgument {
                op: "async task stack",
            });
        }
        match execution {
            VmExecution::Suspended(continuation) => {
                let task = self
                    .tasks
                    .get_mut(&id)
                    .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
                if !matches!(task.state, AsyncTaskState::WaitingTask(_)) {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async task suspension",
                    });
                }
                task.continuation = Some(continuation);
                Ok(())
            }
            VmExecution::Complete(outcome) => {
                self.finish_task_body(id, outcome)?;
                self.reap_abandoned_tasks();
                Ok(())
            }
        }
    }

    fn poll_plugin_operation(
        &mut self,
        id: usize,
        mut operation: Box<dyn NativeAsyncOperation>,
        context: &mut RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        match operation.poll(context) {
            Ok(Some(value)) => self.complete_task(id, Ok(value)),
            Ok(None) => {
                self.pending_plugin_operations = self
                    .pending_plugin_operations
                    .checked_add(1)
                    .ok_or(RuntimeError::ResourceLimit {
                        resource: "async plugin operations",
                    })?;
                let task = self
                    .tasks
                    .get_mut(&id)
                    .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
                task.target = AsyncTaskTarget::NativePlugin { operation };
                task.state = AsyncTaskState::WaitingPlugin;
                Ok(())
            }
            Err(error) => self.complete_task(id, Err(error)),
        }
    }

    fn start_sleep(
        &mut self,
        id: usize,
        seconds: Value,
        context: &RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        if let Value::Error(error) = &seconds {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        let duration = match crate::runtime::async_sleep_duration(context, &[seconds]) {
            Ok(duration) => duration,
            Err(error) => return self.complete_task(id, Err(error)),
        };
        let Some(deadline) = Instant::now().checked_add(duration) else {
            return self.complete_task(
                id,
                Err(RuntimeError::InvalidArgument {
                    op: "async::time::sleep",
                }),
            );
        };
        self.timers
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async timers",
            })?;
        self.timers.push(Reverse((deadline, id)));
        self.tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .state = AsyncTaskState::WaitingUntil(deadline);
        Ok(())
    }

    fn start_timeout(
        &mut self,
        id: usize,
        task_value: Value,
        seconds: Value,
        context: &RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        if let Value::Error(error) = task_value {
            return self.complete_task(id, Ok(Value::Error(error)));
        }
        if let Value::Error(error) = seconds {
            return self.complete_task(id, Ok(Value::Error(error)));
        }
        let Value::Task(task_handle) = task_value else {
            return self.complete_task(
                id,
                Err(RuntimeError::TypeMismatch {
                    op: "async::time::timeout",
                }),
            );
        };
        let duration = match crate::runtime::async_timeout_duration(context, &[seconds]) {
            Ok(duration) => duration,
            Err(error) => return self.complete_task(id, Err(error)),
        };
        let Some(deadline) = Instant::now().checked_add(duration) else {
            return self.complete_task(
                id,
                Err(RuntimeError::InvalidArgument {
                    op: "async::time::timeout",
                }),
            );
        };
        let target = match self.task_id(&task_handle) {
            Ok(target) => target,
            Err(error) => return self.complete_task(id, Err(error)),
        };
        let target_outcome = match self
            .tasks
            .get(&target)
            .map(|task| &task.state)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
        {
            AsyncTaskState::Complete { outcome, .. } => Some(outcome.clone()),
            AsyncTaskState::Running if target == id => {
                return self.complete_task(
                    id,
                    Err(RuntimeError::InvalidArgument {
                        op: "async task cycle",
                    }),
                );
            }
            AsyncTaskState::Running => {
                return self.complete_task(
                    id,
                    Err(RuntimeError::InvalidArgument {
                        op: "async task state",
                    }),
                );
            }
            _ => None,
        };
        if let Some(outcome) = target_outcome {
            self.tasks
                .get_mut(&target)
                .expect("timed task was checked above")
                .observed = true;
            return self.complete_task(id, outcome);
        }
        if self.wait_path_contains(id, target)? {
            return self.complete_task(
                id,
                Err(RuntimeError::InvalidArgument {
                    op: "async task cycle",
                }),
            );
        }
        self.timers
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async timers",
            })?;
        if !self.waiters.contains_key(&target) {
            self.waiters
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async waiters",
                })?;
        }
        self.waiters
            .entry(target)
            .or_default()
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async waiters",
            })?;
        self.enqueue(target)?;
        self.waiters
            .get_mut(&target)
            .expect("waiter list was reserved above")
            .push(id);
        let timeout = self
            .tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        timeout.state = AsyncTaskState::WaitingTimeout { target, deadline };
        timeout.pending_await = Some(task_handle);
        self.timers.push(Reverse((deadline, id)));
        Ok(())
    }

    fn start_select2(
        &mut self,
        id: usize,
        first_value: Value,
        second_value: Value,
    ) -> Result<(), RuntimeError> {
        if let Value::Error(error) = &first_value {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        if let Value::Error(error) = &second_value {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        let (Value::Task(first), Value::Task(second)) = (first_value, second_value) else {
            return self.complete_task(
                id,
                Err(RuntimeError::TypeMismatch {
                    op: "task::select2",
                }),
            );
        };
        let mut handles = Vec::new();
        handles
            .try_reserve(2)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async task selection",
            })?;
        handles.extend([first, second]);
        self.start_select_handles(id, handles)
    }

    fn start_select(&mut self, id: usize, tasks_value: Value) -> Result<(), RuntimeError> {
        if let Value::Error(error) = &tasks_value {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        let Value::Array(array) = tasks_value else {
            return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: "task::select" }));
        };
        let values = match array.try_borrow() {
            Ok(values) => values,
            Err(_) => {
                return self
                    .complete_task(id, Err(RuntimeError::TypeMismatch { op: "task::select" }));
            }
        };
        if values.is_empty() {
            return self.complete_task(
                id,
                Err(RuntimeError::InvalidArgument {
                    op: "task::select requires at least one task",
                }),
            );
        }
        if values.len() > MAX_ASYNC_TASKS {
            return self.complete_task(
                id,
                Err(RuntimeError::ResourceLimit {
                    resource: "async task selection",
                }),
            );
        }
        let mut handles = Vec::new();
        handles
            .try_reserve(values.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async task selection",
            })?;
        for value in values.iter() {
            match value {
                Value::Task(handle) => handles.push(handle.clone()),
                Value::Error(error) => {
                    return self.complete_task(id, Ok(Value::Error(error.clone())));
                }
                _ => {
                    return self
                        .complete_task(id, Err(RuntimeError::TypeMismatch { op: "task::select" }));
                }
            }
        }
        drop(values);
        self.start_select_handles(id, handles)
    }

    fn start_select_handles(
        &mut self,
        id: usize,
        handles: Vec<TaskHandle>,
    ) -> Result<(), RuntimeError> {
        if handles.is_empty() {
            return self.complete_task(
                id,
                Err(RuntimeError::InvalidArgument {
                    op: "task::select requires at least one task",
                }),
            );
        }
        if handles.len() > MAX_ASYNC_TASKS {
            return self.complete_task(
                id,
                Err(RuntimeError::ResourceLimit {
                    resource: "async task selection",
                }),
            );
        }
        let mut targets = Vec::new();
        targets
            .try_reserve(handles.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async task selection",
            })?;
        let mut seen = HashSet::new();
        seen.try_reserve(handles.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async task selection",
            })?;
        let mut parent = None;
        let mut completed_winner: Option<(u64, usize)> = None;
        for (index, handle) in handles.iter().enumerate() {
            let target = match self.task_id(handle) {
                Ok(target) => target,
                Err(error) => return self.complete_task(id, Err(error)),
            };
            if seen.insert(target) {
                targets.push((target, index));
            }
            let Some(source) = self.tasks.get(&target) else {
                return self
                    .complete_task(id, Err(RuntimeError::InvalidArgument { op: "async task" }));
            };
            let Some(source_parent) = source.parent else {
                return self.complete_task(
                    id,
                    Err(RuntimeError::InvalidArgument {
                        op: "task::select requires spawned tasks",
                    }),
                );
            };
            if matches!(source.state, AsyncTaskState::Cold) {
                return self.complete_task(
                    id,
                    Err(RuntimeError::InvalidArgument {
                        op: "task::select requires spawned tasks",
                    }),
                );
            }
            if parent.is_some_and(|parent| parent != source_parent) {
                return self.complete_task(
                    id,
                    Err(RuntimeError::InvalidArgument {
                        op: "task::select requires sibling tasks",
                    }),
                );
            }
            parent = Some(source_parent);
            if let AsyncTaskState::Complete { completed_at, .. } = source.state
                && completed_winner.is_none_or(|(order, winner)| {
                    completed_at < order || completed_at == order && index < winner
                })
            {
                completed_winner = Some((completed_at, index));
            }
        }
        if let Some((_, index)) = completed_winner {
            return self.complete_task(id, Ok(Value::Int(index as i64)));
        }
        for (target, _) in &targets {
            if self.wait_path_contains(id, *target)? {
                return self.complete_task(
                    id,
                    Err(RuntimeError::InvalidArgument {
                        op: "async task cycle",
                    }),
                );
            }
        }
        let missing_waiter_lists = targets
            .iter()
            .filter(|(target, _)| !self.waiters.contains_key(target))
            .count();
        self.waiters
            .try_reserve(missing_waiter_lists)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async waiters",
            })?;
        for (target, _) in &targets {
            self.waiters
                .entry(*target)
                .or_default()
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "async waiters",
                })?;
        }
        let targets = Rc::new(targets);
        for (target, _) in targets.iter() {
            self.waiters
                .get_mut(target)
                .expect("waiter list was reserved above")
                .push(id);
        }
        let selector = self
            .tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        selector.state = AsyncTaskState::WaitingSelect {
            targets: Rc::clone(&targets),
        };
        selector.pending_select = Some(handles);
        Ok(())
    }

    fn start_filesystem(
        &mut self,
        id: usize,
        operation: AsyncFsOperation,
        arguments: Vec<Value>,
        context: &RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        let op = operation.name();
        if arguments.len() != operation.arity() {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
        }
        if let Some(Value::Error(error)) = arguments
            .iter()
            .find(|argument| matches!(argument, Value::Error(_)))
        {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        let path = match arguments.first() {
            Some(Value::Str(path)) => path.to_string(),
            _ => {
                return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
            }
        };
        if operation.writes() && !context.capabilities.allow_fs_write {
            return self.complete_task(id, Err(RuntimeError::PermissionDenied { op }));
        }
        if !operation.writes() && !context.capabilities.allow_fs_read {
            return self.complete_task(id, Err(RuntimeError::PermissionDenied { op }));
        }
        let job = match operation {
            AsyncFsOperation::Metadata => AsyncFsJob::Metadata {
                task: id,
                path,
                limit: context.limits.buffer_limit(),
            },
            AsyncFsOperation::ReadDir => AsyncFsJob::ReadDir {
                task: id,
                path,
                limit: context.limits.buffer_limit(),
            },
            AsyncFsOperation::ReadBytes => AsyncFsJob::ReadBytes {
                task: id,
                path,
                limit: context.limits.buffer_limit(),
            },
            AsyncFsOperation::ReadText => AsyncFsJob::ReadText {
                task: id,
                path,
                limit: context.limits.buffer_limit(),
            },
            AsyncFsOperation::ReadLines => AsyncFsJob::ReadLines {
                task: id,
                path,
                limit: context.limits.buffer_limit(),
            },
            AsyncFsOperation::WriteBytes => {
                let Some(Value::Bytes(bytes)) = arguments.get(1) else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let bytes = match copy_async_write_bytes(bytes, context.limits.buffer_limit(), op) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                AsyncFsJob::WriteBytes {
                    task: id,
                    path,
                    bytes,
                }
            }
            AsyncFsOperation::WriteText => {
                let Some(Value::Str(text)) = arguments.get(1) else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let bytes = match copy_async_write_text(text, context.limits.buffer_limit()) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                AsyncFsJob::WriteText {
                    task: id,
                    path,
                    bytes,
                }
            }
            AsyncFsOperation::AppendBytes => {
                let Some(Value::Bytes(bytes)) = arguments.get(1) else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let bytes = match copy_async_write_bytes(bytes, context.limits.buffer_limit(), op) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                AsyncFsJob::AppendBytes {
                    task: id,
                    path,
                    bytes,
                }
            }
            AsyncFsOperation::AppendText => {
                let Some(Value::Str(text)) = arguments.get(1) else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let bytes = match copy_async_write_text(text, context.limits.buffer_limit()) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                AsyncFsJob::AppendText {
                    task: id,
                    path,
                    bytes,
                }
            }
            AsyncFsOperation::Copy | AsyncFsOperation::Rename => {
                let Some(Value::Str(destination)) = arguments.get(1) else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let destination = destination.to_string();
                if matches!(operation, AsyncFsOperation::Copy) {
                    AsyncFsJob::Copy {
                        task: id,
                        source: path,
                        destination,
                    }
                } else {
                    AsyncFsJob::Rename {
                        task: id,
                        source: path,
                        destination,
                    }
                }
            }
            AsyncFsOperation::CreateDir => AsyncFsJob::CreateDir { task: id, path },
            AsyncFsOperation::CreateDirAll => AsyncFsJob::CreateDirAll { task: id, path },
            AsyncFsOperation::RemoveFile => AsyncFsJob::RemoveFile { task: id, path },
            AsyncFsOperation::RemoveDir => AsyncFsJob::RemoveDir { task: id, path },
        };
        let cancelled = self.create_io_cancellation(id)?;
        self.submit_io(id, AsyncIoJob::Filesystem { job, cancelled })
    }

    fn start_file_handle(
        &mut self,
        id: usize,
        operation: AsyncFileOperation,
        arguments: Vec<Value>,
        context: &mut RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        let op = operation.name();
        if arguments.len() != operation.arity() {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
        }
        if let Some(Value::Error(error)) = arguments
            .iter()
            .find(|argument| matches!(argument, Value::Error(_)))
        {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        if matches!(operation, AsyncFileOperation::Open) {
            return self.start_file_open(id, arguments, context);
        }

        let submission = match operation {
            AsyncFileOperation::Open => unreachable!(),
            AsyncFileOperation::Read => {
                let max_bytes = match &arguments[1] {
                    Value::Int(bytes) => usize::try_from(*bytes).ok(),
                    Value::UInt(bytes) => usize::try_from(*bytes).ok(),
                    _ => {
                        return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                    }
                };
                let Some(max_bytes) = max_bytes else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                if max_bytes > context.limits.buffer_limit() {
                    return self
                        .complete_task(id, Err(RuntimeError::ResourceLimit { resource: "bytes" }));
                }
                let stream = match context.async_io_read_stream(&arguments[0], op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                match stream {
                    AsyncIoReadStream::File(file) => {
                        AsyncIoSubmission::Worker(AsyncIoJob::FileRead {
                            task: id,
                            file,
                            max_bytes,
                            cancelled,
                            op,
                        })
                    }
                    AsyncIoReadStream::Stdin => {
                        AsyncIoSubmission::Stdin(AsyncStdioInputJob::Read {
                            task: id,
                            max_bytes,
                            cancelled,
                            op,
                        })
                    }
                }
            }
            AsyncFileOperation::ReadExact => {
                let size = match &arguments[1] {
                    Value::Int(size) => usize::try_from(*size).ok(),
                    Value::UInt(size) => usize::try_from(*size).ok(),
                    _ => return self.complete_task(id, Err(RuntimeError::TypeMismatch { op })),
                };
                let Some(size) = size else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                if size > context.limits.buffer_limit() {
                    return self
                        .complete_task(id, Err(RuntimeError::ResourceLimit { resource: "bytes" }));
                }
                let stream = match context.async_io_read_stream(&arguments[0], op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                match stream {
                    AsyncIoReadStream::File(file) => {
                        AsyncIoSubmission::Worker(AsyncIoJob::FileReadExact {
                            task: id,
                            file,
                            size,
                            cancelled,
                            op,
                        })
                    }
                    AsyncIoReadStream::Stdin => {
                        AsyncIoSubmission::Stdin(AsyncStdioInputJob::ReadExact {
                            task: id,
                            size,
                            cancelled,
                            op,
                        })
                    }
                }
            }
            AsyncFileOperation::ReadAll => {
                let stream = match context.async_io_read_stream(&arguments[0], op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let limit = context.limits.buffer_limit();
                let cancelled = self.create_io_cancellation(id)?;
                match stream {
                    AsyncIoReadStream::File(file) => {
                        AsyncIoSubmission::Worker(AsyncIoJob::FileReadAll {
                            task: id,
                            file,
                            max_bytes: limit,
                            cancelled,
                            op,
                        })
                    }
                    AsyncIoReadStream::Stdin => {
                        AsyncIoSubmission::Stdin(AsyncStdioInputJob::ReadAll {
                            task: id,
                            limit,
                            cancelled,
                            op,
                        })
                    }
                }
            }
            AsyncFileOperation::Write => {
                let Value::Bytes(bytes) = &arguments[1] else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let bytes = match copy_async_write_bytes(bytes, context.limits.buffer_limit(), op) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let stream = match context.async_io_write_stream(&arguments[0], op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                match stream {
                    AsyncIoWriteStream::File(file) => {
                        AsyncIoSubmission::Worker(AsyncIoJob::FileWrite {
                            task: id,
                            file,
                            bytes,
                            cancelled,
                            op,
                        })
                    }
                    AsyncIoWriteStream::Stdout => {
                        AsyncIoSubmission::Stdout(AsyncStdioOutputJob::Write {
                            task: id,
                            bytes,
                            all: false,
                            cancelled,
                            op,
                        })
                    }
                }
            }
            AsyncFileOperation::WriteAll => {
                let Value::Bytes(bytes) = &arguments[1] else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let bytes = match copy_async_write_bytes(bytes, context.limits.buffer_limit(), op) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let stream = match context.async_io_write_stream(&arguments[0], op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                match stream {
                    AsyncIoWriteStream::File(file) => {
                        AsyncIoSubmission::Worker(AsyncIoJob::FileWriteAll {
                            task: id,
                            file,
                            bytes,
                            cancelled,
                            op,
                        })
                    }
                    AsyncIoWriteStream::Stdout => {
                        AsyncIoSubmission::Stdout(AsyncStdioOutputJob::Write {
                            task: id,
                            bytes,
                            all: true,
                            cancelled,
                            op,
                        })
                    }
                }
            }
            AsyncFileOperation::Flush => {
                let stream = match context.async_io_write_stream(&arguments[0], op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                match stream {
                    AsyncIoWriteStream::File(file) => {
                        AsyncIoSubmission::Worker(AsyncIoJob::FileFlush {
                            task: id,
                            file,
                            cancelled,
                            op,
                        })
                    }
                    AsyncIoWriteStream::Stdout => {
                        AsyncIoSubmission::Stdout(AsyncStdioOutputJob::Flush {
                            task: id,
                            cancelled,
                            op,
                        })
                    }
                }
            }
            AsyncFileOperation::Seek => {
                let Value::Int(offset) = &arguments[1] else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let Value::Str(origin) = &arguments[2] else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                let position = match origin.as_ref() {
                    "start" => match u64::try_from(*offset) {
                        Ok(offset) => std::io::SeekFrom::Start(offset),
                        Err(_) => {
                            return self
                                .complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                        }
                    },
                    "current" => std::io::SeekFrom::Current(*offset),
                    "end" => std::io::SeekFrom::End(*offset),
                    _ => return self.complete_task(id, Err(RuntimeError::InvalidArgument { op })),
                };
                let file = match context.file_stream_for_seek(&arguments[0], op) {
                    Ok(file) => file,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                AsyncIoSubmission::Worker(AsyncIoJob::FileSeek {
                    task: id,
                    file,
                    position,
                    cancelled: self.create_io_cancellation(id)?,
                    op,
                })
            }
        };
        self.submit_io_submission(id, submission)
    }

    fn start_file_open(
        &mut self,
        id: usize,
        arguments: Vec<Value>,
        context: &mut RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        const OP: &str = "async::io::open";
        let Some(Value::Str(path)) = arguments.first() else {
            return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: OP }));
        };
        let Some(Value::Str(mode_name)) = arguments.get(1) else {
            return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: OP }));
        };
        let Some(mode) = FileOpenMode::parse(mode_name.as_ref()) else {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op: OP }));
        };
        if (mode.readable && !context.capabilities.allow_fs_read)
            || (mode.writable && !context.capabilities.allow_fs_write)
        {
            return self.complete_task(id, Err(RuntimeError::PermissionDenied { op: OP }));
        }
        let Some(Value::Int(buffer_size)) = arguments.get(2) else {
            return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: OP }));
        };
        let capacity = match checked_size(context, *buffer_size) {
            Ok(capacity) => capacity,
            Err(error) => return self.complete_task(id, Err(error)),
        };
        if self.file_open_reservations.try_reserve(1).is_err() {
            return self.complete_task(
                id,
                Err(RuntimeError::ResourceLimit {
                    resource: "async file reservations",
                }),
            );
        }
        let reservation = match context.reserve_file_slot() {
            Ok(slot) => slot,
            Err(error) => return self.complete_task(id, Err(error)),
        };
        let buffer = match file_write_buffer(mode, capacity) {
            Ok(buffer) => buffer,
            Err(error) => {
                context.release_file_slot(reservation)?;
                return self.complete_task(id, Err(error));
            }
        };
        let cancelled = match self.create_io_cancellation(id) {
            Ok(cancelled) => cancelled,
            Err(error) => {
                context.release_file_slot(reservation)?;
                return Err(error);
            }
        };
        let job = AsyncIoJob::FileOpen {
            task: id,
            path: path.to_string(),
            mode,
            buffer,
            cancelled,
            op: OP,
        };
        if let Err(error) = self.submit_io(id, job) {
            context.release_file_slot(reservation)?;
            return Err(error);
        }
        if self
            .tasks
            .get(&id)
            .is_some_and(|task| matches!(task.state, AsyncTaskState::WaitingIo))
        {
            self.file_open_reservations.insert(id, reservation);
        } else {
            context.release_file_slot(reservation)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn start_process_run(
        &mut self,
        id: usize,
        program: Value,
        arguments: Value,
        input: Option<Value>,
        timeout: Option<Value>,
        output_limit: Option<Value>,
        context: &RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        const OP: &str = "process::run";
        let (program, array) = match process_call_parts(program, arguments) {
            Ok(arguments) => arguments,
            Err(value) => return self.complete_task(id, Ok(value)),
        };
        if !context.capabilities.allow_exec {
            return self.complete_task(id, Err(RuntimeError::PermissionDenied { op: OP }));
        }
        let timeout = match timeout {
            Some(Value::Nil) => None,
            Some(value) => match async_timeout_millis(Some(&value)) {
                Some(timeout) if (1..=300_000).contains(&timeout) => {
                    Some(std::time::Duration::from_millis(timeout))
                }
                _ => {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op: OP }));
                }
            },
            None => None,
        };
        let output_limit = match output_limit {
            Some(value) => match async_output_limit(&value) {
                Some(limit) => Some(limit),
                None => {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op: OP }));
                }
            },
            None => None,
        };
        let arguments = match process_argument_strings(array) {
            Ok(arguments) => arguments,
            Err(value) => return self.complete_task(id, Ok(value)),
        };
        let input = match input {
            Some(Value::Bytes(bytes)) => {
                match copy_async_write_bytes(&bytes, context.limits.buffer_limit(), OP) {
                    Ok(input) => Some(input),
                    Err(error) => return self.complete_task(id, Err(error)),
                }
            }
            Some(_) => return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: OP })),
            None => None,
        };
        let cancelled = self.create_io_cancellation(id)?;
        self.submit_io(
            id,
            AsyncIoJob::ProcessRun {
                task: id,
                program,
                arguments,
                input,
                limit: context.limits.buffer_limit(),
                output_limit,
                timeout,
                cancelled,
            },
        )
    }

    fn create_io_cancellation(&mut self, id: usize) -> Result<Arc<AtomicBool>, RuntimeError> {
        let cancelled = Arc::new(AtomicBool::new(false));
        self.tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .io_cancel = Some(Arc::clone(&cancelled));
        Ok(cancelled)
    }

    fn start_network_exchange(
        &mut self,
        id: usize,
        arguments: Vec<Value>,
        context: &RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        const OP: &str = "async::net::exchange";
        if arguments.len() != 4 {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op: OP }));
        }
        if let Some(Value::Error(error)) = arguments
            .iter()
            .find(|argument| matches!(argument, Value::Error(_)))
        {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        if !context.capabilities.allow_network {
            return self.complete_task(id, Err(RuntimeError::PermissionDenied { op: OP }));
        }
        let Some(Value::Str(host)) = arguments.first() else {
            return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: OP }));
        };
        let port = match arguments.get(1) {
            Some(Value::Int(port)) => u16::try_from(*port).ok(),
            Some(Value::UInt(port)) => u16::try_from(*port).ok(),
            _ => None,
        };
        let Some(port) = port.filter(|port| *port != 0) else {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op: OP }));
        };
        let Some(Value::Bytes(payload)) = arguments.get(2) else {
            return self.complete_task(id, Err(RuntimeError::TypeMismatch { op: OP }));
        };
        let timeout_millis = match arguments.get(3) {
            Some(Value::Int(timeout)) => u64::try_from(*timeout).ok(),
            Some(Value::UInt(timeout)) => Some(*timeout),
            _ => None,
        };
        let Some(timeout_millis) = timeout_millis.filter(|timeout| (1..=300_000).contains(timeout))
        else {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op: OP }));
        };
        let payload = match copy_async_write_bytes(payload, context.limits.buffer_limit(), OP) {
            Ok(payload) => payload,
            Err(error) => return self.complete_task(id, Err(error)),
        };
        let cancelled = self.create_io_cancellation(id)?;
        self.submit_io(
            id,
            AsyncIoJob::NetworkExchange {
                task: id,
                host: host.to_string(),
                port,
                payload,
                timeout: std::time::Duration::from_millis(timeout_millis),
                limit: context.limits.buffer_limit(),
                cancelled,
            },
        )
    }

    fn start_tcp_operation(
        &mut self,
        id: usize,
        operation: AsyncTcpOperation,
        arguments: Vec<Value>,
        context: &RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        let op = operation.name();
        if arguments.len() != operation.arity() {
            return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
        }
        if let Some(Value::Error(error)) = arguments
            .iter()
            .find(|argument| matches!(argument, Value::Error(_)))
        {
            return self.complete_task(id, Ok(Value::Error(error.clone())));
        }
        if !context.capabilities.allow_network {
            return self.complete_task(id, Err(RuntimeError::PermissionDenied { op }));
        }

        let job = match operation {
            AsyncTcpOperation::Listen => {
                let Some(Value::Str(host)) = arguments.first() else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                if host.is_empty() {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                }
                if host.len() > context.limits.buffer_limit() {
                    return self
                        .complete_task(id, Err(RuntimeError::ResourceLimit { resource: "bytes" }));
                }
                let port = match arguments.get(1) {
                    Some(Value::Int(port)) => u16::try_from(*port).ok(),
                    Some(Value::UInt(port)) => u16::try_from(*port).ok(),
                    _ => None,
                };
                let Some(port) = port.filter(|port| *port != 0) else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpListen {
                    task: id,
                    host: host.to_string(),
                    port,
                    cancelled,
                }
            }
            AsyncTcpOperation::Accept => {
                let accept_timeout = match async_timeout_millis(arguments.get(1)) {
                    Some(timeout) if (1..=300_000).contains(&timeout) => {
                        std::time::Duration::from_millis(timeout)
                    }
                    _ => {
                        return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                    }
                };
                let stream_timeout = match async_timeout_millis(arguments.get(2)) {
                    Some(timeout) if (1..=300_000).contains(&timeout) => {
                        std::time::Duration::from_millis(timeout)
                    }
                    _ => {
                        return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                    }
                };
                let listener = match context.tcp_listener(&arguments[0], op) {
                    Ok(listener) => listener,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpAccept {
                    task: id,
                    listener,
                    accept_timeout,
                    stream_timeout,
                    cancelled,
                    op,
                }
            }
            AsyncTcpOperation::Connect => {
                let Some(Value::Str(host)) = arguments.first() else {
                    return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                };
                if host.is_empty() {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                }
                if host.len() > context.limits.buffer_limit() {
                    return self
                        .complete_task(id, Err(RuntimeError::ResourceLimit { resource: "bytes" }));
                }
                let port = match arguments.get(1) {
                    Some(Value::Int(port)) => u16::try_from(*port).ok(),
                    Some(Value::UInt(port)) => u16::try_from(*port).ok(),
                    _ => None,
                };
                let Some(port) = port.filter(|port| *port != 0) else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                let timeout_millis = match arguments.get(2) {
                    Some(Value::Int(timeout)) => u64::try_from(*timeout).ok(),
                    Some(Value::UInt(timeout)) => Some(*timeout),
                    _ => None,
                };
                let Some(timeout_millis) =
                    timeout_millis.filter(|timeout| (1..=300_000).contains(timeout))
                else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpConnect {
                    task: id,
                    host: host.to_string(),
                    port,
                    timeout: std::time::Duration::from_millis(timeout_millis),
                    cancelled,
                }
            }
            AsyncTcpOperation::Read => {
                let max_bytes = match arguments.get(1) {
                    Some(Value::Int(bytes)) => usize::try_from(*bytes).ok(),
                    Some(Value::UInt(bytes)) => usize::try_from(*bytes).ok(),
                    _ => None,
                };
                let Some(max_bytes) = max_bytes.filter(|bytes| *bytes > 0) else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                if max_bytes > context.limits.buffer_limit() {
                    return self
                        .complete_task(id, Err(RuntimeError::ResourceLimit { resource: "bytes" }));
                }
                let (stream, timeout) = match context.tcp_stream(&arguments[0], false, op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpRead {
                    task: id,
                    stream,
                    max_bytes,
                    timeout,
                    cancelled,
                    op,
                }
            }
            AsyncTcpOperation::ReadExact => {
                let size = match arguments.get(1) {
                    Some(Value::Int(size)) => usize::try_from(*size).ok(),
                    Some(Value::UInt(size)) => usize::try_from(*size).ok(),
                    _ => {
                        return self.complete_task(id, Err(RuntimeError::TypeMismatch { op }));
                    }
                };
                let Some(size) = size else {
                    return self.complete_task(id, Err(RuntimeError::InvalidArgument { op }));
                };
                if size > context.limits.buffer_limit() {
                    return self
                        .complete_task(id, Err(RuntimeError::ResourceLimit { resource: "bytes" }));
                }
                let (stream, timeout) = match context.tcp_stream(&arguments[0], false, op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpReadExact {
                    task: id,
                    stream,
                    size,
                    timeout,
                    cancelled,
                    op,
                }
            }
            AsyncTcpOperation::ReadAll => {
                let (stream, timeout) = match context.tcp_stream(&arguments[0], false, op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpReadAll {
                    task: id,
                    stream,
                    limit: context.limits.buffer_limit(),
                    timeout,
                    cancelled,
                    op,
                }
            }
            AsyncTcpOperation::Write => {
                let bytes = match &arguments[1] {
                    Value::Bytes(bytes) => bytes,
                    _ => return self.complete_task(id, Err(RuntimeError::TypeMismatch { op })),
                };
                let (stream, timeout) = match context.tcp_stream(&arguments[0], true, op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let bytes = match copy_async_write_bytes(bytes, context.limits.buffer_limit(), op) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpWrite {
                    task: id,
                    stream,
                    bytes,
                    timeout,
                    cancelled,
                    op,
                }
            }
            AsyncTcpOperation::WriteAll => {
                let bytes = match &arguments[1] {
                    Value::Bytes(bytes) => bytes,
                    _ => return self.complete_task(id, Err(RuntimeError::TypeMismatch { op })),
                };
                let (stream, timeout) = match context.tcp_stream(&arguments[0], true, op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let bytes = match copy_async_write_bytes(bytes, context.limits.buffer_limit(), op) {
                    Ok(bytes) => bytes,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpWriteAll {
                    task: id,
                    stream,
                    bytes,
                    timeout,
                    cancelled,
                    op,
                }
            }
            AsyncTcpOperation::ShutdownWrite => {
                let (stream, _) = match context.tcp_stream(&arguments[0], true, op) {
                    Ok(stream) => stream,
                    Err(error) => return self.complete_task(id, Err(error)),
                };
                let cancelled = self.create_io_cancellation(id)?;
                AsyncIoJob::TcpShutdownWrite {
                    task: id,
                    stream,
                    cancelled,
                    op,
                }
            }
        };
        self.submit_io(id, job)
    }

    fn submit_io(&mut self, id: usize, job: AsyncIoJob) -> Result<(), RuntimeError> {
        if !self.check_io_capacity(id)? {
            return Ok(());
        }
        if self.io_pool.is_none() {
            self.io_pool = match AsyncIoPool::new(self.ensure_io_channel()) {
                Ok(pool) => Some(pool),
                Err(error) => {
                    return self.complete_task(
                        id,
                        Err(RuntimeError::IoFailure {
                            op: "async I/O",
                            message: error.to_string(),
                        }),
                    );
                }
            };
        }
        let pool = self.io_pool.as_ref().expect("pool initialized above");
        if let Err(error) = pool.submit(job) {
            return self.complete_task(id, Err(error.into_runtime_error()));
        }
        self.mark_io_pending(id)
    }

    fn submit_io_submission(
        &mut self,
        id: usize,
        submission: AsyncIoSubmission,
    ) -> Result<(), RuntimeError> {
        match submission {
            AsyncIoSubmission::Worker(job) => self.submit_io(id, job),
            AsyncIoSubmission::Stdin(job) => {
                if !self.check_io_capacity(id)? {
                    return Ok(());
                }
                let result = self
                    .stdio_pool
                    .as_ref()
                    .ok_or_else(|| AsyncIoFailure::Io {
                        op: "async::io",
                        message: "async stdin worker was not initialized".into(),
                    })
                    .and_then(|pool| pool.submit_input(job));
                if let Err(error) = result {
                    return self.complete_task(id, Err(error.into_runtime_error()));
                }
                self.mark_io_pending(id)
            }
            AsyncIoSubmission::Stdout(job) => {
                if !self.check_io_capacity(id)? {
                    return Ok(());
                }
                let result = self
                    .stdio_pool
                    .as_ref()
                    .ok_or_else(|| AsyncIoFailure::Io {
                        op: "async::io",
                        message: "async stdout worker was not initialized".into(),
                    })
                    .and_then(|pool| pool.submit_output(job));
                if let Err(error) = result {
                    return self.complete_task(id, Err(error.into_runtime_error()));
                }
                self.mark_io_pending(id)
            }
        }
    }

    fn check_io_capacity(&mut self, id: usize) -> Result<bool, RuntimeError> {
        if self.pending_io == usize::MAX {
            self.complete_task(
                id,
                Err(RuntimeError::ResourceLimit {
                    resource: "async I/O tasks",
                }),
            )?;
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn mark_io_pending(&mut self, id: usize) -> Result<(), RuntimeError> {
        self.pending_io += 1;
        self.tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .state = AsyncTaskState::WaitingIo;
        Ok(())
    }

    fn next_ready_or_wait(
        &mut self,
        context: &mut RuntimeContext<'_>,
    ) -> Result<Option<usize>, RuntimeError> {
        loop {
            self.wake_timers()?;
            self.wake_plugin_tasks()?;
            if let Some(task) = self.ready.pop_front() {
                return Ok(Some(task));
            }
            let deadline = self.timers.peek().map(|Reverse((deadline, _))| *deadline);
            let plugin_pending = self.pending_plugin_operations > 0;
            if self.pending_io == 0 && deadline.is_none() && !plugin_pending {
                return Ok(None);
            }
            let timer_timeout =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            let mut timeout = timer_timeout;
            if plugin_pending && self.pending_io > 0 {
                let poll_interval = std::time::Duration::from_millis(10);
                timeout = Some(timeout.map_or(poll_interval, |timeout| timeout.min(poll_interval)));
            }
            if self.pending_io == 0 {
                if plugin_pending {
                    match timer_timeout {
                        Some(timeout) => {
                            let received = self
                                .plugin_wake_receiver
                                .as_ref()
                                .ok_or(RuntimeError::InvalidArgument {
                                    op: "async plugin wakeup channel",
                                })?
                                .recv_timeout(timeout);
                            match received {
                                Ok(task) => self.wake_plugin_task(task)?,
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                    return Err(RuntimeError::InvalidArgument {
                                        op: "async plugin wakeup channel",
                                    });
                                }
                            }
                        }
                        None => {
                            let received = self
                                .plugin_wake_receiver
                                .as_ref()
                                .ok_or(RuntimeError::InvalidArgument {
                                    op: "async plugin wakeup channel",
                                })?
                                .recv();
                            match received {
                                Ok(task) => self.wake_plugin_task(task)?,
                                Err(_) => {
                                    return Err(RuntimeError::InvalidArgument {
                                        op: "async plugin wakeup channel",
                                    });
                                }
                            }
                        }
                    }
                } else if let Some(timeout) = timeout.filter(|timeout| !timeout.is_zero()) {
                    std::thread::sleep(timeout);
                }
                continue;
            }
            let completion =
                {
                    let io_completions =
                        self.io_completions
                            .as_ref()
                            .ok_or(RuntimeError::InvalidArgument {
                                op: "async I/O completion channel",
                            })?;
                    match timeout {
                        Some(timeout) => match io_completions.recv_timeout(timeout) {
                            Ok(completion) => Some(completion),
                            Err(mpsc::RecvTimeoutError::Timeout) => None,
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                return Err(RuntimeError::InvalidArgument {
                                    op: "async I/O completion channel",
                                });
                            }
                        },
                        None => Some(io_completions.recv().map_err(|_| {
                            RuntimeError::InvalidArgument {
                                op: "async I/O completion channel",
                            }
                        })?),
                    }
                };
            if let Some(completion) = completion {
                self.complete_io(completion, context)?;
            }
        }
    }

    fn wake_plugin_tasks(&mut self) -> Result<(), RuntimeError> {
        loop {
            let received = match self.plugin_wake_receiver.as_ref() {
                Some(receiver) => receiver.try_recv(),
                None => return Ok(()),
            };
            match received {
                Ok(id) => self.wake_plugin_task(id)?,
                Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(()),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(RuntimeError::InvalidArgument {
                        op: "async plugin wakeup channel",
                    });
                }
            }
        }
    }

    fn wake_plugin_task(&mut self, id: usize) -> Result<(), RuntimeError> {
        let Some(task) = self.tasks.get_mut(&id) else {
            return Ok(());
        };
        if let Some(wake) = &task.plugin_wake {
            wake.reset();
        }
        if !matches!(task.state, AsyncTaskState::WaitingPlugin) {
            return Ok(());
        }
        self.ready
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async ready queue",
            })?;
        self.pending_plugin_operations =
            self.pending_plugin_operations
                .checked_sub(1)
                .ok_or(RuntimeError::InvalidArgument {
                    op: "async plugin operation count",
                })?;
        task.state = AsyncTaskState::Queued;
        self.ready.push_back(id);
        Ok(())
    }

    fn wake_timers(&mut self) -> Result<(), RuntimeError> {
        self.prune_stale_timers();
        let now = Instant::now();
        while self
            .timers
            .peek()
            .is_some_and(|Reverse((deadline, _))| *deadline <= now)
        {
            let Reverse((deadline, id)) = self.timers.pop().expect("timer was checked");
            let action = self.tasks.get(&id).and_then(|task| match task.state {
                AsyncTaskState::WaitingUntil(due) if due == deadline => Some(None),
                AsyncTaskState::WaitingTimeout {
                    target,
                    deadline: due,
                } if due == deadline => Some(Some(target)),
                _ => None,
            });
            match action {
                Some(None) => self.complete_task(id, Ok(Value::Nil))?,
                Some(Some(target)) => {
                    self.remove_waiter(target, id);
                    self.queue_ready_outcome(id, Err(RuntimeError::Timeout))?;
                }
                None => {}
            }
        }
        Ok(())
    }

    fn prune_stale_timers(&mut self) {
        loop {
            let Some(Reverse((deadline, id))) = self.timers.peek().copied() else {
                return;
            };
            let active = self.tasks.get(&id).is_some_and(|task| match task.state {
                AsyncTaskState::WaitingUntil(due) => due == deadline,
                AsyncTaskState::WaitingTimeout { deadline: due, .. } => due == deadline,
                _ => false,
            });
            if active {
                return;
            }
            self.timers.pop();
        }
    }

    fn remove_waiter(&mut self, target: usize, waiter: usize) {
        if let Some(waiters) = self.waiters.get_mut(&target) {
            waiters.retain(|candidate| *candidate != waiter);
        }
    }

    fn remove_waiter_target(&mut self, targets: &WaiterTarget, waiter: usize) {
        match targets {
            WaiterTarget::One(target) => self.remove_waiter(*target, waiter),
            WaiterTarget::Many(targets) => {
                for (target, _) in targets.iter().copied() {
                    self.remove_waiter(target, waiter);
                }
            }
        }
    }

    fn queue_ready_outcome(
        &mut self,
        id: usize,
        outcome: Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        self.ready
            .try_reserve(1)
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "async ready queue",
            })?;
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?;
        task.state = AsyncTaskState::Queued;
        task.target = AsyncTaskTarget::Ready(outcome);
        task.pending_await = None;
        task.pending_select = None;
        self.ready.push_back(id);
        Ok(())
    }

    fn complete_io(
        &mut self,
        completion: AsyncIoCompletion,
        context: &mut RuntimeContext<'_>,
    ) -> Result<(), RuntimeError> {
        let task = completion.task;
        self.pending_io = self
            .pending_io
            .checked_sub(1)
            .ok_or(RuntimeError::InvalidArgument {
                op: "async I/O completion",
            })?;
        let cancellation_requested = self
            .tasks
            .get(&task)
            .is_some_and(|task| task.cancel_requested);
        let reservation = self.file_open_reservations.remove(&task);
        let mut opened_file = None;
        let result = match completion.result {
            Ok(AsyncIoResult::FileOpened { file, mode }) => {
                opened_file = Some((file, mode));
                Ok(AsyncIoResult::Nil)
            }
            result => result,
        };
        let outcome = match (reservation, opened_file) {
            (Some(slot), Some((file, mode))) => {
                if cancellation_requested {
                    drop(file);
                    context.release_file_slot(slot)?;
                    Ok(Value::Nil)
                } else {
                    context.insert_reserved_file(slot, file, mode)
                }
            }
            (Some(slot), None) => {
                context.release_file_slot(slot)?;
                match result {
                    Err(AsyncIoFailure::Cancelled) | Ok(AsyncIoResult::Cancelled) => Ok(Value::Nil),
                    Err(error) => Err(error.into_runtime_error()),
                    Ok(_) => Err(RuntimeError::InvalidNativeCall),
                }
            }
            (None, Some((file, _))) => {
                drop(file);
                Err(RuntimeError::InvalidNativeCall)
            }
            (None, None) => match result {
                Err(AsyncIoFailure::Cancelled) => Ok(Value::Nil),
                Err(error) => Err(error.into_runtime_error()),
                Ok(AsyncIoResult::Cancelled) => Ok(Value::Nil),
                Ok(AsyncIoResult::Nil) => Ok(Value::Nil),
                Ok(AsyncIoResult::Bytes(bytes)) => {
                    HeapRef::try_new(RefCell::new(bytes)).map(Value::Bytes)
                }
                Ok(AsyncIoResult::Text(text)) => Ok(Value::Str(text.into())),
                Ok(AsyncIoResult::UInt(value)) => Ok(Value::UInt(value)),
                Ok(AsyncIoResult::Int(value)) => Ok(Value::Int(value)),
                Ok(AsyncIoResult::TcpStream { stream, .. }) if cancellation_requested => {
                    drop(stream);
                    Ok(Value::Nil)
                }
                Ok(AsyncIoResult::TcpStream { stream, timeout }) => {
                    context.insert_tcp_stream(stream, timeout)
                }
                Ok(AsyncIoResult::TcpListener(listener)) if cancellation_requested => {
                    drop(listener);
                    Ok(Value::Nil)
                }
                Ok(AsyncIoResult::TcpListener(listener)) => context.insert_tcp_listener(listener),
                Ok(AsyncIoResult::Directory(names)) => {
                    let values = names
                        .into_iter()
                        .map(|name| Value::Str(name.into()))
                        .collect();
                    let array = ArrayValue::typed(values, RuntimeType::Str)
                        .map_err(|_| RuntimeError::InvalidNativeCall)?;
                    HeapRef::try_new(array).map(Value::Array)
                }
                Ok(AsyncIoResult::Lines(lines)) => {
                    let mut values = Vec::new();
                    values
                        .try_reserve_exact(lines.len())
                        .map_err(|_| RuntimeError::ResourceLimit { resource: "bytes" })?;
                    values.extend(lines.into_iter().map(|line| Value::Str(line.into())));
                    let array = ArrayValue::typed(values, RuntimeType::Str)
                        .map_err(|_| RuntimeError::InvalidNativeCall)?;
                    HeapRef::try_new(array).map(Value::Array)
                }
                Ok(AsyncIoResult::Metadata {
                    is_file,
                    is_dir,
                    is_readonly,
                    len,
                }) => {
                    let fields = HashMap::from([
                        ("is_file".into(), Value::Bool(is_file)),
                        ("is_dir".into(), Value::Bool(is_dir)),
                        ("is_readonly".into(), Value::Bool(is_readonly)),
                        ("len".into(), Value::Int(len)),
                    ]);
                    HeapRef::try_new(RecordValue::new(fields)).map(Value::Record)
                }
                Ok(AsyncIoResult::Process {
                    exit_code,
                    success,
                    stdout,
                    stderr,
                }) => {
                    let fields = HashMap::from([
                        (
                            "exit_code".into(),
                            exit_code.map_or(Value::Nil, |code| Value::Int(i64::from(code))),
                        ),
                        ("success".into(), Value::Bool(success)),
                        (
                            "stdout".into(),
                            Value::Bytes(HeapRef::try_new(RefCell::new(stdout))?),
                        ),
                        (
                            "stderr".into(),
                            Value::Bytes(HeapRef::try_new(RefCell::new(stderr))?),
                        ),
                    ]);
                    HeapRef::try_new(RecordValue::new(fields)).map(Value::Record)
                }
                Ok(AsyncIoResult::FileOpened { .. }) => Err(RuntimeError::InvalidNativeCall),
            },
        };
        self.complete_task(task, outcome)
    }

    fn join_unobserved_errors(
        &self,
        id: usize,
        outcome: &mut Result<Value, RuntimeError>,
    ) -> Result<(), RuntimeError> {
        let children = &self
            .tasks
            .get(&id)
            .ok_or(RuntimeError::InvalidArgument { op: "async task" })?
            .children;
        let mut last_order = 0;
        for _ in 0..children.len() {
            let mut earliest: Option<(u64, RuntimeError)> = None;
            for child in children {
                let Some(task) = self.tasks.get(child) else {
                    continue;
                };
                if task.observed {
                    continue;
                }
                let AsyncTaskState::Complete {
                    outcome: child_outcome,
                    completed_at,
                } = &task.state
                else {
                    continue;
                };
                if *completed_at <= last_order {
                    continue;
                }
                let error = match child_outcome {
                    Err(error) => Some(error.clone()),
                    Ok(Value::Error(error)) => Some((**error).clone()),
                    Ok(_) => None,
                };
                if let Some(error) = error
                    && earliest
                        .as_ref()
                        .is_none_or(|(order, _)| completed_at < order)
                {
                    earliest = Some((*completed_at, error));
                }
            }
            let Some((order, child_error)) = earliest else {
                break;
            };
            last_order = order;
            match outcome {
                Err(primary) if !child_error.is_recoverable() && primary.is_recoverable() => {
                    *outcome = Err(child_error.with_suppressed(primary.clone()));
                }
                Err(primary) => {
                    *primary = primary.clone().with_suppressed(child_error);
                }
                Ok(Value::Error(primary)) if !child_error.is_recoverable() => {
                    *outcome = Err(child_error.with_suppressed((**primary).clone()));
                }
                Ok(Value::Error(primary)) => {
                    let combined = (**primary).clone().with_suppressed(child_error);
                    *outcome = Ok(Value::Error(Box::new(combined)));
                }
                Ok(_) if !child_error.is_recoverable() => *outcome = Err(child_error),
                Ok(_) => *outcome = Ok(Value::Error(Box::new(child_error))),
            }
        }
        Ok(())
    }

    fn next_completion_order(&mut self) -> Result<u64, RuntimeError> {
        self.completion_order =
            self.completion_order
                .checked_add(1)
                .ok_or(RuntimeError::ResourceLimit {
                    resource: "async completion order",
                })?;
        Ok(self.completion_order)
    }
}

fn async_timeout_millis(value: Option<&Value>) -> Option<u64> {
    match value {
        Some(Value::Int(timeout)) => u64::try_from(*timeout).ok(),
        Some(Value::UInt(timeout)) => Some(*timeout),
        _ => None,
    }
}

fn async_output_limit(value: &Value) -> Option<usize> {
    match value {
        Value::Int(limit) => usize::try_from(*limit).ok(),
        Value::UInt(limit) => usize::try_from(*limit).ok(),
        _ => None,
    }
}

fn copy_async_write_bytes(
    bytes: &HeapRef<RefCell<Vec<u8>>>,
    limit: usize,
    op: &'static str,
) -> Result<Vec<u8>, RuntimeError> {
    let bytes = bytes
        .try_borrow()
        .map_err(|_| RuntimeError::InvalidArgument { op })?;
    if bytes.len() > limit {
        return Err(RuntimeError::ResourceLimit { resource: "bytes" });
    }
    let mut copy = Vec::new();
    copy.try_reserve_exact(bytes.len())
        .map_err(|_| RuntimeError::ResourceLimit { resource: "bytes" })?;
    copy.extend_from_slice(&bytes);
    Ok(copy)
}

fn copy_async_write_text(text: &str, limit: usize) -> Result<Vec<u8>, RuntimeError> {
    if text.len() > limit {
        return Err(RuntimeError::ResourceLimit { resource: "bytes" });
    }
    let mut copy = Vec::new();
    copy.try_reserve_exact(text.len())
        .map_err(|_| RuntimeError::ResourceLimit { resource: "bytes" })?;
    copy.extend_from_slice(text.as_bytes());
    Ok(copy)
}

fn process_call_parts(
    program: Value,
    arguments: Value,
) -> Result<(String, HeapRef<ArrayValue>), Value> {
    let program = match program {
        Value::Error(error) => return Err(Value::Error(error)),
        Value::Str(program) => program.to_string(),
        _ => {
            return Err(Value::Error(Box::new(RuntimeError::TypeMismatch {
                op: "process::run",
            })));
        }
    };
    let arguments = match arguments {
        Value::Error(error) => return Err(Value::Error(error)),
        Value::Array(arguments) => arguments,
        _ => {
            return Err(Value::Error(Box::new(RuntimeError::TypeMismatch {
                op: "process::run",
            })));
        }
    };
    Ok((program, arguments))
}

fn process_argument_strings(arguments: HeapRef<ArrayValue>) -> Result<Vec<String>, Value> {
    let arguments = match arguments.try_borrow() {
        Ok(arguments) => arguments,
        Err(_) => {
            return Err(Value::Error(Box::new(RuntimeError::InvalidArgument {
                op: "process::run arguments",
            })));
        }
    };
    let mut strings = Vec::new();
    if strings.try_reserve_exact(arguments.len()).is_err() {
        return Err(Value::Error(Box::new(RuntimeError::ResourceLimit {
            resource: "bytes",
        })));
    }
    for argument in arguments.iter() {
        match argument {
            Value::Error(error) => return Err(Value::Error(error.clone())),
            Value::Str(argument) => strings.push(argument.to_string()),
            _ => {
                return Err(Value::Error(Box::new(RuntimeError::TypeMismatch {
                    op: "process::run arguments",
                })));
            }
        }
    }
    Ok(strings)
}

fn reserve_cleanup_roots(stack: &mut Vec<Value>, counts: [usize; 4]) -> Result<(), RuntimeError> {
    let failure = || RuntimeError::ResourceLimit {
        resource: "VM cleanup roots",
    };
    let additional = counts
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(failure)?;
    stack.try_reserve_exact(additional).map_err(|_| failure())
}

fn reserve_frame(frames: &mut Vec<CallFrame>, limit: Option<usize>) -> Result<(), RuntimeError> {
    if limit.is_some_and(|limit| frames.len() >= limit) {
        return Err(RuntimeError::ResourceLimit {
            resource: "call stack",
        });
    }
    frames
        .try_reserve(1)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "call stack",
        })
}

fn try_record_key_with_capacity(key: &str, capacity: usize) -> Result<String, RuntimeError> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(capacity.max(key.len()))
        .map_err(|_| RuntimeError::ResourceLimit { resource: "record" })?;
    owned.push_str(key);
    Ok(owned)
}

fn try_record_key(key: &str) -> Result<String, RuntimeError> {
    try_record_key_with_capacity(key, key.len())
}

fn missing_field(name: &str) -> Result<Value, RuntimeError> {
    Err(RuntimeError::MissingField {
        field: try_record_key(name)?,
    })
}

#[inline]
fn check_jump(program: &[Instruction], target: usize) -> Result<usize, RuntimeError> {
    if target < program.len() {
        Ok(target)
    } else {
        Err(RuntimeError::InvalidJump(target))
    }
}

fn check_call(
    artifact: &CompiledProgram,
    target: usize,
    signature: &CallableSignature,
    parameters: usize,
    captures: usize,
) -> Result<usize, RuntimeError> {
    if target == 0 || signature.parameters != parameters || signature.captures != captures {
        return Err(RuntimeError::InvalidCall(target));
    }
    check_jump(&artifact.instructions, target)
}

fn next_callable_boundary(artifact: &CompiledProgram, entry: usize) -> Option<usize> {
    artifact
        .callables
        .range((std::ops::Bound::Excluded(entry), std::ops::Bound::Unbounded))
        .next()
        .map(|(entry, _)| *entry)
}

fn callable_region_start(artifact: &CompiledProgram, pc: usize) -> usize {
    artifact
        .callables
        .range(..=pc)
        .next_back()
        .map_or(0, |(entry, _)| *entry)
}

pub(crate) fn matches_runtime_type(expected: &RuntimeType, value: &Value) -> bool {
    if matches!(value, Value::Error(_)) {
        return true;
    }
    match expected {
        RuntimeType::Int => matches!(value, Value::Int(_)),
        RuntimeType::UInt => matches!(value, Value::UInt(_)),
        RuntimeType::BigInt => matches!(value, Value::BigInt(_)),
        RuntimeType::Float => matches!(value, Value::Float(_)),
        RuntimeType::Bool => matches!(value, Value::Bool(_)),
        RuntimeType::Char => matches!(value, Value::Char(_)),
        RuntimeType::Nil => matches!(value, Value::Nil),
        RuntimeType::Str => matches!(value, Value::Str(_) | Value::StrBuf(_)),
        _ => expected.accepts(&RuntimeType::of(value)),
    }
}

pub(crate) fn call_arguments_match(signature: &CallableSignature, arguments: &[Value]) -> bool {
    let Some(parameter_types) = &signature.parameter_types else {
        return true;
    };
    parameter_types.len() == arguments.len()
        && parameter_types
            .iter()
            .zip(arguments)
            .all(|(expected, value)| {
                expected
                    .as_ref()
                    .is_none_or(|expected| matches_runtime_type(expected, value))
            })
}

fn check_branch(
    program: &[Instruction],
    pc: usize,
    target: usize,
    callable_start: usize,
    callable_end: usize,
) -> Result<usize, RuntimeError> {
    check_jump(program, target)?;
    if !(callable_start..callable_end).contains(&pc)
        || !(callable_start..callable_end).contains(&target)
    {
        return Err(RuntimeError::InvalidJump(target));
    }
    Ok(target)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn numeric_binop(
    a: Value,
    b: Value,
    op_name: &'static str,
    int_op: impl Fn(i64, i64) -> Option<i64>,
    uint_op: impl Fn(u64, u64) -> Option<u64>,
    bigint_op: impl Fn(BigInt, BigInt) -> BigInt,
    numeric_digit_limit: Option<usize>,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, RuntimeError> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            int_op(x, y).map(Value::Int).ok_or(RuntimeError::Overflow)
        }
        (Value::UInt(x), Value::UInt(y)) => {
            uint_op(x, y).map(Value::UInt).ok_or(RuntimeError::Overflow)
        }
        (Value::BigInt(x), Value::BigInt(y)) => {
            check_bigint_limit(&x, numeric_digit_limit)?;
            check_bigint_limit(&y, numeric_digit_limit)?;
            if op_name == "mul"
                && let Some(limit) = numeric_digit_limit
                && minimum_decimal_digits_from_bits(
                    x.bits().saturating_add(y.bits()).saturating_sub(1),
                ) > limit
            {
                return Err(RuntimeError::ResourceLimit {
                    resource: "numeric digits",
                });
            }
            let result = bigint_op(x, y);
            check_bigint_limit(&result, numeric_digit_limit)?;
            Ok(Value::BigInt(result))
        }
        (Value::Int(0), Value::BigInt(y)) if op_name == "sub" => {
            check_bigint_limit(&y, numeric_digit_limit)?;
            let result = -y;
            check_bigint_limit(&result, numeric_digit_limit)?;
            Ok(Value::BigInt(result))
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(float_op(x, y))),
        (Value::Float(x), Value::UInt(y)) => Ok(Value::Float(float_op(x, y as f64))),
        (Value::UInt(x), Value::Float(y)) => Ok(Value::Float(float_op(x as f64, y))),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(float_op(x, y as f64))),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(float_op(x as f64, y))),
        _ => Err(RuntimeError::TypeMismatch { op: op_name }),
    }
}

fn check_bigint_limit(value: &BigInt, limit: Option<usize>) -> Result<(), RuntimeError> {
    if limit.is_some_and(|limit| crate::value::bigint_exceeds_decimal_digits(value, limit)) {
        Err(RuntimeError::ResourceLimit {
            resource: "numeric digits",
        })
    } else {
        Ok(())
    }
}

fn minimum_decimal_digits_from_bits(bits: u64) -> usize {
    usize::try_from(bits.saturating_sub(1).saturating_mul(30_102) / 100_000 + 1)
        .unwrap_or(usize::MAX)
}

#[inline]
fn compare_binop(
    a: Value,
    b: Value,
    op_name: &'static str,
    predicate: impl FnOnce(Option<std::cmp::Ordering>) -> bool,
) -> Result<Value, RuntimeError> {
    let order = match (a, b) {
        (Value::Char(x), Value::Char(y)) => Some(x.cmp(&y)),
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(&y)),
        (Value::UInt(x), Value::UInt(y)) => Some(x.cmp(&y)),
        (Value::BigInt(x), Value::BigInt(y)) => Some(x.cmp(&y)),
        (Value::BigInt(x), Value::Int(y)) => Some(x.cmp(&BigInt::from(y))),
        (Value::Int(x), Value::BigInt(y)) => Some(BigInt::from(x).cmp(&y)),
        (Value::BigInt(x), Value::UInt(y)) => Some(x.cmp(&BigInt::from(y))),
        (Value::UInt(x), Value::BigInt(y)) => Some(BigInt::from(x).cmp(&y)),
        (Value::Int(x), Value::UInt(y)) => Some(crate::value::compare_int_uint(x, y)),
        (Value::UInt(x), Value::Int(y)) => Some(crate::value::compare_int_uint(y, x).reverse()),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(&y),
        (Value::Float(x), Value::UInt(y)) => {
            crate::value::compare_uint_float(y, x).map(std::cmp::Ordering::reverse)
        }
        (Value::UInt(x), Value::Float(y)) => crate::value::compare_uint_float(x, y),
        (Value::Float(x), Value::Int(y)) => {
            crate::value::compare_int_float(y, x).map(std::cmp::Ordering::reverse)
        }
        (Value::Int(x), Value::Float(y)) => crate::value::compare_int_float(x, y),
        (Value::Float(x), Value::BigInt(y)) => {
            crate::value::compare_bigint_float(&y, x).map(std::cmp::Ordering::reverse)
        }
        (Value::BigInt(x), Value::Float(y)) => crate::value::compare_bigint_float(&x, y),
        _ => return Err(RuntimeError::TypeMismatch { op: op_name }),
    };
    Ok(Value::Bool(predicate(order)))
}

#[inline]
fn pop2(stack: &mut Vec<Value>) -> Result<(Value, Value), RuntimeError> {
    let b = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
    let a = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
    Ok((a, b))
}

fn take_stack_tail(stack: &mut Vec<Value>, count: usize) -> Result<Vec<Value>, RuntimeError> {
    let begin = stack
        .len()
        .checked_sub(count)
        .ok_or(RuntimeError::StackUnderflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "VM aggregate",
        })?;
    values.extend(stack.drain(begin..));
    Ok(values)
}

#[inline]
fn pop3(stack: &mut Vec<Value>) -> Result<(Value, Value, Value), RuntimeError> {
    let c = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
    let b = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
    let a = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
    Ok((a, b, c))
}

fn local_address(base: usize, index: usize) -> Result<usize, RuntimeError> {
    // Match the portable slot contract even when an embedder bypasses decode.
    if index >= crate::bytecode::LIMIT {
        return Err(RuntimeError::InvalidLocal(index));
    }
    base.checked_add(index)
        .ok_or(RuntimeError::InvalidLocal(index))
}

#[inline]
fn borrow_conflict(op: &'static str) -> RuntimeError {
    RuntimeError::InvalidArgument { op }
}

fn store_slot(
    slots: &mut Vec<Option<Value>>,
    address: usize,
    value: Value,
    limit: Option<usize>,
) -> Result<(), RuntimeError> {
    if limit.is_some_and(|limit| address >= limit) {
        return Err(RuntimeError::ResourceLimit {
            resource: "VM slots",
        });
    }
    if address >= slots.len() {
        let length = address
            .checked_add(1)
            .ok_or(RuntimeError::InvalidLocal(address))?;
        slots
            .try_reserve(length - slots.len())
            .map_err(|_| RuntimeError::ResourceLimit {
                resource: "VM slots",
            })?;
        slots.resize(length, None);
    }
    slots[address] = Some(value);
    Ok(())
}

#[inline]
fn instantiate_scalar(value: &Value, owner: &Rc<()>) -> Value {
    match value {
        Value::Function(entry) => Value::BoundFunction(BoundFunction {
            entry: *entry,
            owner: owner.clone(),
        }),
        value => value.clone(),
    }
}

#[inline]
fn instantiate_array(template: &ArrayTemplate, owner: &Rc<()>) -> Result<Value, RuntimeError> {
    Ok(match template {
        ArrayTemplate::Scalar(v) => instantiate_scalar(v, owner),
        ArrayTemplate::Array(items) => {
            let mut vec = Vec::new();
            for item in items {
                vec.try_reserve(item.repeat)
                    .map_err(|_| RuntimeError::ResourceLimit { resource: "array" })?;
                for _ in 0..item.repeat {
                    vec.push(instantiate_array(&item.template, owner)?);
                }
            }
            match ArrayValue::try_new(vec) {
                Ok(array) => Value::Array(HeapRef::try_new(array)?),
                Err(error) => Value::Error(Box::new(error)),
            }
        }
    })
}

/// Return the number of managed array objects and expanded template nodes that
/// one `PushArray` will instantiate. Bytecode validation already applies these
/// limits to decoded artifacts; this runtime check also protects callers that
/// construct `CompiledProgram` values directly through the Rust API.
fn array_template_cost(template: &ArrayTemplate) -> Result<(usize, usize), RuntimeError> {
    const MAX_TEMPLATE_NODES: usize = crate::bytecode::LIMIT;
    const MAX_TEMPLATE_DEPTH: usize = 64;

    let mut pending = Vec::new();
    pending
        .try_reserve(1)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "array template",
        })?;
    pending.push((template, 1usize, 0usize));
    let (mut scheduled, mut expanded, mut objects) = (1usize, 0usize, 0usize);

    while let Some((template, multiplier, depth)) = pending.pop() {
        if depth >= MAX_TEMPLATE_DEPTH {
            return Err(RuntimeError::ResourceLimit {
                resource: "array template",
            });
        }
        expanded = expanded
            .checked_add(multiplier)
            .filter(|count| *count <= MAX_TEMPLATE_NODES)
            .ok_or(RuntimeError::ResourceLimit {
                resource: "array template",
            })?;

        let ArrayTemplate::Array(items) = template else {
            continue;
        };
        scheduled = scheduled
            .checked_add(items.len())
            .filter(|count| *count <= MAX_TEMPLATE_NODES)
            .ok_or(RuntimeError::ResourceLimit {
                resource: "array template",
            })?;
        objects = objects
            .checked_add(multiplier)
            .filter(|count| *count <= MAX_TEMPLATE_NODES)
            .ok_or(RuntimeError::ResourceLimit {
                resource: "array template",
            })?;

        for item in items {
            if item.repeat > MAX_TEMPLATE_NODES {
                return Err(RuntimeError::ResourceLimit {
                    resource: "array template",
                });
            }
            if item.repeat == 0 {
                continue;
            }
            let child_multiplier = multiplier
                .checked_mul(item.repeat)
                .filter(|count| *count <= MAX_TEMPLATE_NODES)
                .ok_or(RuntimeError::ResourceLimit {
                    resource: "array template",
                })?;
            pending
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "array template",
                })?;
            pending.push((&item.template, child_multiplier, depth + 1));
        }
    }

    Ok((objects, expanded))
}

fn check_array_template_resources(
    template: &ArrayTemplate,
    context: &RuntimeContext<'_>,
) -> Result<(), RuntimeError> {
    let (objects, _) = array_template_cost(template)?;
    let mut pending = Vec::new();
    pending
        .try_reserve(1)
        .map_err(|_| RuntimeError::ResourceLimit {
            resource: "array template",
        })?;
    pending.push(template);
    while let Some(template) = pending.pop() {
        match template {
            ArrayTemplate::Scalar(Value::BigInt(value)) => {
                check_bigint_limit(value, context.limits.max_numeric_digits)?;
            }
            ArrayTemplate::Scalar(_) => {}
            ArrayTemplate::Array(items) => {
                pending
                    .try_reserve(items.len())
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "array template",
                    })?;
                pending.extend(items.iter().map(|item| &item.template));
            }
        }
    }

    let Some(limit) = context.limits.max_heap_objects else {
        return Ok(());
    };
    if objects > limit {
        return Err(RuntimeError::ResourceLimit {
            resource: "VM heap objects",
        });
    }

    if heap::live_objects().saturating_add(objects) > limit {
        heap::collect(&[])?;
    }
    if heap::live_objects().saturating_add(objects) > limit {
        return Err(RuntimeError::ResourceLimit {
            resource: "VM heap objects",
        });
    }
    Ok(())
}

#[inline]
fn binop(
    stack: &mut Vec<Value>,
    f: impl FnOnce(Value, Value) -> Result<Value, RuntimeError>,
) -> Result<(), RuntimeError> {
    let (a, b) = pop2(stack)?;

    if let Value::Error(_) = a {
        stack.push(a);
        return Ok(());
    }
    if let Value::Error(_) = b {
        stack.push(b);
        return Ok(());
    }

    match f(a, b) {
        Ok(result) => stack.push(result),
        Err(e) if e.is_recoverable() => stack.push(Value::Error(Box::new(e))),
        Err(e) => return Err(e),
    }
    Ok(())
}

#[inline]
fn int_binop(
    stack: &mut Vec<Value>,
    op: impl FnOnce(i64, i64) -> Result<i64, RuntimeError>,
) -> Result<bool, RuntimeError> {
    let len = stack.len();
    let Some((left, right)) = stack
        .get(len.saturating_sub(2)..)
        .and_then(|values| match values {
            [Value::Int(left), Value::Int(right)] => Some((*left, *right)),
            _ => None,
        })
    else {
        return Ok(false);
    };

    stack.truncate(len - 2);
    stack.push(match op(left, right) {
        Ok(value) => Value::Int(value),
        Err(error) if error.is_recoverable() => Value::Error(Box::new(error)),
        Err(error) => return Err(error),
    });
    Ok(true)
}

pub fn run(program: &CompiledProgram) -> Result<(), RuntimeError> {
    run_with(program, &mut std::io::stdout())
}

pub fn run_with<W: std::io::Write + Send>(
    program: &CompiledProgram,
    out: &mut W,
) -> Result<(), RuntimeError> {
    run_with_capabilities(program, out, VmCapabilities::default())
}

pub fn run_with_capabilities<W: std::io::Write + Send>(
    program: &CompiledProgram,
    out: &mut W,
    caps: VmCapabilities,
) -> Result<(), RuntimeError> {
    let mut context = RuntimeContext::new(out, caps);
    run_with_context(program, &mut context)
}

/// Execute with an explicit set of process arguments available to
/// `process::args`, excluding the host executable name.
pub fn run_with_process_args<W: std::io::Write + Send>(
    program: &CompiledProgram,
    out: &mut W,
    caps: VmCapabilities,
    arguments: Vec<std::ffi::OsString>,
) -> Result<(), RuntimeError> {
    let mut context = RuntimeContext::new(out, caps);
    context.set_process_arguments(arguments);
    run_with_context(program, &mut context)
}

/// Execute already-linked imports and finish all context resources on every exit.
/// A primary VM failure takes precedence over a secondary cleanup failure.
/// Callable values produced by this low-level entry point are scoped to one
/// invocation. Use [`crate::sdk::Program`] to retain callables across runs.
pub fn run_with_context(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
) -> Result<(), RuntimeError> {
    run_with_callable_owner(artifact, context, &Rc::new(()))
}

pub(crate) fn run_with_callable_owner(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
    owner: &Rc<()>,
) -> Result<(), RuntimeError> {
    let execution_plan = VmExecutionPlan::new(artifact);
    if execution_plan.has_async {
        let shared_plan = Rc::new(execution_plan);
        run_with_execution_plan(
            artifact,
            context,
            owner,
            &shared_plan,
            Some(Rc::clone(&shared_plan)),
        )
    } else {
        run_with_execution_plan(artifact, context, owner, &execution_plan, None)
    }
}

pub(crate) fn run_with_callable_owner_and_plan(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
    owner: &Rc<()>,
    execution_plan: &Rc<VmExecutionPlan>,
) -> Result<(), RuntimeError> {
    let scheduler_plan = execution_plan.has_async.then(|| Rc::clone(execution_plan));
    run_with_execution_plan(artifact, context, owner, execution_plan, scheduler_plan)
}

fn run_with_execution_plan(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
    owner: &Rc<()>,
    execution_plan: &VmExecutionPlan,
    scheduler_plan: Option<Rc<VmExecutionPlan>>,
) -> Result<(), RuntimeError> {
    let _heap_object_limit = heap::limit_object_allocations(context.limits.max_heap_objects);
    let result = execute(artifact, context, owner, execution_plan, scheduler_plan);
    let cleanup = context.finish();
    match (result, cleanup) {
        (Err(primary), Err(secondary)) if primary.is_recoverable() => {
            Err(primary.with_suppressed(secondary))
        }
        (result, cleanup) => result.and(cleanup),
    }
}

fn execute(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
    owner: &Rc<()>,
    execution_plan: &VmExecutionPlan,
    scheduler_plan: Option<Rc<VmExecutionPlan>>,
) -> Result<(), RuntimeError> {
    if execution_plan.needs_stdin || execution_plan.needs_stdout {
        std::thread::scope(|scope| {
            let (io_completion_sender, io_completions) = mpsc::channel();
            let stdio_pool = AsyncStdioPool::new(
                scope,
                context.shared_input_stream(),
                context.shared_output_stream(),
                io_completion_sender.clone(),
                execution_plan.needs_stdin,
                execution_plan.needs_stdout,
            );
            let mut scheduler = AsyncScheduler::new(
                scheduler_plan,
                Some((io_completion_sender, io_completions)),
                Some(stdio_pool),
            );
            let result =
                execute_top_level_vm(artifact, context, owner, execution_plan, &mut scheduler);
            drop(scheduler);
            result
        })
    } else {
        let mut scheduler = AsyncScheduler::new(scheduler_plan, None, None);
        let result = execute_top_level_vm(artifact, context, owner, execution_plan, &mut scheduler);
        drop(scheduler);
        result
    }
}

fn execute_top_level_vm(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
    owner: &Rc<()>,
    execution_plan: &VmExecutionPlan,
    scheduler: &mut AsyncScheduler,
) -> Result<(), RuntimeError> {
    let mut globals = Vec::new();
    let mut remaining_steps = context.limits.max_steps;
    match execute_vm(
        artifact,
        context,
        owner,
        execution_plan,
        scheduler,
        &mut globals,
        &mut remaining_steps,
        None,
        None,
        None,
        None,
    ) {
        VmExecution::Complete(Ok(value)) => {
            if value.is_heap_managed()
                && let Err(error) = heap::collect(std::slice::from_ref(&value))
            {
                // Preserve the result graph if the collector cannot reserve
                // its tracing storage; recursive destruction could overflow.
                std::mem::forget(value);
                return Err(error);
            }
            drop(value);
            Ok(())
        }
        VmExecution::Complete(Err(error)) => Err(error),
        VmExecution::Suspended(_) => Err(RuntimeError::InvalidArgument {
            op: "top-level async suspension",
        }),
    }
}

fn instruction_may_allocate_heap(instruction: &Instruction, stack: &[Value]) -> bool {
    match instruction {
        Instruction::Add => stack
            .iter()
            .rev()
            .take(2)
            .any(|value| matches!(value, Value::Array(_))),
        Instruction::CallNative(_)
        | Instruction::ProcessRunAsync
        | Instruction::ProcessRunTimeoutAsync
        | Instruction::ProcessRunInputAsync
        | Instruction::ProcessRunInputTimeoutAsync
        | Instruction::ProcessRunInputTimeoutOutputLimitAsync
        | Instruction::AsyncIoRead
        | Instruction::AsyncIoReadExact
        | Instruction::AsyncIoReadAll
        | Instruction::AsyncIoOpen
        | Instruction::AsyncIoWrite
        | Instruction::AsyncIoWriteAll
        | Instruction::AsyncIoFlush
        | Instruction::AsyncIoSeek
        | Instruction::NetworkExchangeAsync
        | Instruction::TcpListenAsync
        | Instruction::TcpAcceptAsync
        | Instruction::TcpConnectAsync
        | Instruction::TcpReadAsync
        | Instruction::TcpReadExactAsync
        | Instruction::TcpReadAllAsync
        | Instruction::TcpWriteAsync
        | Instruction::TcpWriteAllAsync
        | Instruction::TcpShutdownWriteAsync
        | Instruction::PushArray(_)
        | Instruction::Await
        | Instruction::CallValue(_)
        | Instruction::MakeCell
        | Instruction::MakeRecord(_)
        | Instruction::MakeArray(_)
        | Instruction::MakeTuple(_)
        | Instruction::MakeTypedTuple(_, _)
        | Instruction::MakeTypedArray(_, _)
        | Instruction::ArrayEmptyLike
        | Instruction::MakeClosure(_, _) => true,
        _ => false,
    }
}

// Compact loops repeat instruction effects. Invalid bytecode uses the regular
// path so errors still occur only if execution reaches the invalid instruction.
fn prepare_stack_effect_cache(
    program: &[Instruction],
    imports: &[NativeImport],
    has_back_edge: bool,
) -> Option<Vec<(usize, usize)>> {
    if program.len() > MAX_STACK_EFFECT_CACHE_INSTRUCTIONS || !has_back_edge {
        return None;
    }

    let mut cache = Vec::new();
    cache.try_reserve_exact(program.len()).ok()?;
    for instruction in program {
        cache.push(instruction.stack_effect(imports).ok()?);
    }
    Some(cache)
}

#[allow(clippy::too_many_arguments)]
fn execute_vm(
    artifact: &CompiledProgram,
    context: &mut RuntimeContext<'_>,
    owner: &Rc<()>,
    execution_plan: &VmExecutionPlan,
    scheduler: &mut AsyncScheduler,
    globals: &mut Vec<Option<Value>>,
    remaining_steps: &mut Option<u64>,
    start: Option<(usize, Vec<Value>)>,
    async_task: Option<usize>,
    continuation: Option<VmContinuation>,
    resume_outcome: Option<Result<Value, RuntimeError>>,
) -> VmExecution {
    context.set_source_location(None);
    let program = &artifact.instructions;
    let start_entry = start.as_ref().map(|(entry, _)| *entry);
    let (
        mut pc,
        mut stack,
        mut locals,
        mut bp,
        mut call_stack,
        mut operand_base,
        mut captures,
        mut error_value_pc,
        mut halt_error_pc,
        mut propagation_pending,
        mut callable_start,
        mut callable_end,
    ) = match continuation {
        Some(state) if start.is_none() && async_task.is_some() => (
            state.pc,
            state.stack,
            state.locals,
            state.bp,
            state.call_stack,
            state.operand_base,
            state.captures,
            state.error_value_pc,
            state.halt_error_pc,
            state.propagation_pending,
            state.callable_start,
            state.callable_end,
        ),
        Some(_) => {
            return VmExecution::Complete(Err(RuntimeError::InvalidArgument {
                op: "async continuation",
            }));
        }
        None => {
            let (pc, stack) = start.unwrap_or_else(|| (0, Vec::new()));
            (
                pc,
                stack,
                Vec::new(),
                0,
                Vec::new(),
                0,
                CaptureValues::default(),
                None,
                None,
                false,
                start_entry.unwrap_or_else(|| execution_plan.callable_region_start(artifact, pc)),
                execution_plan.next_callable_boundary(artifact, pc),
            )
        }
    };
    let mut suspended = false;
    let result = (|| -> Result<Value, RuntimeError> {
        if let Some(outcome) = resume_outcome {
            let value = await_outcome(outcome)?;
            if matches!(&value, Value::Error(_)) {
                error_value_pc = pc.checked_sub(1);
            }
            stack
                .try_reserve(1)
                .map_err(|_| RuntimeError::ResourceLimit {
                    resource: "operand stack",
                })?;
            stack.push(value);
        }
        if let Some(entry) = start_entry {
            let signature = execution_plan
                .callable_signature(artifact, entry)
                .ok_or(RuntimeError::InvalidCall(entry))?;
            if !signature.asynchronous
                || signature.captures != 0
                || !call_arguments_match(signature, &stack)
            {
                return Err(RuntimeError::InvalidCall(entry));
            }
            check_jump(program, entry)?;
        }
        if context
            .limits
            .max_operand_values
            .is_some_and(|limit| stack.len() > limit)
        {
            return Err(RuntimeError::ResourceLimit {
                resource: "operand stack",
            });
        }
        while pc < program.len() {
            scheduler.reap_abandoned_tasks();
            let heap_limit = context.limits.max_heap_objects;
            let collection_due = heap::collection_due();
            let heap_pressure = heap_limit.is_some_and(|limit| {
                heap::live_objects() >= limit && instruction_may_allocate_heap(&program[pc], &stack)
            });
            if collection_due || heap_pressure {
                let reaped = scheduler.reap_unreachable_cold_tasks()?;
                if collection_due || reaped {
                    heap::collect(&[])?;
                }
            }
            if heap_limit.is_some_and(|limit| heap::live_objects() > limit) {
                return Err(RuntimeError::ResourceLimit {
                    resource: "VM heap objects",
                });
            }
            if let Some(steps) = remaining_steps {
                *steps = steps.checked_sub(1).ok_or(RuntimeError::ResourceLimit {
                    resource: "instruction steps",
                })?;
            }
            let (required, produced) = if let Some(cache) = &execution_plan.stack_effect_cache {
                cache[pc]
            } else {
                program[pc].stack_effect(&artifact.imports)?
            };
            let available = stack
                .len()
                .checked_sub(operand_base)
                .ok_or(RuntimeError::StackUnderflow)?;
            if available < required {
                return Err(RuntimeError::StackUnderflow);
            }
            if context
                .limits
                .max_operand_values
                .is_some_and(|limit| stack.len() - required + produced > limit)
            {
                return Err(RuntimeError::ResourceLimit {
                    resource: "operand stack",
                });
            }
            let stack_growth = produced.saturating_sub(required);
            if stack_growth != 0 {
                stack
                    .try_reserve(stack_growth)
                    .map_err(|_| RuntimeError::ResourceLimit {
                        resource: "operand stack",
                    })?;
            }
            match &program[pc] {
                Instruction::Push(value) => {
                    if let Value::BigInt(value) = value {
                        check_bigint_limit(value, context.limits.max_numeric_digits)?;
                    }
                    stack.push(instantiate_scalar(value, owner));
                }
                Instruction::PushArray(template) => {
                    check_array_template_resources(template, context)?;
                    stack.push(instantiate_array(template, owner)?);
                }
                Instruction::Halt => {
                    if let Some(Value::Error(error)) = stack[operand_base..].last() {
                        halt_error_pc = Some(error_value_pc.unwrap_or(pc));
                        return Err((**error).clone());
                    }
                    return Ok(stack[operand_base..].last().cloned().unwrap_or(Value::Nil));
                }
                Instruction::Add => {
                    if !int_binop(&mut stack, |x, y| {
                        x.checked_add(y).ok_or(RuntimeError::Overflow)
                    })? {
                        binop(&mut stack, |a, b| {
                            if let (Value::Str(x), Value::Str(y)) = (&a, &b) {
                                let length = x
                                    .len()
                                    .checked_add(y.len())
                                    .filter(|length| *length <= context.limits.buffer_limit())
                                    .ok_or(RuntimeError::ResourceLimit { resource: "str" })?;
                                let mut s = String::new();
                                s.try_reserve_exact(length)
                                    .map_err(|_| RuntimeError::ResourceLimit { resource: "str" })?;
                                s.push_str(x);
                                s.push_str(y);
                                return Ok(Value::Str(Rc::from(s)));
                            }
                            if let Value::Array(arr_a) = &a {
                                let left = arr_a
                                    .try_borrow()
                                    .map_err(|_| borrow_conflict("array_concat"))?;
                                let mut new_vec = Vec::new();
                                new_vec.try_reserve_exact(left.len()).map_err(|_| {
                                    RuntimeError::ResourceLimit { resource: "array" }
                                })?;
                                new_vec.extend(left.iter().cloned());
                                drop(left);
                                let mut expected = arr_a.element_type();
                                if let Value::Array(arr_b) = &b {
                                    let right_type = arr_b.element_type();
                                    if expected
                                        .as_ref()
                                        .zip(right_type.as_ref())
                                        .is_some_and(|(left, right)| left != right)
                                    {
                                        return Err(RuntimeError::TypeMismatch {
                                            op: "array_concat",
                                        });
                                    }
                                    let right = arr_b
                                        .try_borrow()
                                        .map_err(|_| borrow_conflict("array_concat"))?;
                                    new_vec.try_reserve(right.len()).map_err(|_| {
                                        RuntimeError::ResourceLimit { resource: "array" }
                                    })?;
                                    new_vec.extend(right.iter().cloned());
                                    expected = expected.or(right_type);
                                } else {
                                    new_vec.try_reserve(1).map_err(|_| {
                                        RuntimeError::ResourceLimit { resource: "array" }
                                    })?;
                                    new_vec.push(b.clone());
                                }
                                let result = if let Some(expected) = expected {
                                    ArrayValue::try_typed(new_vec, expected)
                                } else {
                                    ArrayValue::try_new(new_vec)
                                };
                                return match result {
                                    Ok(array) => Ok(Value::Array(HeapRef::try_new(array)?)),
                                    Err(RuntimeError::TypeMismatch { .. }) => {
                                        Err(RuntimeError::TypeMismatch { op: "array_concat" })
                                    }
                                    Err(error) => Err(error),
                                };
                            }
                            if let Value::Array(arr_b) = &b {
                                let right = arr_b
                                    .try_borrow()
                                    .map_err(|_| borrow_conflict("array_concat"))?;
                                let capacity = right
                                    .len()
                                    .checked_add(1)
                                    .ok_or(RuntimeError::ResourceLimit { resource: "array" })?;
                                let mut new_vec: Vec<Value> = Vec::new();
                                new_vec.try_reserve_exact(capacity).map_err(|_| {
                                    RuntimeError::ResourceLimit { resource: "array" }
                                })?;
                                new_vec.push(a.clone());
                                new_vec.extend(right.iter().cloned());
                                drop(right);
                                let result = if let Some(expected) = arr_b.element_type() {
                                    ArrayValue::try_typed(new_vec, expected)
                                } else {
                                    ArrayValue::try_new(new_vec)
                                };
                                return match result {
                                    Ok(array) => Ok(Value::Array(HeapRef::try_new(array)?)),
                                    Err(RuntimeError::TypeMismatch { .. }) => {
                                        Err(RuntimeError::TypeMismatch { op: "array_concat" })
                                    }
                                    Err(error) => Err(error),
                                };
                            }
                            numeric_binop(
                                a,
                                b,
                                "add",
                                i64::checked_add,
                                u64::checked_add,
                                |x, y| x + y,
                                context.limits.max_numeric_digits,
                                |x, y| x + y,
                            )
                        })?;
                    }
                }
                Instruction::ArrayPush => {
                    // `arr << x`: left operand must be the array, mutated in place so
                    // every alias of it (another variable, a captured closure, a field
                    // on some other value) observes the push — unlike the old
                    // "reuse Add" approach, which built a brand-new array and left the
                    // original untouched, silently doing nothing for a bare `arr << x;`.
                    binop(&mut stack, |a, b| match a {
                        Value::Array(arr) => {
                            arr.try_push(b, false)?;
                            Ok(Value::Array(arr))
                        }
                        _ => Err(RuntimeError::TypeMismatch { op: "arrpush" }),
                    })?;
                }
                Instruction::ArrayPrepend => {
                    // `x >> arr`: right operand must be the array (see hlparser's
                    // `parse_shift` / the `prepend_operator_parses` test — Prepend's
                    // AST is `Binary(Prepend, elem, arr)`), mutated in place for the
                    // same aliasing reason as ArrayPush above.
                    binop(&mut stack, |a, b| match b {
                        Value::Array(arr) => {
                            arr.try_push(a, true)?;
                            Ok(Value::Array(arr))
                        }
                        _ => Err(RuntimeError::TypeMismatch { op: "arrprepend" }),
                    })?;
                }
                Instruction::Sub => {
                    if !int_binop(&mut stack, |x, y| {
                        x.checked_sub(y).ok_or(RuntimeError::Overflow)
                    })? {
                        binop(&mut stack, |a, b| {
                            numeric_binop(
                                a,
                                b,
                                "sub",
                                i64::checked_sub,
                                u64::checked_sub,
                                |x, y| x - y,
                                context.limits.max_numeric_digits,
                                |x, y| x - y,
                            )
                        })?;
                    }
                }
                Instruction::Mul => {
                    if !int_binop(&mut stack, |x, y| {
                        x.checked_mul(y).ok_or(RuntimeError::Overflow)
                    })? {
                        binop(&mut stack, |a, b| {
                            numeric_binop(
                                a,
                                b,
                                "mul",
                                i64::checked_mul,
                                u64::checked_mul,
                                |x, y| x * y,
                                context.limits.max_numeric_digits,
                                |x, y| x * y,
                            )
                        })?;
                    }
                }
                Instruction::Div => {
                    if !int_binop(&mut stack, |x, y| {
                        if y == 0 {
                            Err(RuntimeError::DivisionByZero)
                        } else {
                            x.checked_div(y).ok_or(RuntimeError::Overflow)
                        }
                    })? {
                        binop(&mut stack, |a, b| match (a, b) {
                            (Value::Int(_), Value::Int(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::Int(x), Value::Int(y)) => x
                                .checked_div(y)
                                .map(Value::Int)
                                .ok_or(RuntimeError::Overflow),
                            (Value::UInt(_), Value::UInt(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::UInt(x), Value::UInt(y)) => x
                                .checked_div(y)
                                .map(Value::UInt)
                                .ok_or(RuntimeError::Overflow),
                            (Value::BigInt(_), Value::BigInt(y)) if y == BigInt::from(0) => {
                                Err(RuntimeError::DivisionByZero)
                            }
                            (Value::BigInt(x), Value::BigInt(y)) => Ok(Value::BigInt(x / y)),
                            (Value::Float(_), Value::Float(0.0)) => {
                                Err(RuntimeError::DivisionByZero)
                            }
                            (Value::Float(_), Value::Int(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::Float(_), Value::UInt(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::Int(_), Value::Float(0.0)) => Err(RuntimeError::DivisionByZero),
                            (Value::UInt(_), Value::Float(0.0)) => {
                                Err(RuntimeError::DivisionByZero)
                            }
                            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x / y)),
                            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x / (y as f64))),
                            (Value::Int(x), Value::Float(y)) => Ok(Value::Float((x as f64) / y)),
                            (Value::Float(x), Value::UInt(y)) => Ok(Value::Float(x / (y as f64))),
                            (Value::UInt(x), Value::Float(y)) => Ok(Value::Float((x as f64) / y)),
                            _ => Err(RuntimeError::TypeMismatch { op: "div" }),
                        })?;
                    }
                }
                Instruction::Mod => {
                    if !int_binop(&mut stack, |x, y| {
                        if y == 0 {
                            Err(RuntimeError::DivisionByZero)
                        } else {
                            x.checked_rem(y).ok_or(RuntimeError::Overflow)
                        }
                    })? {
                        binop(&mut stack, |a, b| match (a, b) {
                            (Value::Int(_), Value::Int(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::Int(x), Value::Int(y)) => x
                                .checked_rem(y)
                                .map(Value::Int)
                                .ok_or(RuntimeError::Overflow),
                            (Value::UInt(_), Value::UInt(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::UInt(x), Value::UInt(y)) => x
                                .checked_rem(y)
                                .map(Value::UInt)
                                .ok_or(RuntimeError::Overflow),
                            (Value::BigInt(_), Value::BigInt(y)) if y == BigInt::from(0) => {
                                Err(RuntimeError::DivisionByZero)
                            }
                            (Value::BigInt(x), Value::BigInt(y)) => Ok(Value::BigInt(x % y)),
                            (Value::Float(_), Value::Float(0.0)) => {
                                Err(RuntimeError::DivisionByZero)
                            }
                            (Value::Float(_), Value::Int(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::Float(_), Value::UInt(0)) => Err(RuntimeError::DivisionByZero),
                            (Value::Int(_), Value::Float(0.0)) => Err(RuntimeError::DivisionByZero),
                            (Value::UInt(_), Value::Float(0.0)) => {
                                Err(RuntimeError::DivisionByZero)
                            }
                            (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x % y)),
                            (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x % (y as f64))),
                            (Value::Int(x), Value::Float(y)) => Ok(Value::Float((x as f64) % y)),
                            (Value::Float(x), Value::UInt(y)) => Ok(Value::Float(x % (y as f64))),
                            (Value::UInt(x), Value::Float(y)) => Ok(Value::Float((x as f64) % y)),
                            _ => Err(RuntimeError::TypeMismatch { op: "mod" }),
                        })?;
                    }
                }
                Instruction::Eq => {
                    binop(&mut stack, |a, b| match (&a, &b) {
                        (Value::Nil, Value::Nil) => Ok(Value::Bool(true)),
                        // Nullable values can be compared to nil independently
                        // of their non-nil normal type. Nullability does not
                        // imply an omitted call argument or a default value.
                        (Value::Nil, _) | (_, Value::Nil) => Ok(Value::Bool(false)),
                        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x == y)),
                        (Value::Char(x), Value::Char(y)) => Ok(Value::Bool(x == y)),
                        (Value::Str(x), Value::Str(y)) | (Value::TypeTag(x), Value::TypeTag(y)) => {
                            Ok(Value::Bool(x == y))
                        }
                        (
                            Value::Int(_) | Value::UInt(_) | Value::BigInt(_) | Value::Float(_),
                            Value::Int(_) | Value::UInt(_) | Value::BigInt(_) | Value::Float(_),
                        ) => compare_binop(a, b, "eq", |order| {
                            order == Some(std::cmp::Ordering::Equal)
                        }),
                        // Different normal types compare unequal. Equality of
                        // arrays, records and other unsupported same-kind values
                        // produces a recoverable TypeMismatch.
                        _ if std::mem::discriminant(&a) != std::mem::discriminant(&b) => {
                            Ok(Value::Bool(false))
                        }
                        _ => compare_binop(a, b, "eq", |order| {
                            order == Some(std::cmp::Ordering::Equal)
                        }),
                    })?;
                }
                Instruction::Lt => {
                    binop(&mut stack, |a, b| {
                        compare_binop(a, b, "lt", |order| {
                            order.is_some_and(std::cmp::Ordering::is_lt)
                        })
                    })?;
                }
                Instruction::Gt => {
                    binop(&mut stack, |a, b| {
                        compare_binop(a, b, "gt", |order| {
                            order.is_some_and(std::cmp::Ordering::is_gt)
                        })
                    })?;
                }
                Instruction::Ne => {
                    binop(&mut stack, |a, b| match (&a, &b) {
                        (Value::Nil, Value::Nil) => Ok(Value::Bool(false)),
                        (Value::Nil, _) | (_, Value::Nil) => Ok(Value::Bool(true)),
                        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x != y)),
                        (Value::Char(x), Value::Char(y)) => Ok(Value::Bool(x != y)),
                        (Value::Str(x), Value::Str(y)) | (Value::TypeTag(x), Value::TypeTag(y)) => {
                            Ok(Value::Bool(x != y))
                        }
                        (
                            Value::Int(_) | Value::UInt(_) | Value::BigInt(_) | Value::Float(_),
                            Value::Int(_) | Value::UInt(_) | Value::BigInt(_) | Value::Float(_),
                        ) => compare_binop(a, b, "ne", |order| {
                            order != Some(std::cmp::Ordering::Equal)
                        }),
                        // Mirror of `Eq`'s mismatched-type case above — see its
                        // comment.
                        _ if std::mem::discriminant(&a) != std::mem::discriminant(&b) => {
                            Ok(Value::Bool(true))
                        }
                        _ => compare_binop(a, b, "ne", |order| {
                            order != Some(std::cmp::Ordering::Equal)
                        }),
                    })?;
                }
                Instruction::Le => {
                    binop(&mut stack, |a, b| {
                        compare_binop(a, b, "le", |order| {
                            order.is_some_and(std::cmp::Ordering::is_le)
                        })
                    })?;
                }
                Instruction::Ge => {
                    binop(&mut stack, |a, b| {
                        compare_binop(a, b, "ge", |order| {
                            order.is_some_and(std::cmp::Ordering::is_ge)
                        })
                    })?;
                }
                Instruction::CallNative(index) => {
                    let import = artifact
                        .imports
                        .get(*index)
                        .ok_or(RuntimeError::InvalidNativeCall)?;
                    let count = import.parameters().len();
                    let base = stack
                        .len()
                        .checked_sub(count)
                        .filter(|base| *base >= operand_base)
                        .ok_or(RuntimeError::StackUnderflow)?;
                    let arguments = &stack[base..];
                    let result =
                        if import.is_async() {
                            let propagated = arguments.iter().zip(import.parameters()).find_map(
                                |(value, ty)| {
                                    (*ty != crate::runtime::NativeType::Any)
                                        .then_some(value)
                                        .filter(|value| matches!(value, Value::Error(_)))
                                },
                            );
                            if let Some(error) = propagated {
                                Ok(error.clone())
                            } else {
                                import.validate_arguments(arguments).and_then(|()| {
                                    let mut captured = Vec::new();
                                    captured.try_reserve_exact(arguments.len()).map_err(|_| {
                                        RuntimeError::ResourceLimit {
                                            resource: "async plugin arguments",
                                        }
                                    })?;
                                    captured.extend(arguments.iter().cloned());
                                    scheduler
                                        .create_target(AsyncTaskTarget::Native {
                                            import: import.clone(),
                                            arguments: captured,
                                        })
                                        .map(Value::Task)
                                })
                            }
                        } else {
                            import.call_from_vm(context, arguments)
                        };
                    stack.truncate(base);
                    match result {
                        Ok(value) => stack.push(value),
                        Err(error) if error.is_recoverable() => {
                            stack.push(Value::Error(Box::new(error)))
                        }
                        Err(error) => return Err(error),
                    }
                }
                Instruction::Store(index) => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    store_slot(
                        &mut locals,
                        local_address(bp, *index)?,
                        value,
                        context
                            .limits
                            .max_slots
                            .map(|limit| limit.saturating_sub(globals.len())),
                    )?;
                }
                Instruction::Load(index) => {
                    let abs = local_address(bp, *index)?;
                    let value = locals
                        .get(abs)
                        .cloned()
                        .flatten()
                        .ok_or(RuntimeError::InvalidLocal(*index))?;
                    stack.push(value);
                }
                Instruction::StoreGlobal(index) => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    store_slot(
                        globals,
                        local_address(0, *index)?,
                        value,
                        context
                            .limits
                            .max_slots
                            .map(|limit| limit.saturating_sub(locals.len())),
                    )?;
                }
                Instruction::LoadGlobal(index) => {
                    local_address(0, *index)?;
                    let value = globals
                        .get(*index)
                        .cloned()
                        .flatten()
                        .ok_or(RuntimeError::InvalidLocal(*index))?;
                    stack.push(value);
                }
                Instruction::Jump(index) => {
                    let target = check_branch(program, pc, *index, callable_start, callable_end)?;
                    if context.jit_enabled
                        && target <= pc
                        && let Some(exit) = execution_plan.try_jit_osr(
                            artifact,
                            callable_start,
                            target,
                            &mut pc,
                            &mut locals,
                            bp,
                            &mut stack,
                            operand_base,
                            &captures,
                            &mut propagation_pending,
                            context,
                        )
                    {
                        pc = match exit {
                            JitExit::Completed => pc,
                            JitExit::ResumeAt { pc } | JitExit::FaultAt { pc } => pc,
                        };
                        continue;
                    }
                    pc = target;
                    continue;
                }
                Instruction::JumpIfFalse(index) => {
                    let condition = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    match condition {
                        Value::Bool(false) => {
                            let target =
                                check_branch(program, pc, *index, callable_start, callable_end)?;
                            if context.jit_enabled
                                && target <= pc
                                && let Some(exit) = execution_plan.try_jit_osr(
                                    artifact,
                                    callable_start,
                                    target,
                                    &mut pc,
                                    &mut locals,
                                    bp,
                                    &mut stack,
                                    operand_base,
                                    &captures,
                                    &mut propagation_pending,
                                    context,
                                )
                            {
                                pc = match exit {
                                    JitExit::Completed => pc,
                                    JitExit::ResumeAt { pc } | JitExit::FaultAt { pc } => pc,
                                };
                                continue;
                            }
                            pc = target;
                            continue;
                        }
                        Value::Bool(true) => {}
                        _ => {
                            return Err(RuntimeError::TypeMismatch {
                                op: "jump_if_false",
                            });
                        }
                    }
                }
                Instruction::JumpIfTrue(index) => {
                    let condition = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    match condition {
                        Value::Bool(true) => {
                            let target =
                                check_branch(program, pc, *index, callable_start, callable_end)?;
                            if context.jit_enabled
                                && target <= pc
                                && let Some(exit) = execution_plan.try_jit_osr(
                                    artifact,
                                    callable_start,
                                    target,
                                    &mut pc,
                                    &mut locals,
                                    bp,
                                    &mut stack,
                                    operand_base,
                                    &captures,
                                    &mut propagation_pending,
                                    context,
                                )
                            {
                                pc = match exit {
                                    JitExit::Completed => pc,
                                    JitExit::ResumeAt { pc } | JitExit::FaultAt { pc } => pc,
                                };
                                continue;
                            }
                            pc = target;
                            continue;
                        }
                        Value::Bool(false) => {}
                        _ => {
                            return Err(RuntimeError::TypeMismatch { op: "jump_if_true" });
                        }
                    }
                }
                Instruction::JumpIfNoPropagation(index) => {
                    if !std::mem::take(&mut propagation_pending) {
                        pc = check_branch(program, pc, *index, callable_start, callable_end)?;
                        continue;
                    }
                }
                Instruction::IsType(ty) => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let is_match = ty.matches(&value);
                    stack.push(Value::Bool(is_match));
                }
                Instruction::ErrorKind => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let name = match &value {
                        Value::Error(err) => err.kind(),
                        _ => "ok",
                    };
                    stack.push(Value::Str(Rc::from(name)));
                }
                Instruction::Len => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = match value {
                        Value::Error(_) => value,
                        Value::Str(ref s) => Value::Int(s.chars().count() as i64),
                        Value::Array(ref arr) => match arr.try_borrow() {
                            Ok(items) => Value::Int(items.len() as i64),
                            Err(_) => Value::Error(Box::new(borrow_conflict("len"))),
                        },
                        _ => Value::Error(Box::new(RuntimeError::TypeMismatch { op: "len" })),
                    };
                    stack.push(result);
                }
                Instruction::Exclusive => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = match value {
                        Value::Error(_) => value,
                        Value::Array(arr) => Value::Bool(arr.strong_count() == 1),
                        Value::Record(fields) => Value::Bool(fields.strong_count() == 1),
                        Value::Cell(cell) => Value::Bool(cell.strong_count() == 1),
                        Value::StrBuf(buf) => Value::Bool(buf.strong_count() == 1),
                        Value::Str(s) => Value::Bool(Rc::strong_count(&s) == 1),
                        Value::Closure(_, captures) => Value::Bool(captures.strong_count() == 1),
                        _ => Value::Bool(true),
                    };
                    stack.push(result);
                }
                Instruction::CallAsync(index, arity) => {
                    let signature = execution_plan
                        .callable_signature(artifact, *index)
                        .ok_or(RuntimeError::InvalidCall(*index))?;
                    check_call(artifact, *index, signature, *arity, 0)?;
                    if !signature.asynchronous {
                        return Err(RuntimeError::InvalidCall(*index));
                    }
                    check_branch(program, pc, pc + 1, callable_start, callable_end)?;
                    let next_base = stack
                        .len()
                        .checked_sub(*arity)
                        .filter(|base| *base >= operand_base)
                        .ok_or(RuntimeError::StackUnderflow)?;
                    if !call_arguments_match(signature, &stack[next_base..]) {
                        stack.truncate(next_base);
                        stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                            op: "async call",
                        })));
                    } else {
                        let arguments = take_stack_tail(&mut stack, *arity)?;
                        match scheduler.create_task(*index, arguments) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::Await => {
                    let allowed = execution_plan
                        .callable_signature(artifact, callable_start)
                        .is_none_or(|signature| signature.asynchronous);
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    if !allowed {
                        stack.push(Value::Error(Box::new(RuntimeError::InvalidArgument {
                            op: "await outside async function",
                        })));
                    } else {
                        match value {
                            Value::Error(_) => stack.push(value),
                            Value::Task(task) => {
                                let outcome = if let Some(current) = async_task {
                                    match scheduler.suspend_on(current, task) {
                                        Ok(Some(value)) => Some(Ok(value)),
                                        Ok(None) => None,
                                        Err(error) => Some(Err(error)),
                                    }
                                } else {
                                    Some(scheduler.await_task(
                                        task,
                                        artifact,
                                        context,
                                        owner,
                                        globals,
                                        remaining_steps,
                                    ))
                                };
                                if let Some(outcome) = outcome {
                                    match outcome {
                                        Ok(result) => stack.push(result),
                                        Err(error) if error.is_recoverable() => {
                                            stack.push(Value::Error(Box::new(error)))
                                        }
                                        Err(error) => return Err(error),
                                    }
                                } else {
                                    if callable_end < program.len() && pc + 1 == callable_end {
                                        return Err(RuntimeError::InvalidJump(pc + 1));
                                    }
                                    pc += 1;
                                    suspended = true;
                                    break;
                                }
                            }
                            _ => stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                op: "await",
                            }))),
                        }
                    }
                }
                Instruction::Spawn => {
                    let allowed = execution_plan
                        .callable_signature(artifact, callable_start)
                        .is_some_and(|signature| signature.asynchronous);
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    if !allowed {
                        stack.push(Value::Error(Box::new(RuntimeError::InvalidArgument {
                            op: "spawn outside async function",
                        })));
                    } else {
                        match value {
                            Value::Error(_) => stack.push(value),
                            Value::Task(task) => {
                                match scheduler.task_id(&task).and_then(|id| scheduler.spawn(id)) {
                                    Ok(()) => stack.push(Value::Task(task)),
                                    Err(error) if error.is_recoverable() => {
                                        stack.push(Value::Error(Box::new(error)))
                                    }
                                    Err(error) => return Err(error),
                                }
                            }
                            _ => stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                op: "spawn",
                            }))),
                        }
                    }
                }
                Instruction::Cancel => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    match value {
                        Value::Error(_) => stack.push(value),
                        Value::Task(task) => match scheduler
                            .task_id(&task)
                            .and_then(|id| scheduler.cancel_task(id))
                        {
                            Ok(cancelled) => stack.push(Value::Bool(cancelled)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        },
                        _ => stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                            op: "cancel",
                        }))),
                    }
                }
                Instruction::CheckCancel => {
                    let task = async_task.ok_or(RuntimeError::InvalidArgument {
                        op: "cancellation check outside async task",
                    })?;
                    if scheduler.take_cancel_request(task)? {
                        stack.push(Value::Error(Box::new(RuntimeError::Cancelled)));
                    } else {
                        stack.push(Value::Nil);
                    }
                }
                Instruction::SleepAsync => {
                    let argument = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    if matches!(argument, Value::Error(_)) {
                        stack.push(argument);
                    } else {
                        match scheduler.create_target(AsyncTaskTarget::Sleep(argument)) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::YieldNow => match scheduler.create_target(AsyncTaskTarget::YieldNow) {
                    Ok(task) => stack.push(Value::Task(task)),
                    Err(error) if error.is_recoverable() => {
                        stack.push(Value::Error(Box::new(error)))
                    }
                    Err(error) => return Err(error),
                },
                Instruction::TimeoutAsync => {
                    let arguments = take_stack_tail(&mut stack, 2)?;
                    if let Some(Value::Error(error)) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(Value::Error(error.clone()));
                    } else {
                        let mut arguments = arguments.into_iter();
                        let task = arguments.next().ok_or(RuntimeError::StackUnderflow)?;
                        let seconds = arguments.next().ok_or(RuntimeError::StackUnderflow)?;
                        match scheduler.create_target(AsyncTaskTarget::Timeout { task, seconds }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::Select2 => {
                    let arguments = take_stack_tail(&mut stack, 2)?;
                    if let Some(Value::Error(error)) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(Value::Error(error.clone()));
                    } else {
                        let mut arguments = arguments.into_iter();
                        let first = arguments.next().ok_or(RuntimeError::StackUnderflow)?;
                        let second = arguments.next().ok_or(RuntimeError::StackUnderflow)?;
                        match scheduler.create_target(AsyncTaskTarget::Select2 { first, second }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::Select => {
                    let tasks = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    if let Value::Error(error) = tasks {
                        stack.push(Value::Error(error));
                    } else {
                        match scheduler.create_target(AsyncTaskTarget::Select(tasks)) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::ReadBytesAsync => {
                    let arguments = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    if matches!(arguments, Value::Error(_)) {
                        stack.push(arguments);
                    } else {
                        match scheduler.create_target(AsyncTaskTarget::Filesystem {
                            operation: AsyncFsOperation::ReadBytes,
                            arguments: vec![arguments],
                        }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::FsMetadataAsync
                | Instruction::FsReadDirAsync
                | Instruction::FsWriteBytesAsync
                | Instruction::FsCreateDirAsync
                | Instruction::FsCreateDirAllAsync
                | Instruction::FsRemoveFileAsync
                | Instruction::FsRemoveDirAsync
                | Instruction::FsReadTextAsync
                | Instruction::FsWriteTextAsync
                | Instruction::FsReadLinesAsync
                | Instruction::FsAppendBytesAsync
                | Instruction::FsAppendTextAsync
                | Instruction::FsCopyAsync
                | Instruction::FsRenameAsync => {
                    let operation = match &program[pc] {
                        Instruction::FsMetadataAsync => AsyncFsOperation::Metadata,
                        Instruction::FsReadDirAsync => AsyncFsOperation::ReadDir,
                        Instruction::FsWriteBytesAsync => AsyncFsOperation::WriteBytes,
                        Instruction::FsCreateDirAsync => AsyncFsOperation::CreateDir,
                        Instruction::FsCreateDirAllAsync => AsyncFsOperation::CreateDirAll,
                        Instruction::FsRemoveFileAsync => AsyncFsOperation::RemoveFile,
                        Instruction::FsRemoveDirAsync => AsyncFsOperation::RemoveDir,
                        Instruction::FsReadTextAsync => AsyncFsOperation::ReadText,
                        Instruction::FsWriteTextAsync => AsyncFsOperation::WriteText,
                        Instruction::FsReadLinesAsync => AsyncFsOperation::ReadLines,
                        Instruction::FsAppendBytesAsync => AsyncFsOperation::AppendBytes,
                        Instruction::FsAppendTextAsync => AsyncFsOperation::AppendText,
                        Instruction::FsCopyAsync => AsyncFsOperation::Copy,
                        Instruction::FsRenameAsync => AsyncFsOperation::Rename,
                        _ => unreachable!(),
                    };
                    let arguments = take_stack_tail(&mut stack, operation.arity())?;
                    if let Some(Value::Error(error)) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(Value::Error(error.clone()));
                    } else {
                        match scheduler.create_target(AsyncTaskTarget::Filesystem {
                            operation,
                            arguments,
                        }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::AsyncIoRead
                | Instruction::AsyncIoReadExact
                | Instruction::AsyncIoReadAll
                | Instruction::AsyncIoOpen
                | Instruction::AsyncIoWrite
                | Instruction::AsyncIoWriteAll
                | Instruction::AsyncIoFlush
                | Instruction::AsyncIoSeek => {
                    let operation = match &program[pc] {
                        Instruction::AsyncIoRead => AsyncFileOperation::Read,
                        Instruction::AsyncIoReadExact => AsyncFileOperation::ReadExact,
                        Instruction::AsyncIoReadAll => AsyncFileOperation::ReadAll,
                        Instruction::AsyncIoOpen => AsyncFileOperation::Open,
                        Instruction::AsyncIoWrite => AsyncFileOperation::Write,
                        Instruction::AsyncIoWriteAll => AsyncFileOperation::WriteAll,
                        Instruction::AsyncIoFlush => AsyncFileOperation::Flush,
                        Instruction::AsyncIoSeek => AsyncFileOperation::Seek,
                        _ => unreachable!(),
                    };
                    let arguments = take_stack_tail(&mut stack, operation.arity())?;
                    if let Some(Value::Error(error)) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(Value::Error(error.clone()));
                    } else {
                        match scheduler.create_target(AsyncTaskTarget::FileHandle {
                            operation,
                            arguments,
                        }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::ProcessRunAsync
                | Instruction::ProcessRunTimeoutAsync
                | Instruction::ProcessRunInputAsync
                | Instruction::ProcessRunInputTimeoutAsync
                | Instruction::ProcessRunInputTimeoutOutputLimitAsync => {
                    let process_instruction = &program[pc];
                    let arity = match process_instruction {
                        Instruction::ProcessRunAsync => 2,
                        Instruction::ProcessRunTimeoutAsync | Instruction::ProcessRunInputAsync => {
                            3
                        }
                        Instruction::ProcessRunInputTimeoutAsync => 4,
                        Instruction::ProcessRunInputTimeoutOutputLimitAsync => 5,
                        _ => unreachable!(),
                    };
                    let arguments = take_stack_tail(&mut stack, arity)?;
                    if let Some(error) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(error.clone());
                    } else {
                        let mut arguments = arguments.into_iter();
                        let program = arguments.next().ok_or(RuntimeError::StackUnderflow)?;
                        let process_arguments =
                            arguments.next().ok_or(RuntimeError::StackUnderflow)?;
                        let (input, timeout, output_limit) = match arity {
                            2 => (None, None, None),
                            3 if matches!(
                                process_instruction,
                                Instruction::ProcessRunInputAsync
                            ) =>
                            {
                                (arguments.next(), None, None)
                            }
                            3 => match arguments.next() {
                                Some(value @ Value::Bytes(_)) => (Some(value), None, None),
                                Some(value @ (Value::Int(_) | Value::UInt(_))) => {
                                    (None, Some(value), None)
                                }
                                value => (value, None, None),
                            },
                            4 => {
                                let first = arguments.next();
                                let second = arguments.next();
                                match first {
                                    Some(value @ Value::Bytes(_)) => (Some(value), second, None),
                                    Some(value @ (Value::Int(_) | Value::UInt(_))) => {
                                        (None, Some(value), second)
                                    }
                                    Some(Value::Nil) => (None, None, second),
                                    value => (value, second, None),
                                }
                            }
                            5 => {
                                let input = arguments.next();
                                let timeout = arguments.next();
                                let output_limit = arguments.next();
                                (input, timeout, output_limit)
                            }
                            _ => unreachable!(),
                        };
                        match scheduler.create_target(AsyncTaskTarget::ProcessRun {
                            program,
                            arguments: process_arguments,
                            input,
                            timeout,
                            output_limit,
                        }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::NetworkExchangeAsync => {
                    let arguments = take_stack_tail(&mut stack, 4)?;
                    if let Some(Value::Error(error)) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(Value::Error(error.clone()));
                    } else {
                        match scheduler
                            .create_target(AsyncTaskTarget::NetworkExchange { arguments })
                        {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::TcpConnectAsync
                | Instruction::TcpListenAsync
                | Instruction::TcpAcceptAsync
                | Instruction::TcpReadAsync
                | Instruction::TcpReadExactAsync
                | Instruction::TcpReadAllAsync
                | Instruction::TcpWriteAsync
                | Instruction::TcpWriteAllAsync
                | Instruction::TcpShutdownWriteAsync => {
                    let operation = match &program[pc] {
                        Instruction::TcpListenAsync => AsyncTcpOperation::Listen,
                        Instruction::TcpAcceptAsync => AsyncTcpOperation::Accept,
                        Instruction::TcpConnectAsync => AsyncTcpOperation::Connect,
                        Instruction::TcpReadAsync => AsyncTcpOperation::Read,
                        Instruction::TcpReadExactAsync => AsyncTcpOperation::ReadExact,
                        Instruction::TcpReadAllAsync => AsyncTcpOperation::ReadAll,
                        Instruction::TcpWriteAsync => AsyncTcpOperation::Write,
                        Instruction::TcpWriteAllAsync => AsyncTcpOperation::WriteAll,
                        Instruction::TcpShutdownWriteAsync => AsyncTcpOperation::ShutdownWrite,
                        _ => unreachable!(),
                    };
                    let arguments = take_stack_tail(&mut stack, operation.arity())?;
                    if let Some(Value::Error(error)) = arguments
                        .iter()
                        .find(|argument| matches!(argument, Value::Error(_)))
                    {
                        stack.push(Value::Error(error.clone()));
                    } else {
                        match scheduler.create_target(AsyncTaskTarget::Tcp {
                            operation,
                            arguments,
                        }) {
                            Ok(task) => stack.push(Value::Task(task)),
                            Err(error) if error.is_recoverable() => {
                                stack.push(Value::Error(Box::new(error)))
                            }
                            Err(error) => return Err(error),
                        }
                    }
                }
                Instruction::Call(index, arity) => {
                    let signature = execution_plan
                        .callable_signature(artifact, *index)
                        .ok_or(RuntimeError::InvalidCall(*index))?;
                    check_call(artifact, *index, signature, *arity, 0)?;
                    if signature.asynchronous {
                        return Err(RuntimeError::InvalidCall(*index));
                    }
                    check_branch(program, pc, pc + 1, callable_start, callable_end)?;
                    let next_base = stack
                        .len()
                        .checked_sub(*arity)
                        .ok_or(RuntimeError::StackUnderflow)?;
                    if !call_arguments_match(signature, &stack[next_base..]) {
                        stack.truncate(next_base);
                        stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                            op: "call",
                        })));
                    } else {
                        reserve_frame(&mut call_stack, context.limits.max_call_frames)?;
                        call_stack.push((
                            pc + 1,
                            bp,
                            captures.clone(),
                            operand_base,
                            callable_start,
                            callable_end,
                        ));
                        callable_start = *index;
                        callable_end = execution_plan.next_callable_boundary(artifact, *index);
                        operand_base = next_base;
                        captures = CaptureValues::default();
                        bp = locals.len();
                        pc = check_jump(program, *index)?;
                        if let Some(exit) = execution_plan.try_jit_call(
                            artifact,
                            callable_start,
                            &mut pc,
                            &mut locals,
                            bp,
                            &mut stack,
                            operand_base,
                            &captures,
                            &mut propagation_pending,
                            context,
                        ) {
                            pc = match exit {
                                JitExit::Completed => pc,
                                JitExit::ResumeAt { pc } | JitExit::FaultAt { pc } => pc,
                            };
                        }
                        continue;
                    }
                }
                Instruction::Return | Instruction::ReturnPropagate => {
                    let propagates = matches!(program[pc], Instruction::ReturnPropagate);
                    if stack.len() <= operand_base {
                        return Err(RuntimeError::StackUnderflow);
                    }
                    let result = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = execution_plan
                        .callable_signature(artifact, callable_start)
                        .and_then(|signature| signature.result_type.as_ref())
                        .filter(|expected| !matches_runtime_type(expected, &result))
                        .map_or(result, |_| {
                            Value::Error(Box::new(RuntimeError::TypeMismatch { op: "return" }))
                        });
                    if propagates && !matches!(result, Value::Error(_)) {
                        return Err(RuntimeError::InvalidArgument {
                            op: "propagate non-error return",
                        });
                    }
                    if matches!(result, Value::Error(_)) {
                        error_value_pc = Some(pc);
                    }
                    if call_stack.is_empty() {
                        if async_task.is_some() {
                            if propagates {
                                let Value::Error(error) = result else {
                                    unreachable!("propagated return was checked above")
                                };
                                return Err(*error);
                            }
                            return Ok(result);
                        }
                        if propagates {
                            let Value::Error(error) = result else {
                                unreachable!("propagated return was checked above")
                            };
                            return Err(*error);
                        }
                        return Err(RuntimeError::CallStackUnderflow);
                    }
                    let (
                        return_pc,
                        saved_bp,
                        saved_captures,
                        saved_operand_base,
                        saved_callable_start,
                        saved_callable_end,
                    ) = call_stack.pop().ok_or(RuntimeError::CallStackUnderflow)?;
                    stack.truncate(operand_base);
                    stack.push(result);
                    propagation_pending |= propagates;
                    operand_base = saved_operand_base;
                    callable_start = saved_callable_start;
                    callable_end = saved_callable_end;
                    locals.truncate(bp);
                    bp = saved_bp;
                    captures = saved_captures;
                    pc = check_jump(program, return_pc)?;
                    continue;
                }
                Instruction::And => {
                    binop(&mut stack, |a, b| match (a, b) {
                        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x && y)),
                        _ => Err(RuntimeError::TypeMismatch { op: "and" }),
                    })?;
                }
                Instruction::Or => {
                    binop(&mut stack, |a, b| match (a, b) {
                        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x || y)),
                        _ => Err(RuntimeError::TypeMismatch { op: "or" }),
                    })?;
                }
                Instruction::Xor => {
                    binop(&mut stack, |a, b| match (a, b) {
                        (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x != y)),
                        _ => Err(RuntimeError::TypeMismatch { op: "xor" }),
                    })?;
                }
                Instruction::Dup => {
                    let top = stack.last().ok_or(RuntimeError::StackUnderflow)?.clone();
                    stack.push(top);
                }
                Instruction::Pop => {
                    stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                }
                Instruction::Swap => {
                    let (a, b) = pop2(&mut stack)?;
                    stack.push(b);
                    stack.push(a);
                }
                Instruction::StackTruncate(depth) => {
                    let target = operand_base
                        .checked_add(*depth)
                        .filter(|target| *target <= stack.len())
                        .ok_or(RuntimeError::StackUnderflow)?;
                    stack.truncate(target);
                }
                Instruction::CallValue(arity) => {
                    let target = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let next_base = stack
                        .len()
                        .checked_sub(*arity)
                        .ok_or(RuntimeError::StackUnderflow)?;
                    let foreign = match &target {
                        Value::Function(_) => true,
                        Value::BoundFunction(function) => !Rc::ptr_eq(&function.owner, owner),
                        Value::Closure(_, captures) => !captures.belongs_to(owner),
                        _ => false,
                    };
                    if foreign {
                        stack.truncate(next_base);
                        stack.push(Value::Error(Box::new(RuntimeError::InvalidArgument {
                            op: "foreign callable",
                        })));
                    } else {
                        match target {
                            Value::BoundFunction(function) => {
                                let addr = function.entry;
                                let signature = execution_plan
                                    .callable_signature(artifact, addr)
                                    .ok_or(RuntimeError::InvalidCall(addr))?;
                                check_call(artifact, addr, signature, *arity, 0)?;
                                check_branch(program, pc, pc + 1, callable_start, callable_end)?;
                                if signature.asynchronous {
                                    stack.truncate(next_base);
                                    stack.push(Value::Error(Box::new(
                                        RuntimeError::InvalidArgument {
                                            op: "call async function value",
                                        },
                                    )));
                                } else if !call_arguments_match(signature, &stack[next_base..]) {
                                    stack.truncate(next_base);
                                    stack.push(Value::Error(Box::new(
                                        RuntimeError::TypeMismatch { op: "call" },
                                    )));
                                } else {
                                    reserve_frame(&mut call_stack, context.limits.max_call_frames)?;
                                    call_stack.push((
                                        pc + 1,
                                        bp,
                                        captures.clone(),
                                        operand_base,
                                        callable_start,
                                        callable_end,
                                    ));
                                    callable_start = addr;
                                    callable_end =
                                        execution_plan.next_callable_boundary(artifact, addr);
                                    operand_base = next_base;
                                    captures = CaptureValues::default();
                                    bp = locals.len();
                                    pc = check_jump(program, addr)?;
                                    continue;
                                }
                            }
                            Value::Closure(addr, closure_captures) => {
                                let signature = execution_plan
                                    .callable_signature(artifact, addr)
                                    .ok_or(RuntimeError::InvalidCall(addr))?;
                                check_call(
                                    artifact,
                                    addr,
                                    signature,
                                    *arity,
                                    closure_captures.len(),
                                )?;
                                check_branch(program, pc, pc + 1, callable_start, callable_end)?;
                                if signature.asynchronous {
                                    stack.truncate(next_base);
                                    stack.push(Value::Error(Box::new(
                                        RuntimeError::InvalidArgument {
                                            op: "call async function value",
                                        },
                                    )));
                                } else if !call_arguments_match(signature, &stack[next_base..]) {
                                    stack.truncate(next_base);
                                    stack.push(Value::Error(Box::new(
                                        RuntimeError::TypeMismatch { op: "call" },
                                    )));
                                } else {
                                    reserve_frame(&mut call_stack, context.limits.max_call_frames)?;
                                    call_stack.push((
                                        pc + 1,
                                        bp,
                                        captures.clone(),
                                        operand_base,
                                        callable_start,
                                        callable_end,
                                    ));
                                    callable_start = addr;
                                    callable_end =
                                        execution_plan.next_callable_boundary(artifact, addr);
                                    operand_base = next_base;
                                    captures = CaptureValues(Some(closure_captures));
                                    bp = locals.len();
                                    pc = check_jump(program, addr)?;
                                    continue;
                                }
                            }
                            Value::Error(_) => {
                                stack.truncate(next_base);
                                stack.push(target);
                            }
                            _ => {
                                stack.truncate(next_base);
                                stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                    op: "call_value",
                                })));
                            }
                        }
                    }
                }
                Instruction::Index => {
                    let (object, key) = pop2(&mut stack)?;
                    let key = match (&object, key) {
                        (Value::Record(_), Value::Int(index)) if index >= 0 => {
                            Value::Str(Rc::from(index.to_string()))
                        }
                        (_, key) => key,
                    };
                    // Type metadata is unspellable in source. This generic query
                    // lets lowered patterns inspect nominal identity without a
                    // match/enum instruction or a forgeable record field.
                    let result = match (&object, &key) {
                        (Value::Variant(tag, _), Value::TypeTag(_)) => {
                            Ok(Value::TypeTag(tag.clone()))
                        }
                        (Value::Error(err), Value::TypeTag(_)) => Ok(Value::TypeTag(err.tag())),
                        (_, Value::TypeTag(_)) => Ok(Value::Nil),
                        (Value::Error(err), Value::Str(name)) => {
                            err.field(name).map_or_else(|| missing_field(name), Ok)
                        }
                        (Value::Error(_), _) => Ok(object),
                        (_, Value::Error(_)) => Ok(key),
                        (Value::Array(arr), Value::Int(i)) => arr
                            .try_borrow()
                            .map_err(|_| borrow_conflict("index"))
                            .and_then(|items| {
                                items
                                    .get(usize::try_from(*i).unwrap_or(usize::MAX))
                                    .cloned()
                                    .ok_or(RuntimeError::IndexOutOfBounds {
                                        index: *i,
                                        len: items.len(),
                                    })
                            }),
                        (Value::Record(record), Value::Str(name)) => record
                            .try_get_field(name.as_ref())
                            .map_err(|_| borrow_conflict("index"))
                            .and_then(|value| value.map_or_else(|| missing_field(name), Ok)),
                        (Value::Variant(_, fields), Value::Str(name)) => fields
                            .get(name.as_ref())
                            .map_or_else(|| missing_field(name), |value| Ok(value.clone())),
                        _ => Err(RuntimeError::TypeMismatch { op: "index" }),
                    };
                    stack.push(match result {
                        Ok(value) => value,
                        Err(error) => Value::Error(Box::new(error)),
                    });
                }
                Instruction::IndexSet => {
                    let (array, idx, value) = pop3(&mut stack)?;
                    let idx = match (&array, idx) {
                        (Value::Record(_), Value::Int(index)) if index >= 0 => {
                            Value::Str(Rc::from(index.to_string()))
                        }
                        (_, index) => index,
                    };
                    if let Value::Error(_) = array {
                        stack.push(array);
                    } else if let Value::Error(_) = idx {
                        stack.push(idx);
                    } else {
                        match (&array, &idx) {
                            (Value::Array(arr), Value::Int(i)) => match arr.try_borrow() {
                                Err(_) => {
                                    stack.push(Value::Error(Box::new(borrow_conflict("indexset"))))
                                }
                                Ok(items) => {
                                    let len = items.len();
                                    drop(items);
                                    if *i < 0 || (*i as usize) >= len {
                                        stack.push(Value::Error(Box::new(
                                            RuntimeError::IndexOutOfBounds { index: *i, len },
                                        )));
                                    } else {
                                        match arr.try_set(*i as usize, value) {
                                            Ok(()) => stack.push(array.clone()),
                                            Err(error) => stack.push(Value::Error(Box::new(error))),
                                        }
                                    }
                                }
                            },
                            (Value::Record(record), Value::Str(key)) => {
                                match record.try_get_field(key.as_ref()) {
                                    Err(_) => stack
                                        .push(Value::Error(Box::new(borrow_conflict("indexset")))),
                                    Ok(current) => {
                                        let validation = match (&value, current.as_ref()) {
                                            (
                                                Value::Array(replacement),
                                                Some(Value::Array(template)),
                                            ) if record.is_nominal() => {
                                                replacement.try_constrain_like(template)
                                            }
                                            (
                                                Value::Array(replacement),
                                                Some(Value::Array(template)),
                                            ) if replacement.needs_type_context() => {
                                                replacement.try_constrain_from(template)
                                            }
                                            (Value::Array(replacement), _)
                                                if replacement.needs_type_context() =>
                                            {
                                                Err(RuntimeError::TypeMismatch { op: "indexset" })
                                            }
                                            _ => Ok(()),
                                        };
                                        if let Err(error) = validation {
                                            let error = match error {
                                                RuntimeError::TypeMismatch { .. } => {
                                                    RuntimeError::TypeMismatch { op: "indexset" }
                                                }
                                                error => error,
                                            };
                                            stack.push(Value::Error(Box::new(error)));
                                        } else {
                                            match record.try_set_field(key.as_ref(), value) {
                                                Ok(()) => stack.push(array.clone()),
                                                Err(error) => {
                                                    stack.push(Value::Error(Box::new(error)))
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {
                                stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                    op: "indexset",
                                })));
                            }
                        }
                    }
                }
                Instruction::MakeCell => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    stack.push(Value::Cell(HeapRef::try_new(RefCell::new(value))?));
                }
                Instruction::CellGet => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = match value {
                        Value::Error(_) => value,
                        Value::Cell(ref c) => match c.try_borrow() {
                            Ok(value) => value.clone(),
                            Err(_) => Value::Error(Box::new(borrow_conflict("cellget"))),
                        },
                        _ => Value::Error(Box::new(RuntimeError::TypeMismatch { op: "cellget" })),
                    };
                    stack.push(result);
                }
                Instruction::MakeArray(count) => {
                    let items = take_stack_tail(&mut stack, *count)?;
                    stack.push(match ArrayValue::try_new(items) {
                        Ok(array) => Value::Array(HeapRef::try_new(array)?),
                        Err(error) => Value::Error(Box::new(error)),
                    });
                }
                Instruction::MakeTuple(count) => {
                    let items = take_stack_tail(&mut stack, *count)?;
                    stack.push(match ArrayValue::try_tuple(items) {
                        Ok(tuple) => Value::Array(HeapRef::try_new(tuple)?),
                        Err(error) => Value::Error(Box::new(error)),
                    });
                }
                Instruction::MakeTypedTuple(count, expected) => {
                    let items = take_stack_tail(&mut stack, *count)?;
                    stack.push(match ArrayValue::try_typed_tuple(items, expected.clone()) {
                        Ok(tuple) => Value::Array(HeapRef::try_new(tuple)?),
                        Err(error) => Value::Error(Box::new(error)),
                    });
                }
                Instruction::MakeTypedArray(count, expected) => {
                    let items = take_stack_tail(&mut stack, *count)?;
                    stack.push(match ArrayValue::try_typed(items, expected.clone()) {
                        Ok(array) => Value::Array(HeapRef::try_new(array)?),
                        Err(error) => Value::Error(Box::new(error)),
                    });
                }
                Instruction::ConstrainArrayType(expected) => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = match value {
                        Value::Array(array) => match array.try_constrain(expected.clone()) {
                            Ok(()) => Value::Array(array),
                            Err(error) => Value::Error(Box::new(error)),
                        },
                        Value::Error(error) => Value::Error(error),
                        _ => {
                            Value::Error(Box::new(RuntimeError::TypeMismatch { op: "array_type" }))
                        }
                    };
                    stack.push(result);
                }
                Instruction::ConstrainRecordFields(contracts) => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = match value {
                        Value::Record(record) => match record.try_constrain_fields(contracts) {
                            Ok(()) => Value::Record(record),
                            Err(error) => Value::Error(Box::new(error)),
                        },
                        Value::Error(error) => Value::Error(error),
                        _ => {
                            Value::Error(Box::new(RuntimeError::TypeMismatch { op: "record_type" }))
                        }
                    };
                    stack.push(result);
                }
                Instruction::ConstrainArrayFrom => {
                    let (value, template) = pop2(&mut stack)?;
                    let result = match value {
                        Value::Array(array) if !array.needs_type_context() => Value::Array(array),
                        Value::Array(array) => match template {
                            Value::Array(template) => match array.try_constrain_from(&template) {
                                Ok(()) => Value::Array(array),
                                Err(error) => Value::Error(Box::new(error)),
                            },
                            _ => Value::Error(Box::new(RuntimeError::TypeMismatch {
                                op: "array_type",
                            })),
                        },
                        value => value,
                    };
                    stack.push(result);
                }
                Instruction::ArrayEmptyLike => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    let result = match value {
                        Value::Array(array) => match array.element_type() {
                            Some(element_type) => match ArrayValue::typed(Vec::new(), element_type)
                            {
                                Ok(empty) => Value::Array(HeapRef::try_new(empty)?),
                                Err(_) => Value::Error(Box::new(RuntimeError::TypeMismatch {
                                    op: "array_empty_like",
                                })),
                            },
                            None => Value::Error(Box::new(RuntimeError::TypeMismatch {
                                op: "array_empty_like",
                            })),
                        },
                        Value::Error(error) => Value::Error(error),
                        _ => Value::Error(Box::new(RuntimeError::TypeMismatch {
                            op: "array_empty_like",
                        })),
                    };
                    stack.push(result);
                }
                Instruction::MakeRecord(count) => {
                    let slots = count.checked_mul(2).ok_or(RuntimeError::StackUnderflow)?;
                    let entries = take_stack_tail(&mut stack, slots)?;
                    let mut entries = entries.into_iter();
                    let mut fields = std::collections::HashMap::new();
                    let mut tag = None;
                    let mut error = None;
                    for _ in 0..*count {
                        let key = entries.next().ok_or(RuntimeError::StackUnderflow)?;
                        let value = entries.next().ok_or(RuntimeError::StackUnderflow)?;
                        match key {
                            Value::TypeTag(name) if tag.is_none() => tag = Some(name),
                            Value::Str(key) => {
                                fields
                                    .try_reserve(1)
                                    .map_err(|_| RuntimeError::ResourceLimit {
                                        resource: "record",
                                    })?;
                                fields.insert(try_record_key(&key)?, value);
                            }
                            Value::Error(error_value) if error.is_none() => {
                                error = Some(Value::Error(error_value));
                            }
                            _ if error.is_none() => {
                                error = Some(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                    op: "record",
                                })))
                            }
                            _ => {}
                        }
                    }
                    let value = if let Some(error) = error {
                        error
                    } else {
                        match tag {
                            Some(tag)
                                if tag
                                    .strip_prefix(crate::error::ERROR_TYPE)
                                    .is_some_and(|tail| tail.starts_with("::")) =>
                            {
                                Value::Error(Box::new(RuntimeError::LanguageError {
                                    tag,
                                    fields: HeapRef::try_new(ValueFields::new(fields))?,
                                }))
                            }
                            Some(tag) if tag.starts_with("@struct::") => {
                                Value::Record(HeapRef::try_new(RecordValue::nominal(
                                    fields,
                                    try_record_key(&tag[9..])?,
                                ))?)
                            }
                            Some(tag) => {
                                Value::Variant(tag, HeapRef::try_new(ValueFields::new(fields))?)
                            }
                            None => Value::Record(HeapRef::try_new(RecordValue::new(fields))?),
                        }
                    };
                    stack.push(value);
                }
                Instruction::CellSet => {
                    let (cell, new_value) = pop2(&mut stack)?;
                    if let Value::Error(_) = cell {
                        stack.push(cell);
                    } else if let Value::Error(_) = new_value {
                        stack.push(new_value);
                    } else {
                        match &cell {
                            Value::Cell(c) => match c.try_borrow_mut() {
                                Ok(mut value) => {
                                    *value = new_value;
                                    stack.push(cell.clone());
                                }
                                Err(_) => {
                                    stack.push(Value::Error(Box::new(borrow_conflict("cellset"))))
                                }
                            },
                            _ => stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                op: "cellset",
                            }))),
                        }
                    }
                }
                Instruction::MakeClosure(addr, count) => {
                    check_jump(program, *addr)?;
                    if execution_plan
                        .callable_signature(artifact, *addr)
                        .is_none_or(|signature| signature.captures != *count)
                    {
                        return Err(RuntimeError::InvalidCall(*addr));
                    }
                    let captured = take_stack_tail(&mut stack, *count)?;
                    stack.push(Value::Closure(
                        *addr,
                        HeapRef::try_new(ClosureCaptures::for_program(captured, owner))?,
                    ));
                }
                Instruction::LoadCapture(index) => {
                    let value = captures
                        .get(*index)
                        .ok_or(RuntimeError::InvalidCapture(*index))?;
                    stack.push(value);
                }
                Instruction::Crash(context) => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    return Err(RuntimeError::Crashed {
                        context: context.clone(),
                        value,
                    });
                }
                Instruction::Not => {
                    let value = stack.pop().ok_or(RuntimeError::StackUnderflow)?;
                    if let Value::Error(_) = value {
                        stack.push(value);
                    } else {
                        match value {
                            Value::Bool(x) => stack.push(Value::Bool(!x)),
                            _ => stack.push(Value::Error(Box::new(RuntimeError::TypeMismatch {
                                op: "not",
                            }))),
                        }
                    }
                }
            }
            if produced != 0 {
                let output_start = stack.len().saturating_sub(produced);
                if stack[output_start..]
                    .iter()
                    .any(|value| matches!(value, Value::Error(_)))
                {
                    error_value_pc = Some(pc);
                }
            }
            if callable_end < program.len() && pc + 1 == callable_end {
                return Err(RuntimeError::InvalidJump(pc + 1));
            }
            pc += 1;
        }
        Err(RuntimeError::MissingHalt)
    })();

    if suspended {
        return VmExecution::Suspended(VmContinuation {
            pc,
            stack,
            locals,
            bp,
            call_stack,
            operand_base,
            captures,
            error_value_pc,
            halt_error_pc,
            propagation_pending,
            callable_start,
            callable_end,
        });
    }

    let source_pc = result.as_ref().err().map(|_| halt_error_pc.unwrap_or(pc));
    let source_location = source_pc.and_then(|instruction| {
        artifact
            .source_location(instruction)
            .or_else(|| artifact.source_location_before(instruction))
            .cloned()
    });
    context.set_source_location(source_location);

    // Keep the VM's roots alive until the collector has detached all edges in
    // objects that become unreachable when this execution frame exits.
    if let Err(cleanup) = reserve_cleanup_roots(
        &mut stack,
        [
            locals.iter().flatten().count(),
            if async_task.is_none() {
                globals.iter().flatten().count()
            } else {
                0
            },
            usize::from(captures.0.is_some()),
            call_stack
                .iter()
                .filter(|(_, _, captures, _, _, _)| captures.0.is_some())
                .count(),
        ],
    ) {
        // Match the collector's reservation-failure protocol: retain all roots
        // rather than recursively dropping a graph under allocation pressure.
        std::mem::forget(stack);
        std::mem::forget(locals);
        if async_task.is_none() {
            std::mem::forget(std::mem::take(globals));
        }
        std::mem::forget(captures);
        std::mem::forget(call_stack);
        if let Ok(value) = result {
            std::mem::forget(value);
        }
        return VmExecution::Complete(Err(cleanup));
    }
    let mut discarded = stack;
    discarded.extend(locals.into_iter().flatten());
    if async_task.is_none() {
        discarded.extend(std::mem::take(globals).into_iter().flatten());
    }
    if let Some(captures) = captures.into_root() {
        discarded.push(captures);
    }
    for (_, _, frame_captures, _, _, _) in call_stack {
        if let Some(captures) = frame_captures.into_root() {
            discarded.push(captures);
        }
    }
    let collection = heap::collect(&discarded).map(|_| ());
    if collection.is_ok() {
        drop(discarded);
    } else {
        // Preserve the graph if collection cannot reserve its work list. A
        // recursive fallback drop could overflow the host stack.
        std::mem::forget(discarded);
    }
    VmExecution::Complete(match (result, collection) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(primary), Err(secondary)) if primary.is_recoverable() => {
            Err(primary.with_suppressed(secondary))
        }
        (Err(primary), _) => Err(primary),
        (Ok(_), Err(cleanup)) => Err(cleanup),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_root_reservation_preserves_values_on_size_or_allocation_failure() {
        let mut values = vec![Value::Int(7)];
        for counts in [[usize::MAX, 1, 0, 0], [usize::MAX, 0, 0, 0]] {
            assert!(matches!(
                reserve_cleanup_roots(&mut values, counts),
                Err(RuntimeError::ResourceLimit {
                    resource: "VM cleanup roots"
                })
            ));
            assert_eq!(values.len(), 1);
            assert!(matches!(values[0], Value::Int(7)));
        }
    }

    #[derive(Default)]
    struct TrackingWriter {
        buf: Vec<u8>,
        flush_count: usize,
    }

    impl std::io::Write for TrackingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.buf.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flush_count += 1;
            Ok(())
        }
    }

    fn compile(source: &str) -> CompiledProgram {
        crate::project::compile_source(
            "vm-test.cog",
            &format!("@!include(\"std\") use io; use err::Error; {source}"),
        )
        .unwrap()
    }

    #[test]
    fn runtime_function_references_are_execution_scoped_and_not_serializable() {
        let owner = Rc::new(());
        let function = instantiate_scalar(&Value::Function(3), &owner);
        let mut program = CompiledProgram::new(vec![
            Instruction::Push(function),
            Instruction::CallValue(0),
            Instruction::Halt,
            Instruction::Push(Value::Nil),
            Instruction::Return,
        ]);
        program
            .callables
            .insert(3, crate::backend::CallableSignature::dynamic(0, 0));
        let error = crate::bytecode::encode(&program).unwrap_err();
        assert!(error.contains("runtime objects"), "{error}");

        let mut output = Vec::new();
        let mut context = RuntimeContext::new(&mut output, VmCapabilities::default());
        run_with_callable_owner(&program, &mut context, &owner).unwrap();
        assert!(matches!(
            run_with(&program, &mut Vec::new()),
            Err(RuntimeError::InvalidArgument {
                op: "foreign callable"
            })
        ));
    }

    #[test]
    fn record_key_copy_reports_allocation_failure() {
        assert!(matches!(
            try_record_key_with_capacity("field", usize::MAX),
            Err(RuntimeError::ResourceLimit { resource: "record" })
        ));
    }

    #[test]
    fn record_index_set_can_insert_a_new_field() {
        let record = HeapRef::new(RecordValue::new(std::collections::HashMap::new()));
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Record(record.clone())),
            Instruction::Push(Value::Str("field".into())),
            Instruction::Push(Value::Int(7)),
            Instruction::IndexSet,
            Instruction::Pop,
            Instruction::Halt,
        ]);

        run_with_capabilities(&program, &mut Vec::new(), VmCapabilities::default()).unwrap();

        assert!(matches!(
            record.try_get_field("field").unwrap(),
            Some(Value::Int(7))
        ));
    }

    #[test]
    fn print_is_blocked_when_permission_is_disabled() {
        let program = compile("fn main() { return io::print(7); }");
        let mut out = Vec::new();
        assert!(matches!(
            run_with_capabilities(&program, &mut out, VmCapabilities::restricted()),
            Err(RuntimeError::PermissionDenied { .. })
        ));
        assert!(out.is_empty());
    }

    #[test]
    fn execution_collects_unreachable_self_referential_cells() {
        assert_eq!(crate::heap::live_objects(), 0);
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Nil),
            Instruction::MakeCell,
            Instruction::Dup,
            Instruction::Dup,
            Instruction::CellSet,
            Instruction::Pop,
            Instruction::Halt,
        ]);
        run_with_capabilities(&program, &mut Vec::new(), VmCapabilities::restricted()).unwrap();
        assert_eq!(crate::heap::live_objects(), 0);
    }

    #[test]
    fn forgotten_heap_roots_remain_charged_to_the_runtime_object_limit() {
        let limit = crate::heap::live_objects();
        let leaked = Value::Record(HeapRef::new(RecordValue::new(
            std::collections::HashMap::new(),
        )));
        std::mem::forget(leaked);

        let program = CompiledProgram::new(vec![Instruction::Halt]);
        let mut output = Vec::new();
        let mut context = RuntimeContext::new(&mut output, VmCapabilities::restricted());
        context.limits.max_heap_objects = Some(limit);
        assert!(matches!(
            run_with_context(&program, &mut context),
            Err(RuntimeError::ResourceLimit {
                resource: "VM heap objects"
            })
        ));
        assert_eq!(crate::heap::live_objects(), limit + 1);
    }

    #[test]
    fn execution_drops_deep_managed_graphs_iteratively() {
        assert_eq!(crate::heap::live_objects(), 0);
        const DEPTH: usize = 5_000;
        let mut instructions = Vec::with_capacity(DEPTH * 3 + 3);
        instructions.push(Instruction::Push(Value::Int(0)));
        for _ in 0..DEPTH {
            instructions.push(Instruction::Push(Value::Str("next".into())));
            instructions.push(Instruction::MakeRecord(1));
            instructions.push(Instruction::MakeArray(1));
        }
        instructions.push(Instruction::Pop);
        instructions.push(Instruction::Halt);
        let program = CompiledProgram::new(instructions);
        run_with_capabilities(&program, &mut Vec::new(), VmCapabilities::restricted()).unwrap();
        assert_eq!(crate::heap::live_objects(), 0);
    }

    #[test]
    fn push_array_checks_nested_heap_cost_before_allocating() {
        let template = ArrayTemplate::Array(vec![crate::template::ArrayItem {
            template: ArrayTemplate::Array(Vec::new()),
            repeat: 2,
        }]);
        assert_eq!(array_template_cost(&template).unwrap(), (3, 3));

        let program =
            CompiledProgram::new(vec![Instruction::PushArray(template), Instruction::Halt]);
        let mut out = Vec::new();
        let mut context = RuntimeContext::new(&mut out, VmCapabilities::restricted());
        context.limits.max_heap_objects = Some(2);
        assert!(matches!(
            run_with_context(&program, &mut context),
            Err(RuntimeError::ResourceLimit {
                resource: "VM heap objects"
            })
        ));
        assert_eq!(crate::heap::live_objects(), 0);
    }

    #[test]
    fn push_array_cost_rejects_excessive_in_memory_templates() {
        let template = ArrayTemplate::Array(vec![crate::template::ArrayItem {
            template: ArrayTemplate::Scalar(Value::Nil),
            repeat: crate::bytecode::LIMIT + 1,
        }]);
        assert!(matches!(
            array_template_cost(&template),
            Err(RuntimeError::ResourceLimit {
                resource: "array template"
            })
        ));
    }

    #[test]
    fn print_and_println_flush_output() {
        let program = compile("fn main() { io::print(\"hi\"); io::println(42); }");
        let mut out = TrackingWriter::default();
        run_with(&program, &mut out).unwrap();
        assert_eq!(out.buf, b"hi42\n");
        assert_eq!(out.flush_count, 2);
    }

    #[test]
    fn exclusive_reports_shared_values() {
        let program = compile("fn main() { const a = [1]; const b = a; io::print(exclusive(a)); }");
        let mut out = Vec::new();
        run_with(&program, &mut out).unwrap();
        assert_eq!(out, b"false");
    }

    #[test]
    fn sleep_is_blocked_when_permission_is_disabled() {
        let program = compile("use time; fn main() { io::print(time::sleep(1)); }");
        let caps = VmCapabilities {
            allow_sleep: false,
            ..VmCapabilities::default()
        };
        let mut out = Vec::new();
        run_with_capabilities(&program, &mut out, caps).unwrap();
        assert_eq!(out, b"Error: permission denied for `time::sleep`");
    }

    #[test]
    fn array_push_mutates_in_place_through_alias() {
        // `arr << 3;` as a bare statement (result popped/discarded, exactly like
        // codegen emits for `Stmt::Expr`) must still be visible through any other
        // handle to the same array — that's the whole point of it being
        // `Rc<RefCell<..>>` rather than a plain Vec. The old "reuse Add" codegen
        // built a brand-new array and left `arr` untouched, so a bare `arr << x;`
        // silently did nothing.
        let arr = HeapRef::new(ArrayValue::new(vec![Value::Int(1), Value::Int(2)]).unwrap());
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Array(arr.clone())),
            Instruction::Push(Value::Int(3)),
            Instruction::ArrayPush,
            Instruction::Pop, // as if `arr << 3;` were a statement, not assigned
            Instruction::Halt,
        ]);
        let mut out = Vec::new();
        run_with_capabilities(&program, &mut out, VmCapabilities::default()).unwrap();
        let borrowed = arr.borrow();
        assert_eq!(borrowed.len(), 3);
        assert!(matches!(borrowed[0], Value::Int(1)));
        assert!(matches!(borrowed[1], Value::Int(2)));
        assert!(matches!(borrowed[2], Value::Int(3)));
    }

    #[test]
    fn array_prepend_mutates_in_place_through_alias() {
        let arr = HeapRef::new(ArrayValue::new(vec![Value::Int(2), Value::Int(3)]).unwrap());
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Int(1)),
            Instruction::Push(Value::Array(arr.clone())),
            Instruction::ArrayPrepend,
            Instruction::Pop, // as if `1 >> arr;` were a statement
            Instruction::Halt,
        ]);
        let mut out = Vec::new();
        run_with_capabilities(&program, &mut out, VmCapabilities::default()).unwrap();
        let borrowed = arr.borrow();
        assert_eq!(borrowed.len(), 3);
        assert!(matches!(borrowed[0], Value::Int(1)));
        assert!(matches!(borrowed[1], Value::Int(2)));
        assert!(matches!(borrowed[2], Value::Int(3)));
    }

    #[test]
    fn array_push_on_non_array_is_type_mismatch() {
        let program = compile(
            "fn bad() { return 1; } fn main() { when bad() << 2 is Error { io::print(true); } }",
        );
        let mut out = Vec::new();
        run_with(&program, &mut out).unwrap();
        assert_eq!(out, b"true");
    }

    #[test]
    fn eq_between_nil_and_non_nil_is_false_not_type_mismatch() {
        let program = compile("fn main() { io::print(5 == nil); }");
        let mut out = Vec::new();
        run_with(&program, &mut out).unwrap();
        assert_eq!(out, b"false");
    }

    #[test]
    fn ne_between_nil_and_non_nil_is_true_not_type_mismatch() {
        let program = compile("fn main() { io::print(\"hi\" != nil); }");
        let mut out = Vec::new();
        run_with(&program, &mut out).unwrap();
        assert_eq!(out, b"true");
    }

    #[test]
    fn array_push_rejects_an_incompatible_value_without_mutating_aliases() {
        let arr = HeapRef::new(ArrayValue::new(vec![Value::Int(1)]).unwrap());
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Array(arr.clone())),
            Instruction::Push(Value::Str("wrong".into())),
            Instruction::ArrayPush,
            Instruction::Halt,
        ]);
        let error = run_with_capabilities(&program, &mut Vec::new(), VmCapabilities::default())
            .unwrap_err();
        assert!(matches!(
            error,
            RuntimeError::TypeMismatch { op: "arrpush" }
        ));
        assert_eq!(arr.borrow().len(), 1);
        assert!(matches!(arr.borrow()[0], Value::Int(1)));
    }

    #[test]
    fn array_index_set_rejects_an_incompatible_value_without_mutating_aliases() {
        let arr = HeapRef::new(ArrayValue::new(vec![Value::Int(1)]).unwrap());
        let program = CompiledProgram::new(vec![
            Instruction::Push(Value::Array(arr.clone())),
            Instruction::Push(Value::Int(0)),
            Instruction::Push(Value::Str("wrong".into())),
            Instruction::IndexSet,
            Instruction::Halt,
        ]);
        let error = run_with_capabilities(&program, &mut Vec::new(), VmCapabilities::default())
            .unwrap_err();
        assert!(matches!(
            error,
            RuntimeError::TypeMismatch { op: "indexset" }
        ));
        assert_eq!(arr.borrow().len(), 1);
        assert!(matches!(arr.borrow()[0], Value::Int(1)));
    }
}
