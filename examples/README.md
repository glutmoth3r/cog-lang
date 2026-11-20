# Cog examples

Run these commands from the repository root after installing Cog. Alternatively,
replace `cog` with `cargo run --locked --offline --bin cog --`.

```sh
cog run examples/hello.cog
cog run examples/command_line_args.cog -- first "two words"
cog run examples/functions.cog
cog run examples/comptime_factorial.cog
cog run examples/comptime_types.cog
cog run examples/comptime_structs.cog
cog run examples/comptime_errors.cog
cog run examples/extensions.cog
cog run examples/generics_traits.cog
cog run examples/package_include
cog check examples/library_package
cog run examples/number_detective.cog
cog run examples/cog_pass.cog -- 32
cog run examples/cog_pass.cog -- 32 | wl-copy
cog run examples/todo.cog
cog run examples/async_concurrency.cog
cog run examples/async_stdio.cog
cog run examples/async_timeout.cog
cog run examples/async_select.cog
cog run examples/async_select_many.cog
cog run examples/async_cancellation.cog
cog run examples/async_process_cancel.cog
cog run examples/async_process_input.cog
cog run examples/async_tcp_cancel.cog
cog run examples/async_filesystem.cog
cog run examples/async_lines_and_append.cog
cog run examples/async_file_handle.cog
cog run examples/async_partial_write.cog
cog run examples/async_methods.cog
```

The todo example reads and writes `txt/todo.txt` relative to the process
working directory. Run it from the repository root. Each line stores one note
with a `[ ]` or `[x]` completion marker. The example data file is included in
the repository and starts empty.

## External libraries

Repository-specific showcases for raylib, SDL, cogweb, and ImGui are collected
in [libraries/](libraries/README.md). Their project versions are `0.1.0`; each
dependency is pinned to the latest library package version.

## Learning path

| Example | What it demonstrates |
| --- | --- |
| [hello.cog](hello.cog) | Imports, main, output, and explicit I/O propagation |
| [command_line_args.cog](command_line_args.cog) | `main(args: str[])` and forwarding arguments after `--` |
| [bindings.cog](bindings.cog) | const/var, normal types, reassignment, and interpolation |
| [functions.cog](functions.cog) | Tails, defaults, named arguments, structs, enum constructors, and trailing commas |
| [comptime_factorial.cog](comptime_factorial.cog) | Compile-time functions, recursion, and a materialized scalar constant |
| [comptime_types.cog](comptime_types.cog) | Type values passed through a compile-time function and substituted into nested annotations |
| [comptime_structs.cog](comptime_structs.cog) | Compile-time struct constructors, nested results, and field defaults |
| [comptime_errors.cog](comptime_errors.cog) | Compile-time nil/Error fallbacks and propagation across function boundaries |
| [extensions.cog](extensions.cog) | Type aliases, inherent methods, and compile-time registered extensions |
| [generics_traits.cog](generics_traits.cog) | Comptime type parameters, trait implementations, and generic standard-library operations |
| [package_include](package_include/) | Compile-time package inclusion and explicit package selection for duplicate module names |
| [control_flow.cog](control_flow.cog) | Ranges, while, break/continue, guards, when, and match |
| [strings.cog](strings.cog) | Literal braces, full interpolation, nesting, and Error handling |
| [tails.cog](tails.cog) | Normal block results, early return, and cleanup snapshots |
| [structs.cog](structs.cog) | Typed/defaulted fields, nullable fields, and reordered constructor arguments |
| [collections.cog](collections.cog) | Shared arrays, read-only aliases, mutation, and sorting |
| [fallbacks.cog](fallbacks.cog) | Distinct lazy nil/Error fallbacks and evaluation counts |
| [propagation.cog](propagation.cog) | !! bypassing local catch, caller handling, and primary/secondary cleanup failures |
| [formatting.cog](formatting.cog) | Shared builders, runtime templates, and immutable snapshots |
| [mathematics.cog](mathematics.cog) | Powers, square roots, rounding, and recoverable domain errors |
| [text_processing.cog](text_processing.cog) | Split, trim, replacement, join, array search, and fresh reversed arrays |
| [byte_encoding.cog](byte_encoding.cog) | ASCII byte conversion, hexadecimal encoding/decoding, and invalid input |
| [base64_encoding.cog](base64_encoding.cog) | Padded Base64 encoding/decoding, UTF-8 conversion, and malformed input |
| [streams.cog](streams.cog) | Byte buffers, managed stdout, write_all, and flush |
| [async_concurrency.cog](async_concurrency.cog) | FIFO task scheduling, explicit yielding, and structured joining |
| [async_stdio.cog](async_stdio.cog) | Concurrent stdin read and stdout write while another task continues |
| [async_timeout.cog](async_timeout.cog) | Deadline races, `Error::Timeout`, and source-task ownership |
| [async_select.cog](async_select.cog) | Selecting the first completed spawned task and observing results explicitly |
| [async_select_many.cog](async_select_many.cog) | Selecting by index from an array of spawned tasks |
| [async_filesystem.cog](async_filesystem.cog) | Concurrent file writes, asynchronous copy, rename, and text reads |
| [async_lines_and_append.cog](async_lines_and_append.cog) | Appending UTF-8 text and bytes, then reading a file as lines |
| [async_file_handle.cog](async_file_handle.cog) | Exact and read-to-EOF operations, write, flush, and seek on a managed file handle |
| [async_partial_write.cog](async_partial_write.cog) | A partial managed-file write and reading back the accepted byte count |
| [async_methods.cog](async_methods.cog) | Concurrent async receiver, generic, associated, and statically dispatched trait methods |
| [async_cancellation.cog](async_cancellation.cog) | Structured cancellation, `defer` cleanup, and `Error::Cancelled` handling |
| [async_process_timeout.cog](async_process_timeout.cog) | Process-specific deadline, Unix process-group termination, and `Error::Timeout` (`sleep`) |
| [async_process_cancel.cog](async_process_cancel.cog) | Task cancellation terminates a Unix process group and reaps its direct child (`sh` and `sleep`) |
| [async_process_input.cog](async_process_input.cog) | Asynchronously sends bytes to child stdin while capturing stdout and stderr (`/bin/cat`) |
| [async_network.cog](async_network.cog) | Concurrent TCP request/response operations on the host worker pool |
| [async_tcp_stream.cog](async_tcp_stream.cog) | Persistent TCP handles, overlapping async read/write, and write half-close |
| [async_tcp_server.cog](async_tcp_server.cog) | Async listener/accept, exact-length request reads, and response reads through EOF |
| [async_tcp_cancel.cog](async_tcp_cancel.cog) | Cancelling pending TCP accept and read-to-EOF operations |
| [async_plugin.cog](async_plugin.cog) | Two native async operations waking the cooperative scheduler concurrently |
| [async_plugin_cancel.cog](async_plugin_cancel.cog) | Native async cancellation and joining its worker before plugin state is freed |

Most examples complete without input. Host output can still fail; the examples use
explicit propagation or lexical handling where appropriate.

The network example uses `async_tcp_echo_server.py` as a loopback peer. Start the
server in another terminal; it accepts two concurrent requests and delays each
response briefly so the overlapping work is visible.

```sh
python3 examples/async_tcp_echo_server.py
# In a second terminal:
cog run examples/async_network.cog
```

The persistent stream example uses `async_tcp_stream_server.py` on port 18766.
Start it in another terminal, then run:

```sh
python3 examples/async_tcp_stream_server.py
# In a second terminal:
cog run examples/async_tcp_stream.cog
```

`async_tcp_server.cog` runs both sides itself: it starts an async listener and a
client task, accepts the connection, then exchanges bytes over persistent handles.
The client reads its response through EOF after the server half-closes its write
side. No external peer process is needed.

The native plugin example compiles a small POSIX C plugin. It prints `start`
twice before either `finish`, then the cancellation companion stops and joins a
five-second worker early:

```sh
cc -std=c11 -O2 -fPIC -shared -pthread -Iinclude \
  examples/native_async_plugin.c -o examples/libnative_async_plugin.so
cog run examples/async_plugin.cog --plugin examples/async_plugin.kdl
cog run examples/async_plugin_cancel.cog --plugin examples/async_plugin.kdl
```

## Interactive programs

| Example | Input and behavior |
| --- | --- |
| [read_number.cog](read_number.cog) | Enter an integer with optional surrounding whitespace; conversion/I/O errors propagate |
| [pyramid.cog](pyramid.cog) | Enter a positive height; build and print a centered star pyramid |
| [number_detective.cog](number_detective.cog) | Guess a secret number in 1..100; commands: stats and quit |
| [async_stdio.cog](async_stdio.cog) | Enter a line while an async task continues to write to stdout |
| [cog_pass.cog](cog_pass.cog) | Generate a password with OS entropy; the optional CLI length defaults to 22 and must be 4..256 |
| [todo.cog](todo.cog) | Add, complete, remove, list, and count notes; load and save them as plain text in `todo.txt` |

Number detective accepts an empty name/attempt limit as Player/6. Invalid guesses
do not consume attempts. It demonstrates shared history, patterns/guards, named
arguments, nullable defaults, fallbacks, lexical handlers, propagation, and cleanup.
The statistics helper takes its array through an unannotated parameter, so it casts
element reads at that dynamic boundary. The command parser is also dynamic, so its
match keeps a wildcard arm.

## Projects and host processes

- [Modules](modules/README.md): two source modules and public visibility.
- [Library package](library_package/README.md): check a library with `package.kdl` and no `main`.
- [Multiple targets](targets/README.md): select separate client/server entry points.
- [processes.cog](processes.cog): async-by-default process capture and the explicit blocking `run_sync` variant.
  It requires `echo` on PATH; it uses no shell. The full shell API is Unix-specific.
- [async_network.cog](async_network.cog): TCP exchange example; run its local peer with
  `python3 examples/async_tcp_echo_server.py` in another terminal.
- [async_tcp_stream.cog](async_tcp_stream.cog): persistent TCP example; run its local
  peer with `python3 examples/async_tcp_stream_server.py` in another terminal.
- [async_tcp_server.cog](async_tcp_server.cog): async loopback listener and client in
  one Cog process.

```sh
cog run examples/modules
cog run examples/targets --target server
cog run examples/processes.cog
cog run examples/async_filesystem.cog
cog run examples/async_concurrency.cog
cog run examples/async_timeout.cog
cog run examples/async_cancellation.cog
cog run examples/async_process_cancel.cog
cog run examples/async_tcp_cancel.cog
cog run examples/async_select.cog
cog run examples/async_select_many.cog
```

Local modules take precedence over standard modules with the same name. This is
why the process example is named `processes.cog`. See [projects](../docs/PROJECT_FORMATS.md).

## Validation

The [build-input project](build_inputs/main.cog) embeds text and binary data from
its package root. Run `cog run examples/build_inputs`; its bytecode continues to
work after the input file is removed.

```sh
cargo test --locked --offline --test examples --test documentation
```

Tests compile the programs, round-trip their bytecode, execute deterministic and
interactive cases, and check both project examples. Process checks are platform-aware
in the project/runtime suites. See the [language reference](../docs/LANGUAGE.md)
and [error contracts](../docs/ERRORS.md) for the implemented semantics.
