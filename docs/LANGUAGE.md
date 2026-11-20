# Language reference

This page describes implemented syntax. See [errors and cleanup](ERRORS.md),
the [standard library](STANDARD_LIBRARY.md), and [projects](PROJECT_FORMATS.md)
for their full contracts. [Future specifications](SEMANTIC_CONTRACTS.md) are separate.

## Source and literals

Sources are UTF-8. Identifiers use ASCII letters, digits, and `_`, starting with a
letter or `_`; names are case-sensitive. `//` and `/* ... */` introduce comments.
Comments are currently discarded rather than retained as documentation metadata.
A shebang is allowed on the first line and preserves diagnostic line numbers.

Literals include decimal integers (`42`), hexadecimal integers (`0x2A`), decimal
floats (`3.5`, `7e-2`), `true`, `false`, `nil`, character and string literals,
arrays, and records.
Underscores may separate digits, as in `1_000` or `0xCA_FE`; they cannot appear at
the start or end of a number or next to a decimal point or exponent marker. Integer
literals through `i64::MAX` are `int`; larger positive literals through `u64::MAX`
are `uint`; values outside those ranges are `bigint`. `int` and `uint` are checked
64-bit values, `bigint` is exact and arbitrary precision, and floats use binary64.
Negative values down to `i64::MIN` are `int`; smaller values are `bigint`.
Character literals use single quotes and contain exactly one Unicode scalar, as
in 'A', 'λ', or '\n'. This counts Unicode scalar values rather than grapheme
clusters. Character literals support `\'`, `\\`, `\n`, `\t`, `\r`, `\0`, and
the ASCII-only `\xNN` escape. Write other Unicode scalars directly in UTF-8;
the `\u{...}` escape form is not supported.

Ordinary statements end with `;`. A newline alone does not terminate an expression.
Existing comma-separated lists allow a trailing comma, including arguments,
parameters, fields, array/record literals, patterns, match arms, and imports.
Empty elements such as `f(,)` or `[1,,]` are errors.

## Directives and declaration attributes

Compiler directives start with `@!`; declaration attributes start with `@` and
can be composed before a declaration. The built-in directive
`@!include("std")` makes the standard package available. Other aliases must be
declared in `dependencies.kdl` and are included with `@!include("alias")` in the
project entry point.

The implemented attributes are `@comptime` on functions, constants, and
extensions, `@public` on declarations that can be public, `@async` on
functions, and `@inline` on runtime functions. For example,
`@public @comptime fn ...` combines visibility and compile-time evaluation,
while `@inline fn ...` asks the compiler to prefer inlining eligible calls.
The compiler may decline the hint when a function is unsupported or would
exceed its code-size budget. `@inline("never")` disables inlining for that
function. `@comptime const NAME = ...` declares a compile-time value, including
inside a function. Unknown attributes and unsupported attribute arguments are
errors; `@repr` and `@derive` do not have semantics yet.
`@comptime var` is not supported because compile-time mutable bindings and
reassignment do not yet have defined semantics. A `var` declared inside an
`@comptime fn` is evaluated
at compile time, but the evaluator currently does not support reassignment.

Bare `include` and `comptime` declarations are not part of the language syntax.
Use `@!include("alias")` and `@comptime const` instead.

## Bindings and mutation

```cog
@!include("std")
use io;
use sort;

fn main() -> nil {
    const title: str = "Scores";
    var scores = [3, 1, 2];
    const view = scores;
    sort::quick(scores);
    scores.append(4);
    io::println($"{title}: {view}")!!;
}
```


`const` forbids reassignment and mutation through its binding/access path, including
passing that path to a mutating function. Runtime initialization is allowed.
`var` permits reassignment and object mutation. A local `var` must have a reachable
mutating use: reading it or writing only after a return/in a constant-false branch
does not justify `var`. Mutating helper calls count; the compiler infers transitive
parameter/global effects.

Scalar assignment copies a value. Array/record assignment shares an object without
moving or invalidating the original binding. A const alias observes changes through
an existing mutable alias; creating it does not freeze the object globally.

## Function values and closures

Cog supports function types, named function values, and synchronous closures.
Closure captures are always explicit. `[name]` copies the value when the closure
is created. For arrays, records, structs, and other reference values, that copies
the reference, so the closure sees changes made through existing mutable aliases.
The captured binding is read-only, and a closure cannot mutate the object through
that capture. `[var name]` shares a mutable local through a cell, allowing the
closure to reassign the binding or mutate the referenced object. Nested closures
must list their own captures; a mutable capture can be forwarded only through a
mutable capture in the enclosing closure. An omitted capture list means the
closure captures no locals.

```cog
fn add_bias(bias: int) -> fn(int) -> int {
    const add = fn[bias](value: int) -> int { value + bias };
    add
}

fn main() -> int {
    var total = 0;
    const next = fn[var total]() -> int {
        total += 1;
        total
    };
    (next)();
    (next)()
}
```

Reference captures keep the same object alive after the creating function
returns. For example, a returned reader of an array observes updates made through
another mutable alias, while it cannot update that array itself through its
read-only capture. Function values can also be captured and called from another
closure. Captured references participate in the VM's ordinary cycle collection.

Function types use `fn(ArgumentTypes...) -> ResultType`. Closure parameters and
results require annotations in this release. Named function values must have
fully annotated parameters and result types; async, generic, and defaulted
functions cannot be used as values. Indirect calls are positional and require an
exact argument count. JIT can OSR into synchronous closure bodies with captures;
captured values remain rooted, and unsupported operations and dynamic callback
dispatch resume in the VM. Indirect calls do not yet carry mutation-effect types,
so reference arguments are treated as potentially mutated and reference-valued
results are read-only. Use `.clone()` to obtain an independent writable copy
where the value supports cloning.
Generic type parameters can be inferred through callback signatures such as
`fn(T) -> U`, as used by `array::map`.

Aliasing cannot increase mutation capability. `var b = const_array` creates a
reassignable binding whose current object remains read-only through that path.
Reassigning `b` to a fresh array can restore writable access. The analysis follows
returned aliases and containers conservatively; complex safe calls can be rejected.
`.clone()` creates an independent copy of arrays and tuples whose elements are
structurally cloneable; nested arrays and tuples are copied recursively, preserving
repeated references inside the copied graph. Strings and scalar values are immutable
or copied by value. A custom nominal type can implement `Clone` for generic use.
Composing custom implementations into automatic array or tuple cloning is outside
the Beta contract.

## Compile-time evaluation

`@comptime const` introduces an immutable compile-time binding. Its initializer
is evaluated before runtime lowering, and its result is materialized for runtime
use.

```cog
@!include("std")
use io;

@comptime const BASE: int = 3;
@comptime const VALUE: int = BASE * 2;

fn main() {
    @comptime const LOCAL: int = VALUE + 1;
    io::println(LOCAL);
}
```


The current evaluator supports scalar literals, earlier `comptime` values,
arithmetic, comparisons, boolean operators, unary operators, and string
concatenation. It can also evaluate struct constructors, including nested
constructors and field defaults. An `@comptime fn` can call other `@comptime fn`s and
can use scalar or struct parameters, named/default arguments, local `const`/`var`
bindings, `return`, and `if` blocks. Calls are evaluated before runtime lowering;
their scalar or struct results are materialized for runtime use, and the function
itself is omitted from the runtime artifact.

```cog
@!include("std")
use io;

@comptime fn factorial(value: int) -> int {
    if value <= 1 { return 1; }
    return value * factorial(value - 1);
}

@comptime const FACTORIAL: int = factorial(5);

fn main() {
    io::println(FACTORIAL);
}
```


Runtime values and ordinary functions are unavailable during compile-time
evaluation. Struct fields cannot yet be inspected by the evaluator. Division by
zero and integer overflow produce Error values that a `when`/`match` pattern can
inspect inside a comptime function. `try`/`catch` can handle these errors in a
comptime function, and `!!` propagates through comptime calls until a caller's
lexical handler or the binding root. Using `!!` directly at a comptime binding
root is rejected because there is no enclosing comptime function to receive the
propagation. An Error reaching the root without a handler is reported as a
diagnostic.
The lazy `??` and `?!` operators are supported, including in compile-time functions;
`!!` propagation bypasses them. Native calls, mutation statements, loops, array,
tuple, and enum values, along with tuple destructuring, are not supported as
comptime data yet. Type values have a separate, limited form: a top-level binding
annotated `: type` can name a primitive, struct, enum, array, tuple, or nullable
type. A type value can also pass
through an `@comptime fn` whose parameter and result are annotated `type`. Bindings
are erased after their types are substituted into annotations, so they have no
runtime value.

```cog
struct Point { x: int, y: int }
@comptime fn identity_type(value: type) -> type { value }
@comptime const Scalar: type = int;
@comptime const Number: type = identity_type(Scalar);
@comptime const Grid: type = Number[][];

fn first(grid: Grid) -> Scalar { grid[0][0] }
fn main() {}
```

Compile-time type bindings are currently global. Runtime functions can declare
inferred compile-time type parameters; these are described below and are erased
from the runtime call. Compile-time value parameters and type parameters on
`@comptime fn` are not supported.

## Types and casts

Available annotations include `int`, `uint`, `bigint`, `float`, `bool`, `char`, `str`, singleton
`nil`, user-defined struct/enum names, homogeneous dynamic arrays (`T[]`),
fixed-shape tuples (`(T, U)`), and nullable forms such as `int?`. These types
compose recursively, so `int[][]` is an array of arrays. Error is an orthogonal
computational state, not an implicit member of the annotated normal type.

```cog
@!include("std")
use io;

fn default_count(value: int?) -> int {
    value ?? 3
}

fn main() -> nil {
    const maybe: int? = 10;
    const count: int = maybe as int;
    io::println(count + default_count(nil))!!;
}
```


A non-nil initializer does not narrow declared `int?` to `int`. Control-flow
checks can refine nullable values on paths that prove nil is absent, including a
terminating nil arm and a positive type-pattern arm. Reassignment and calls that
may mutate the binding invalidate those facts. Use an explicit cast or nil fallback
when no proof is available. An annotation of `int` rejects normal nil.

`as int`, `as uint`, `as bigint`, `as float`, `as bool`, `as char`, and `as str` request
conversions. They do not require try/catch; failed conversion produces Error.
Float-to-integer conversions truncate toward zero and check bounds; negative values
cannot convert to `uint`. `int` and `uint` convert to `bigint` exactly;
`bigint as int` and `bigint as uint` check destination bounds. `float as bigint`
is not supported. String-to-number conversion can fail;
string-to-bool accepts only `"true"` and `"false"`. `str as char` requires
exactly one Unicode scalar. `char as str` creates a one-scalar string.
Converting an Error propagates it
rather than rendering it. Characters compare by scalar value and do not support
arithmetic.

Local bindings infer available normal types. Unknown write constraints can leave a
var dynamic; unannotated global var is currently dynamic. Crossing from a dynamic
value to an annotated type requires `as`. An unannotated parameter or result may be
dynamic, but arrays carry their element-type invariant on the shared object itself.
Known mismatches are rejected statically; mutations through dynamic code are checked
at runtime and return `TypeMismatch` without changing the array. Empty arrays need an
element type from an annotation or an unambiguous use. Inferred comptime type
parameters on runtime functions and parameterized trait bounds are supported as
described below. `Task[T]` is the async result-handle type. Callable types and an
explicit `any` annotation are not implemented.

## Functions, arguments, and results

```cog
@!include("std")
use io;

fn connect(host: str, timeout: int = 10, retry: bool = true,) -> str {
    $"{host}: timeout={timeout}, retry={retry}"
}

fn main() -> nil {
    io::println(connect("localhost", retry: false,))!!;
    io::println(connect(retry: true, host: "example", timeout: 20,))!!;
}
```


Positional arguments fill parameters left to right and must precede named arguments.
Named arguments may be reordered. Unknown names, duplicate parameter slots, excess
arguments, and missing required arguments are compile errors. **Nullable does not
mean optional:** `value: int?` still requires an argument; `value: int? = nil` does not.

Explicit arguments evaluate once in written order, then omitted defaults evaluate
in declaration order in the declaration's environment. Defaults do not capture
caller locals. Struct and enum payload constructors use these same rules. Native
functions and existing receiver methods accept positional arguments only. Extension
method calls may omit trailing parameters with defaults, evaluated after the receiver
and explicit arguments.

The last expression of a block without `;` is its result. A block without a tail
has normal result nil. This is shared by functions, if/when/match, and try/catch.
`expression;` discards the normal result while preserving lexical Error observation.

```cog
fn square(value: int) -> int { value * value }
fn empty() -> nil { 7; }
fn positive(value: int) -> int {
    if value < 0 { return 0; }
    value
}
fn main() -> nil {
    const result: int = square(positive(-2));
}
```

`return value;` exits early; `return;` returns nil. `-> T` checks every normal exit,
including explicit returns and tails. A nonreturning path needs no dummy result.
An unannotated function result remains dynamic; general function inference is
pending. Return/tail values are saved before cleanup using existing shared/value
semantics, not deep copying. Normal loop/defer body results are discarded.

`@async fn load() -> T` returns a cold `Task[T]`; async inherent, associated, and
trait methods use the same call contract. Trait implementations must match the
declaration's async marker. `await task` is allowed only in an async function and
produces `T`. `spawn(task)` queues a structured child of the
current async function. `yield_now()` returns a cold `Task[nil]`; awaiting it gives
other ready tasks a turn. `cancel(task)` returns `true` when it accepts a cancellation
request and `false` when the task is already complete or already requested. Async
`task::select(tasks)` reports the index of the first completed task in an array of
already spawned siblings; `task::select2(first, second)` is its two-task form. Both
leave result observation to `await` and do not cancel their inputs. Async `main` is
awaited by the host. The executor is cooperative. Async timers, filesystem
operations, task timeouts, and process capture can wait without blocking it; other
synchronous I/O and plugin calls may still block. See the [async task contract](ASYNC_DESIGN.md)
for cancellation, resumable awaits, scheduling, and current limits.

## Operators and assignment

Arithmetic uses `+`, `-`, `*`, `/`, `%`; comparisons use `==`, `!=`, `<`, `<=`, `>`,
`>=`. `!` is boolean negation, `^` boolean XOR. `&&` and `||` short-circuit.
Integer overflow produces Error. `<<` and `>>` are sequence mutations, not bit shifts.

Supported compound assignments are `+=`, `-=`, `*=`, `/=`, `%=`, `&&=`, `||=`, and
`^=`. An assignment receiver/index evaluates once, including compound assignments.
`<<=` and `>>=` are not supported.

Arithmetic binds more tightly than sequence operations, which bind more tightly
than comparisons. `&&` binds more tightly than `||`. `??` and `?!` have equal
precedence below `||` and associate to the right. Use parentheses for mixed
expressions and for composing control-block expressions:

```text
a ?? b ?! c   = a ?? (b ?! c)
a ?! b ?? c   = a ?! (b ?? c)
a ?? b || c  = a ?? (b || c)
```

## Control flow

```cog
@!include("std")
use io;

fn main() -> nil {
    var total = 0;
    for i in 0..5 {
        total += i;
    }
    for value in [2, 3, 4] {
        total += value;
    }
    for id in 0..3, name in ["First", "Second", "Third"] {
        io::println($"{id}: {name}")!!;
    }
    var count = 3;
    while count > 0 {
        io::println(count)!!;
        count -= 1;
    }
    for i in 5..0 step -2 {
        if i == 5 { continue; }
        if i == 1 { break; }
        io::println(i)!!;
    }
    const label = if total > 5 { "large" } else { "small" };
    io::println(label)!!;
}
```


`if` takes a boolean condition and supports `else`/`else if`. A value-producing
conditional uses block tails; known normal branch types must be compatible.
`while` loops while its condition is true. `for name in start..end [step value]`
uses an exclusive end bound with `int` or `uint` bounds; the default step matches
the bound type, and zero step produces Error. Unsigned ranges ascend.
Comma-separated `for` clauses iterate in parallel and stop when the first source
is exhausted. For each clause, Cog evaluates its source and initializes its
iterator before moving to the next clause; a custom `into_iter()` call is part of
that clause's initialization. Each clause may be a range, an array/string
iteration, or a user-defined iterator described below. A collection
clause binds either one name or a tuple pattern whose shape must match the element
tuple. Patterns may be nested, and their names are available only in the loop body.
For example, `for (key, value) in map.entries()` binds both positions of each
entry tuple. This works for any array of tuples, independently of `Map`.
`break;` exits the innermost loop; `continue;` advances to its next iteration, and
both run cleanup exited by the transfer. `for item in
collection` evaluates an array or string expression once. Arrays yield their
elements; strings yield one `char` per Unicode scalar. Appending or prepending to
the iterated array through any alias invalidates the iterator and produces
`Error::InvalidArgument` when it next advances. Replacing an existing element does
not invalidate it. Iterate an ordered-map snapshot with
`for entry in items.entries()`; each entry is a `(key, value)` tuple, with fields
available as `entry.0` and `entry.1`. Loop labels are not implemented.

## Patterns: when and match

```cog
@!include("std")
use io;

enum Message { Data(id: int, text: str), Empty }

fn main() -> nil {
    const message = Message::Data(text: "Cog", id: 7);
    when message is Message::Data(id, text) if id > 0 {
        io::println($"{id}: {text}")!!;
    }
    const label = match message {
        Message::Data(id, text) if id > 0 => text,
        Message::Empty => "empty",
        _ => "other",
    };
    io::println(label)!!;
}
```


`when value is Pattern [if guard] { ... } [else { ... }]` handles one conditional
pattern branch. `match value { Pattern [if guard] => result, ... }` selects among
arms and requires exhaustive coverage under the current checker; use `_` when
the scrutinee remains dynamic. Both evaluate the scrutinee once and share pattern
semantics. Arms may contain blocks or expressions and can return a value.

Patterns support `_`, primitive int/uint/bigint/float/bool/char/str/nil types, declared enum
types, character and string literals, and variants. A character literal matches
one exact Unicode scalar. A string literal pattern matches exact text and does
not make a string match exhaustive; include `_` for other values. A variant without
`(bindings)` ignores its payload; bindings require
exact positional arity and exist only in the selected arm. Binding names do not
select payload fields. Alternatives use `|`; guards run after a pattern matches.
Enums have nominal identities, including package/module identity. Struct patterns
and struct type tests such as `when value is Person` are rejected because structs
use runtime record storage without nominal runtime identity. Both are outside the
Beta contract.

`is` is only the separator inside `when`; `if x is int` and `const b = x is int`
are errors. Patterns inspect values without casting or narrowing declared nullable
types. Explicit Error patterns inspect errors before a surrounding catch.

The existing `when var name = expression { ... } else { ... }` form selects success
when the result is not Error, including nil. Its binding is immutable despite the
written `var`; this is a current limitation, not a nil check.

## Strings

```cog
@!include("std")
use io;
use str;

fn main() -> nil {
    const name = "Cog";
    io::println("Hello, {name}; {} {{}}")!!;
    io::println($"Hello, {name}; answer={20 + 22}")!!;
    io::println($"{{{6 * 7}}}")!!;
    io::println($"nested={$"value={1 + 2}"}")!!;
    const count = str::trim(" \t42\r\n") as int;
    io::println(count)!!;
}
```


Plain `"..."` preserves braces literally. `$"...{expression}..."` explicitly
interpolates full expressions, including calls, casts, records, and control flow.
Inside this form, `{{`/`}}` produce literal braces. Strings inside holes use ordinary
quotes without escaping for the outer string. Holes evaluate left to right once
and render immediately, so later mutation does not alter earlier rendered text.

Both forms decode `\"`, `\\`, `\n`, `\r`, `\t`, `\0`, and ASCII hexadecimal
escapes `\xNN` exactly once. A hexadecimal escape requires exactly two hex digits
and a value in `00`..`7F`; for example, `"\x1B[2J\x1B[H"` contains terminal
escape sequences. Use `io::print` to emit them without a trailing newline.
Character literals also accept `\'` for a quote; their other escapes follow the
same rules.
`"\\x1B"` contains the literal text `\x1B`. Unknown or malformed escapes and
unmatched/empty interpolation holes are errors. UTF-8 and physical multiline
strings are supported; write non-ASCII characters directly as UTF-8 text.
Nested interpolation is limited to 64 lexer levels.
Raw strings, Unicode escapes, and `f"..."` are not supported.

Nil renders as `nil`; computational Error remains Error rather than silently
becoming text. String construction respects runtime buffer limits. See
[errors](ERRORS.md) for observation and [text APIs](STANDARD_LIBRARY.md) for trimming
and formatting.

## Arrays, records, structs, and enums

```cog
@!include("std")
use io;

struct Point { x: int, y: int = 0 }
struct Marker;
struct UserId(int);
struct Position(float, float);
enum Status { Ready, Failed(code: int) }

fn main() -> nil {
    var values: int[] = [1, 2];
    values << 3;
    0 >> values;
    values[1] = 9;
    io::println(values)!!;

    var pair: (int, str) = (10, "hello");
    pair.0 = 11;
    io::println($"{pair.0}: {pair.1}")!!;

    var record = { name: "Cog", count: 1 };
    record.count = 2;
    io::println(record["name"])!!;

    const point = Point(10, y: 20);
    const marker = Marker();
    const user = UserId(42);
    const position = Position(1.5, 2.5);
    const status = Status::Failed(code: 7);
    io::println($"{point.x}, {point.y}; {user.0}; {position.0}, {position.1}; {status.code}")!!;
}
```

Structs with named fields use `{}` declarations. An empty struct can use either
`struct Marker {}` or the shorter `struct Marker;`; both are constructed with
`Marker()`. Tuple structs declare one or more typed positional fields, such as
`struct UserId(int);` or `struct Position(float, float);`. Their constructors
take positional arguments, and fields use the same numeric access as tuples
(`user.0`, `position.1`). Tuple structs keep their own nominal type, so `UserId`
is distinct from `int` and from other tuple structs with the same field types.


Arrays use zero-based indexing and share mutable storage. Out-of-bounds access
produces Error. `array << value`, `value >> array`, `.append(value)`, and
`.prepend(value)` mutate and return the same array. Each adds one element; an array
argument becomes a nested element rather than concatenation. Both sequence
operators currently associate left to right; write `x >> (y >> array)` for a
prepend chain. Receiver methods take positional arguments. Assigning an Error to an
array element, tuple position, or record field stores it at that leaf; the aggregate
keeps its normal type. An Error in the assignment receiver or index propagates
without mutation.

Array literals infer a homogeneous element type when possible; incompatible element
types are rejected. An empty array needs a type context, such as an annotation, a
typed parameter or result, assignment to an existing typed array, or a known
append/prepend operation. Empty nested arrays can inherit their type from an
unambiguous sibling element. Otherwise the compiler reports that it cannot infer the
element type; there is no implicit `any[]`.

Element types are checked on indexing, writes, append/prepend, nested arrays, aliases,
and typed function boundaries. Mutable arrays are invariant. An unannotated function
parameter remains dynamic and does not specialize the function, but the shared array
object keeps its actual recursive element type so incompatible runtime mutations
return `TypeMismatch` without changing it. `.empty_like()` creates a fresh empty array
with the receiver's element type.

Tuples use `(first, second, ...)` with at least two elements. Their length and each
position's type are fixed; use numeric member access such as `pair.0` and `pair.1`.
Numeric accesses can be chained, such as `nested.0.1`, including inside string
interpolation. Each position is an integer index; exponent notation is not a
position index.
Positions can be updated when the tuple is mutable, and assigning a tuple value
shares its reference-like storage. Empty/single-element tuple forms are unsupported.
Recursive tuple binding patterns are supported in local `const` and `var`
declarations and collection `for` clauses. The right-hand side of a declaration is
evaluated once, and a type annotation applies to the complete tuple, for example
`const (name, age): (str, int) = ("Ada", 37);`. Function parameter destructuring
remains unsupported. `T[N]` and general user-container operators remain future work.
`len(array)` or `array.len()` returns the element count; `len(str)`
counts Unicode scalars, not UTF-8 bytes.

User-defined collection types can participate in `for` loops through the
`iter::IntoIterator[Iter, Item]` and `iter::Iterator[Item]` traits. Import the
standard `iter` module, implement `into_iter()` on the collection, then implement
`next()` and `current()` on the returned iterator type. `next()` advances the
iterator and reports whether an item is available; the loop calls `current()`
only after `next()` returns `true`.

```cog
@!include("std")
use io;
use iter;

struct Counter { limit: int, value: int }
struct CounterIter { limit: int, value: int }

extend Counter with iter::IntoIterator[CounterIter, int] {
    fn into_iter(self) -> CounterIter {
        CounterIter(self.limit, self.value)
    }
}

extend CounterIter with iter::Iterator[int] {
    fn next(mut self) -> bool {
        if self.value >= self.limit { return false; }
        self.value += 1;
        true
    }
    fn current(self) -> int { self.value }
}

fn main() {
    const counter = Counter(3, 0);
    for value in counter { io::println(value)!!; }
}
```

The compiler resolves these trait methods statically. Arrays, strings, and
dynamic values keep their existing runtime iteration paths. User-defined
iteration requires a statically known concrete collection type; a generic
function cannot currently iterate a parameter through an `IntoIterator` bound.
The effects of iterator methods and the capabilities of returned items participate
in mutability analysis, including for zipped loops.
When JIT is enabled, custom iterator calls may resume in the VM; native execution
of these methods is not guaranteed.

Runtime functions declare compile-time type parameters in square brackets, such as
`fn identity[T: type](value: T)`. Cog infers each type from statically known runtime argument
types, checks repeated uses for consistency, and creates a runtime specialization
for each type combination. Type parameters are erased from the runtime call.
Inference works recursively through arrays and tuples. A type parameter can have
trait bounds after `+`; bounds are checked during specialization. Explicit calls
such as `identity[int](5)` supply every generic argument. Partial explicit arguments
and result-only inference for ordinary generic functions are unsupported; empty
values alone cannot determine a type parameter. A generic type-associated function
can infer its type parameters from a direct declared result context, such as
`const map: Map[int, str] = Map::new();`. A function may use the inferred
type in its parameter, result, local annotations, and method calls.

Structs can also declare compile-time type parameters. Their types are inferred
from constructor arguments, and explicit type annotations use `Name[Arguments]`.
Each concrete type combination receives a deterministic specialization. For
example:

```cog
struct Box[T: type] { value: T }

fn unwrap[T: type](item: Box[T]) -> T { item.value }

fn main() {
    const number: Box[int] = Box(42);
    const nested: Box[Box[int]] = Box(number);
    unwrap(nested.value);
}
```

Struct and function specializations share the 256-instance compilation limit.
The compiler rejects inconsistent inferred field types and type arguments that
do not satisfy declared trait bounds. A type parameter that cannot be inferred
from constructor values requires explicit constructor arguments, such as
`Box[int[]]([])`. A result annotation does not infer constructor parameters.
Everything inside a generic parameter list is already compile-time: writing
`comptime` there is an error. Brackets applied to a generic declaration instantiate
it; brackets applied to a runtime value index it, according to lexical scope and
shadowing. Empty brackets after a type form an array type; empty indexing is invalid.

```cog
trait Measure[Factor: type] {
    fn measure(self, factor: Factor) -> int;
}

extend int with Measure[int] {
    fn measure(self, factor: int) -> int { self * factor }
}

fn scaled[T: type + Measure[int]](value: T, factor: int) -> int {
    value.measure(factor)
}

fn main() {}
```

Trait declarations may take compile-time type parameters, for example
`trait Convert[To: type]`. Apply a trait to a concrete type with
`extend Source with Convert[str]`. Trait type parameters are substituted into
method signatures before the implementation is checked. A function's bound may
refer to another inferred type parameter, as in
`fn convert[T: type, U: type + Convert[T]](value: U) -> T`.
`@comptime extend Source with Convert[str]` may use a compile-time type binding for
the target. Only one implementation of a given trait is allowed per target type;
generic implementations over a family of target types are not supported.

`Self` in a trait signature means the implementing type. A trait method must have
`self` first and cannot declare default arguments. Its implementation must match
parameter names and types, result type, and order; names matter because callers can
use named arguments. Bounds are checked at each generic call, and method calls are
resolved statically for the specialized receiver. Trait objects and dynamic dispatch
are not available. Generic `@comptime fn`s and generic trait implementations are
unsupported. Generic structs and generic extension methods are supported for
generic structs and arrays. Declare extension parameters before the target; the
receiver supplies their inferred types:

```cog
struct Box[T: type] { value: T }

extend[T: type] Box[T] {
    fn get(self) -> T { self.value }
    fn map[U: type](self, value: U) -> Box[U] { Box(value) }
}

fn main() {
    const box = Box(42);
    const text = box.map("cog");
    text.get();
}
```

Extension bounds use the same `[T: type + Trait[T]]` syntax as functions and
must express any constraints needed by the method body. Generic extension
methods are statically resolved and monomorphized with a 256-specialization
compilation limit. Instance method type parameters can be inferred from the receiver
and arguments. Associated functions omit `self` and use `Type::method(...)`; their
type parameters can also be inferred from a direct declared result context. A
concrete extension takes precedence over a matching generic extension. Multiple
matching generic extensions are diagnosed as ambiguous. Generic extensions can target array
patterns such as `T[]`; the standard `array` module uses this to provide
`items.swap(left, right)`.

The standard `clone` module defines `Clone` for primitive values and provides the
generic `clone::value[T: type + Clone](value: T)` helper. Arrays and tuples can be cloned
structurally when their elements are supported primitive values or nested arrays and
tuples. A nominal type can supply its own `Clone` implementation for generic use;
automatic composition of that implementation inside an array or tuple is outside
the Beta contract.

Records use named fields and allow `.field` or string-key indexing. Structs declare
fields, optional primitive/nullable annotations, and defaults. Required fields
precede defaulted fields. Constructors use `Point(1)` or `Point(x: 1)`; records use
`{ x: 1 }`. Identifier case does not change parsing. Runtime structs currently use
record storage rather than a complete nominal struct type system.

Enums declare variants with optional typed payloads. Construct a zero-payload
variant with `Status::Ready()`; patterns use `Status::Ready`. Payload constructors
accept named/default arguments under the same binder as functions and structs.

Built-in receiver methods are `.len()`, `.append(value)`, `.prepend(value)`,
`.empty_like()`, `.clone()`, `.to_int()`, `.to_uint()`, `.to_bigint()`, `.to_float()`, `.to_bool()`, `.to_char()`, `.to_str()`,
`.error_kind()`, `.unwrap()`, and `.expect(message)`. Unwrap/expect crash on Error;
they are different from `!!` propagation and are not nil unwrapping.

User methods can be declared in an extension. `extend` targets a named struct or
enum. `@comptime extend` resolves a compile-time type value and can target primitive,
array, and tuple types; ordinary `fn` methods declared there are generated as
runtime methods. Dispatch is checked against the receiver's static type.

```cog
struct Point { x: int }
@comptime const Scalar: type = int;

trait Double {
    fn doubled(self) -> int;
}

extend Point {
    fn shifted(self, amount: int) -> Point {
        Point(self.x + amount)
    }
}

@comptime extend Scalar with Double {
    fn doubled(self) -> int { self * 2 }
}

fn twice[T: type + Double](value: T) -> int {
    value.doubled()
}

fn main() {}
```

Ordinary extension methods are available for concrete nominal and compile-time
resolved types. An `@comptime extend` can also implement a trait for its resolved
target, including a type alias. Trait method lookup is static and implementations are checked against their
declared signatures. Generic specialization is bounded to 256 instances per
compilation; recursive generic calls reuse the same instance.

## Modules and visibility

The project entry point must include the standard package with
`@!include("std")` before source modules can import it. Use `use io;` for
`io::println`, or `use io::println;` for bare `println`.
`@!include("alias")` is a top-level project-entry directive. The `std` alias is
built in; all other directive aliases must appear in `dependencies.kdl`.
An executable may define `fn main()` or `fn main(args: str[])`; in the latter
form, `run` supplies command-line arguments after `--`, excluding the executable
name. No other `main` parameter signature is supported. `--restricted` denies
access to `process::args()` and therefore to this argument list.
`use math::{PI, max as larger};` and `use math::*;` are supported. Only `len` and
`exclusive` are in the ordinary prelude. Imports can also be block-local.

`public` exposes functions, globals, structs, and enums across modules; declarations
are private by default. Modules follow filesystem paths and packages use explicit
dependency aliases. A project entry point may include a dependency with
`@!include("game")`; import a module from that package with
`use random from game as game_random;`. See
[projects and configuration](PROJECT_FORMATS.md) for layouts and runnable projects.
