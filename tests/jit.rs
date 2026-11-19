use cog::runtime::{RuntimeContext, VmCapabilities};
use cog::{interpreter, project};
use std::path::{Path, PathBuf};

const ASCENDING_SOURCE: &str = r#"
@!include("std")
use io;

fn sum(n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        total += index;
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(sum(6000))!!;
}
"#;

const VIRTUAL_STACK_ASSEMBLY: &str = r#"
call main 0
call emit_value 1
pop
halt

.function main 0 0
main:
push 0
store 0
loop:
load 0
push 1
add
dup
swap
push 999
pop
store 0
stack_truncate 0
load 0
push 6000
lt
jnz loop
load 0
ret

.function emit_value 1 0
emit_value:
store 0
call_native io::stdout
load 0
call_native bytes::render
store 1
load 1
push 0
load 1
call_native bytes::len
call_native io::write_all
dup
is_type error
jnz emit_done
pop
call_native io::stdout
call_native io::flush
emit_done:
ret
"#;

const CAPTURED_CLOSURE_SOURCE: &str = r#"
@!include("std")
use io;

fn make_adder(bias: int) -> fn(int) -> int {
    const add = fn[bias](count: int) -> int {
        var index = 0;
        var total = 0;
        while index < count {
            total += bias;
            index += 1;
        }
        total
    };
    add
}

fn main() {
    const add_seven = make_adder(7);
    const add_eleven = make_adder(11);
    io::println((add_seven)(6000))!!;
    io::println((add_eleven)(6000))!!;
}
"#;

const REFERENCE_CAPTURE_FALLBACK_SOURCE: &str = r#"
@!include("std")
use io;

fn make_sum(values: int[]) -> fn(int) -> int {
    const sum = fn[values](count: int) -> int {
        var index = 0;
        var total = 0;
        while index < count {
            total += values[0];
            index += 1;
        }
        total
    };
    sum
}

fn main() {
    const sum = make_sum([7]);
    io::println((sum)(6000))!!;
}
"#;

const USER_ITERATOR_FALLBACK_SOURCE: &str = r#"
@!include("std")
use io;
use iter;

struct Source { limit: int }
struct Cursor { limit: int, value: int }

extend Source with iter::IntoIterator[Cursor, int] {
    fn into_iter(self) -> Cursor { Cursor(self.limit, 0) }
}

extend Cursor with iter::Iterator[int] {
    fn next(mut self) -> bool {
        if self.value >= self.limit { return false; }
        self.value += 1;
        true
    }
    fn current(self) -> int { self.value }
}

fn main() {
    var total = 0;
    for value in Source(6000) { total += value; }
    io::println(total)!!;
}
"#;

const DESCENDING_SOURCE: &str = r#"
@!include("std")
use io;

fn sum(n: int) -> int {
    var total = 0;
    for index in n..0 step -1 {
        total += index;
    }
    return total;
}

fn main() -> nil {
    io::println(sum(6000))!!;
}
"#;

const UINT_SOURCE: &str = r#"
@!include("std")
use io;

fn score(base: uint, n: uint) -> uint {
    const zero: uint = 0;
    const one: uint = 1;
    const three: uint = 3;
    const five: uint = 5;
    const six: uint = 6;
    const threshold: uint = 9223372036854775807;
    var remaining: uint = n;
    var index: uint = zero;
    var total: uint = zero;
    while remaining > zero {
        if base > threshold {
            total += (((index + one) * six) / three) % five;
        }
        index += one;
        remaining -= one;
    }
    return total;
}

fn main() -> nil {
    const base: uint = 9223372036854775808;
    const n: uint = 6000;
    io::println(score(base, n))!!;
}
"#;

const UINT_OVERFLOW_SOURCE: &str = r#"
@!include("std")
use io;

fn trigger(n: uint) -> uint {
    const zero: uint = 0;
    const one: uint = 1;
    const maximum: uint = 18446744073709551615;
    var index: uint = zero;
    var value: uint = zero;
    while index < n {
        if index + one == n {
            value = maximum + index;
        }
        index += one;
    }
    return value;
}

fn main() -> nil {
    const n: uint = 1024;
    io::println(trigger(n))!!;
}
"#;

const FLOAT_SOURCE: &str = r#"
@!include("std")
use io;

fn score(probe: float, n: int) -> float {
    const half: float = 0.5;
    const quarter: float = 0.25;
    const two: float = 2.0;
    var total: float = 0.0;
    var index = 0;
    while index < n {
        total = (((total + half) * two) / two) - quarter + quarter;
        if probe != probe {
            total += half;
        }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    const probe: float = "NaN" as float;
    io::println(score(probe, 6000))!!;
}
"#;

const FLOAT_DIVISION_BY_ZERO_SOURCE: &str = r#"
@!include("std")
use io;

fn trigger(divisor: float, n: int) -> float {
    const one: float = 1.0;
    var index = 0;
    var result: float = 0.0;
    while index < n {
        if index + 1 == n {
            result = one / divisor;
        }
        index += 1;
    }
    return result;
}

fn main() -> nil {
    io::println(trigger(0.0, 1024))!!;
}
"#;

const FLOAT_MOD_SOURCE: &str = r#"
@!include("std")
use io;

fn score(divisor: float, n: int) -> float {
    var index = 0;
    var total: float = 0.0;
    while index < n {
        total += 5.5 % divisor;
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score(2.0, 6000))!!;
}
"#;

const JIT_DIRTY_LOCAL_FALLBACK_SOURCE: &str = r#"
@!include("std")
use io;

fn identity(value: int) -> int {
    return value;
}

fn sum(n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        index += 1;
        total += index;
        total += identity(0);
    }
    return total;
}

fn main() -> nil {
    io::println(sum(6000))!!;
}
"#;

const CHAR_BOOL_COMPARE_SOURCE: &str = r#"
@!include("std")
use io;

fn score(character: char, enabled: bool, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if character > 'a' {
            total += 1;
        }
        if character == '🦀' {
            total += 2;
        }
        if enabled == false {
            total += 10;
        }
        if enabled != false {
            total += 4;
        }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score('🦀', true, 6000))!!;
}
"#;

const MIXED_INT_UINT_COMPARE_SOURCE: &str = r#"
@!include("std")
use io;

fn score(signed: int, unsigned: uint, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if signed < unsigned { total += 1; }
        if signed <= unsigned { total += 2; }
        if signed > unsigned { total += 4; }
        if signed >= unsigned { total += 8; }
        if signed == unsigned { total += 16; }
        if signed != unsigned { total += 32; }
        if unsigned < signed { total += 64; }
        if unsigned <= signed { total += 128; }
        if unsigned > signed { total += 256; }
        if unsigned >= signed { total += 512; }
        if unsigned == signed { total += 1024; }
        if unsigned != signed { total += 2048; }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score(-1, 9223372036854775808, 6000))!!;
    io::println(score(3, 3, 6000))!!;
}
"#;

const MIXED_INTEGER_FLOAT_COMPARE_SOURCE: &str = r#"
@!include("std")
use io;

fn signed_score(signed: int, probe: float, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if signed < probe { total += 1; }
        if signed <= probe { total += 2; }
        if signed > probe { total += 4; }
        if signed >= probe { total += 8; }
        if signed == probe { total += 16; }
        if signed != probe { total += 32; }
        if probe < signed { total += 64; }
        if probe <= signed { total += 128; }
        if probe > signed { total += 256; }
        if probe >= signed { total += 512; }
        if probe == signed { total += 1024; }
        if probe != signed { total += 2048; }
        index += 1;
    }
    return total;
}

fn unsigned_score(unsigned: uint, probe: float, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if unsigned < probe { total += 4096; }
        if unsigned <= probe { total += 8192; }
        if unsigned > probe { total += 16384; }
        if unsigned >= probe { total += 32768; }
        if unsigned == probe { total += 65536; }
        if unsigned != probe { total += 131072; }
        if probe < unsigned { total += 262144; }
        if probe <= unsigned { total += 524288; }
        if probe > unsigned { total += 1048576; }
        if probe >= unsigned { total += 2097152; }
        if probe == unsigned { total += 4194304; }
        if probe != unsigned { total += 8388608; }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    const nan: float = "NaN" as float;
    const infinity: float = "inf" as float;
    io::println(signed_score(9007199254740993, 9007199254740992.0, 6000))!!;
    io::println(signed_score(9007199254740992, 9007199254740992.0, 6000))!!;
    io::println(signed_score(-9223372036854775808, -9223372036854775808.0, 6000))!!;
    io::println(signed_score(9223372036854775807, 9223372036854775808.0, 6000))!!;
    io::println(signed_score(0, nan, 6000))!!;
    io::println(signed_score(0, infinity, 6000))!!;
    io::println(unsigned_score(9007199254740993, 9007199254740992.0, 6000))!!;
    io::println(unsigned_score(9007199254740992, 9007199254740992.0, 6000))!!;
    io::println(unsigned_score(0, -9223372036854775808.0, 6000))!!;
    io::println(unsigned_score(18446744073709551615, 18446744073709551616.0, 6000))!!;
    io::println(unsigned_score(0, nan, 6000))!!;
    io::println(unsigned_score(0, infinity, 6000))!!;
}
"#;

const SMALL_BIGINT_COMPARE_SOURCE: &str = r#"
@!include("std")
use io;

fn signed_score(value: bigint, probe: int, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if value < probe { total += 1; }
        if value <= probe { total += 2; }
        if value > probe { total += 4; }
        if value >= probe { total += 8; }
        if value == probe { total += 16; }
        if value != probe { total += 32; }
        if probe < value { total += 64; }
        if probe <= value { total += 128; }
        if probe > value { total += 256; }
        if probe >= value { total += 512; }
        if probe == value { total += 1024; }
        if probe != value { total += 2048; }
        when value is bigint { total += 4096; }
        index += 1;
    }
    return total;
}

fn unsigned_score(value: bigint, probe: uint, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if value < probe { total += 1; }
        if value <= probe { total += 2; }
        if value > probe { total += 4; }
        if value >= probe { total += 8; }
        if value == probe { total += 16; }
        if value != probe { total += 32; }
        if probe < value { total += 64; }
        if probe <= value { total += 128; }
        if probe > value { total += 256; }
        if probe >= value { total += 512; }
        if probe == value { total += 1024; }
        if probe != value { total += 2048; }
        when value is bigint { total += 4096; }
        index += 1;
    }
    return total;
}

fn float_score(value: bigint, probe: float, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if value < probe { total += 1; }
        if value <= probe { total += 2; }
        if value > probe { total += 4; }
        if value >= probe { total += 8; }
        if value == probe { total += 16; }
        if value != probe { total += 32; }
        if probe < value { total += 64; }
        if probe <= value { total += 128; }
        if probe > value { total += 256; }
        if probe >= value { total += 512; }
        if probe == value { total += 1024; }
        if probe != value { total += 2048; }
        when value is bigint { total += 4096; }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(signed_score(9007199254740993, 9007199254740992, 6000))!!;
    io::println(signed_score(-9223372036854775808, -9223372036854775808, 6000))!!;
    io::println(unsigned_score(9223372036854775808, 9223372036854775807, 6000))!!;
    io::println(unsigned_score(18446744073709551615, 18446744073709551615, 6000))!!;
    io::println(float_score(9007199254740993, 9007199254740992.0, 6000))!!;
    io::println(float_score(9223372036854775808, 9223372036854775808.0, 6000))!!;
    io::println(float_score(-9223372036854775808, -9223372036854775808.0, 6000))!!;
}
"#;

const LARGE_BIGINT_COMPARE_FALLBACK_SOURCE: &str = r#"
@!include("std")
use io;

fn score(value: bigint, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if value > 9223372036854775807 { total += 1; }
        when value is bigint { total += 1; }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score(18446744073709551616, 6000))!!;
}
"#;

const ROOTED_AGGREGATE_LOCALS_SOURCE: &str = r#"
@!include("std")
use io;

fn score(label: str, items: int[], magnitude: bigint, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        total += index;
        when label is str { total += 1; }
        when magnitude is bigint { total += 1; }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score("kept alive across OSR", [11, 12, 13], 18446744073709551616, 6000))!!;
}
"#;

const JIT_TYPE_AND_BOOLEAN_SOURCE: &str = r#"
@!include("std")
use io;

fn score(value: int?, flag: bool, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        if flag ^ false {
            total += 1;
        }
        if !flag {
            total += 2;
        }
        if value == nil {
            total += 4;
        }
        if nil != value {
            total += 8;
        }
        when value is int {
            total += value;
        }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score(2, true, 6000))!!;
    io::println(score(nil, false, 6000))!!;
}
"#;

const JIT_ENUM_TYPE_SOURCE: &str = r#"
@!include("std")
use io;

enum Signal { Hit(value: int), Miss }

fn score(signal: Signal, n: int) -> int {
    var index = 0;
    var total = 0;
    while index < n {
        when signal is Signal { total += 1; }
        when signal is Signal::Hit { total += 2; }
        when signal is Signal::Miss { total += 4; }
        index += 1;
    }
    return total;
}

fn main() -> nil {
    io::println(score(Signal::Hit(7), 6000))!!;
    io::println(score(Signal::Miss(), 6000))!!;
}
"#;

const JIT_ERROR_VARIANT_FALLBACK_SOURCE: &str = r#"
@!include("std")
use io;
use err::Error;

fn main() -> nil {
    var index = 0;
    var total = 0;
    while index < 6000 {
        when (1 / 0) is Error::DivisionByZero { total += 1; }
        index += 1;
    }
    io::println(total)!!;
}
"#;

fn execute_result(source: &str, jit: bool, profile_path: Option<&Path>) -> Result<String, String> {
    let artifact =
        project::compile_source("jit-smoke.cog", source).unwrap_or_else(|error| panic!("{error}"));
    let mut output = Vec::new();
    let mut context = RuntimeContext::new(&mut output, VmCapabilities::default());
    context.configure_jit(jit, profile_path, false);
    interpreter::run_with_context(&artifact, &mut context).map_err(|error| error.to_string())?;
    drop(context);
    Ok(String::from_utf8(output).expect("Cog output should be UTF-8"))
}

fn execute_assembly_result(
    source: &str,
    jit: bool,
    profile_path: Option<&Path>,
) -> Result<String, String> {
    let artifact = cog::parser::assemble(source).map_err(|error| error.to_string())?;
    let mut output = Vec::new();
    let mut context = RuntimeContext::new(&mut output, VmCapabilities::default());
    context.configure_jit(jit, profile_path, false);
    interpreter::run_with_context(&artifact, &mut context).map_err(|error| error.to_string())?;
    drop(context);
    String::from_utf8(output).map_err(|error| error.to_string())
}

fn execute(source: &str, jit: bool, profile_path: Option<&Path>) -> String {
    execute_result(source, jit, profile_path)
        .unwrap_or_else(|error| panic!("branch smoke program should complete: {error}"))
}

fn assert_jit_matches_vm(source: &str, suffix: &str, expected: &str) {
    let profile_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!(
            "jit-branch-profile-{}-{suffix}.json",
            std::process::id()
        ));
    let _ = std::fs::remove_file(&profile_path);
    let disabled_profile_path = profile_path.with_extension("disabled.json");
    let _ = std::fs::remove_file(&disabled_profile_path);

    let vm_output = execute(source, false, Some(&disabled_profile_path));
    assert!(
        !disabled_profile_path.exists(),
        "disabled JIT must not profile"
    );
    let jit_output = execute(source, true, Some(&profile_path));

    assert_eq!(vm_output, expected);
    assert_eq!(jit_output, vm_output);

    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist a profile after compilation"),
    )
    .expect("APE profile should be valid JSON");
    assert!(
        profile["functions"]
            .as_object()
            .expect("profile should include function records")
            .values()
            .any(|function| {
                function["compile_ns"].as_u64().unwrap_or(0) > 0
                    && function["calibrated"].as_bool() == Some(true)
            })
    );

    let _ = std::fs::remove_file(profile_path);
}

fn assert_jit_matches_vm_computed(source: &str, suffix: &str) {
    let expected = execute(source, false, None);
    assert_jit_matches_vm(source, suffix, &expected);
}

fn assert_jit_fallback_matches_vm(source: &str, suffix: &str, expected: &str) {
    let profile_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!(
            "jit-fallback-profile-{}-{suffix}.json",
            std::process::id()
        ));
    let _ = std::fs::remove_file(&profile_path);

    let vm_output = execute(source, false, None);
    let jit_output = execute(source, true, Some(&profile_path));
    assert_eq!(vm_output, expected);
    assert_eq!(jit_output, vm_output);

    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist its compiled fallback plan"),
    )
    .expect("APE profile should be valid JSON");
    assert!(profile["functions"].as_object().is_some_and(|functions| {
        functions
            .values()
            .any(|function| function["compile_ns"].as_u64().unwrap_or(0) > 0)
    }));
    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_conditional_jumps_match_vm_for_true_and_false_paths() {
    assert_jit_matches_vm(ASCENDING_SOURCE, "ascending", "17997000\n");
    assert_jit_matches_vm(DESCENDING_SOURCE, "descending", "18003000\n");
}

#[test]
fn jit_virtual_operand_stack_matches_vm_for_stack_manipulation_and_truncation() {
    let profile_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("jit-virtual-stack-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&profile_path);

    let vm_output = execute_assembly_result(VIRTUAL_STACK_ASSEMBLY, false, None)
        .expect("interpreter should execute the stack-manipulation fixture");
    let jit_output = execute_assembly_result(VIRTUAL_STACK_ASSEMBLY, true, Some(&profile_path))
        .expect("JIT should execute the stack-manipulation fixture");

    assert_eq!(vm_output, "6000");
    assert_eq!(jit_output, vm_output);
    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist its compiled profile"),
    )
    .expect("APE profile should be valid JSON");
    assert!(profile["functions"].as_object().is_some_and(|functions| {
        functions
            .values()
            .any(|function| function["compile_ns"].as_u64().unwrap_or(0) > 0)
    }));

    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_compiles_hot_direct_calls_to_functions_without_loops() {
    let source = r#"
call main 0
call emit_value 1
pop
halt

.function main 0 0
main:
push 0
store 0
push 0
store 1
loop:
load 0
call add_one 1
load 1
add
store 1
load 0
push 1
add
store 0
load 0
push 6000
lt
jnz loop
load 1
ret

.function add_one 1 0
add_one:
store 0
load 0
push 1
add
ret

.function emit_value 1 0
emit_value:
store 0
call_native io::stdout
load 0
call_native bytes::render
store 1
load 1
push 0
load 1
call_native bytes::len
call_native io::write_all
dup
is_type error
jnz emit_done
pop
call_native io::stdout
call_native io::flush
emit_done:
ret
"#;
    let artifact = cog::parser::assemble(source).expect("fixture assembly should be valid");
    let main_entry = *artifact
        .callables
        .keys()
        .next()
        .expect("assembly should declare main");
    let main_end = artifact
        .callables
        .range((
            std::ops::Bound::Excluded(main_entry),
            std::ops::Bound::Unbounded,
        ))
        .next()
        .map_or(artifact.instructions.len(), |(entry, _)| *entry);
    let helper_entry = artifact.instructions[main_entry..main_end]
        .iter()
        .find_map(|instruction| match instruction {
            cog::instruction::Instruction::Call(target, 1) => Some(*target),
            _ => None,
        })
        .expect("main should call the loop-free helper");
    let profile_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("jit-hot-call-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&profile_path);

    let vm_output = execute_assembly_result(source, false, None)
        .expect("interpreter should execute the hot direct-call fixture");
    let jit_output = execute_assembly_result(source, true, Some(&profile_path))
        .expect("JIT should execute the hot direct-call fixture");

    assert_eq!(vm_output, "18003000");
    assert_eq!(jit_output, vm_output);
    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist compiled call profiles"),
    )
    .expect("APE profile should be valid JSON");
    let helper_profile = &profile["functions"][helper_entry.to_string()];
    assert!(
        helper_profile["compile_ns"].as_u64().unwrap_or(0) > 0,
        "expected loop-free helper at {helper_entry} to compile, got {profile}"
    );
    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_typed_parameter_guard_resumes_vm_for_error_values() {
    let source = r#"
call main 0
call emit_value 1
pop
halt

.function main 0 0
main:
push 0
store 0
loop:
load 0
call increment 1
pop
load 0
push 1
add
dup
store 0
push 160
lt
jnz loop
push 1
push 0
div
call increment 1
ret

.function increment 1 0
increment:
store 0
load 0
push 1
add
ret

.function emit_value 1 0
emit_value:
store 0
call_native io::stdout
load 0
call_native bytes::render
store 1
load 1
push 0
load 1
call_native bytes::len
call_native io::write_all
dup
is_type error
jnz emit_done
pop
call_native io::stdout
call_native io::flush
emit_done:
ret
"#;
    let mut artifact = cog::parser::assemble(source).expect("typed guard fixture should assemble");
    let main_entry = *artifact
        .callables
        .keys()
        .next()
        .expect("fixture should declare main");
    let main_end = artifact
        .callables
        .range((
            std::ops::Bound::Excluded(main_entry),
            std::ops::Bound::Unbounded,
        ))
        .next()
        .map_or(artifact.instructions.len(), |(entry, _)| *entry);
    let increment_entry = artifact.instructions[main_entry..main_end]
        .iter()
        .find_map(|instruction| match instruction {
            cog::instruction::Instruction::Call(target, 1) => Some(*target),
            _ => None,
        })
        .expect("main should call increment");
    artifact
        .callables
        .get_mut(&increment_entry)
        .expect("increment should have a call contract")
        .parameter_types = Some(vec![Some(cog::value::RuntimeType::Int)]);
    let profile_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("jit-typed-guard-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&profile_path);
    let run = |jit| {
        let mut output = Vec::new();
        let mut context = RuntimeContext::new(&mut output, VmCapabilities::default());
        context.configure_jit(jit, jit.then_some(profile_path.as_path()), false);
        interpreter::run_with_context(&artifact, &mut context)
            .map_err(|error| error.to_string())?;
        drop(context);
        String::from_utf8(output).map_err(|error| error.to_string())
    };
    let vm_output = run(false).expect("interpreter fixture should complete");
    let jit_output = run(true).expect("typed JIT guard should resume in the interpreter");
    assert_eq!(jit_output, vm_output);

    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("hot increment should compile"),
    )
    .expect("APE profile should be valid JSON");
    assert!(
        profile["functions"][increment_entry.to_string()]["compile_ns"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_osr_reads_distinct_scalar_captures_from_closure_instances() {
    assert_jit_matches_vm(
        CAPTURED_CLOSURE_SOURCE,
        "captured-closure",
        "42000\n66000\n",
    );
}

#[test]
fn jit_reference_capture_fallback_matches_vm() {
    assert_jit_fallback_matches_vm(
        REFERENCE_CAPTURE_FALLBACK_SOURCE,
        "reference-capture",
        "42000\n",
    );
}

#[test]
fn jit_higher_order_array_callback_fallback_matches_vm() {
    assert_jit_fallback_matches_vm(
        r#"
        @!include("std")
        use io;
        use array;

        fn main() {
            var values: int[] = [];
            var index = 0;
            while index < 6000 {
                values << index;
                index += 1;
            }
            const bias = 7;
            const add_bias = fn[bias](value: int) -> int { value + bias };
            const mapped = array::map(values, add_bias);
            io::println(mapped.len())!!;
            io::println(mapped[5999]!!)!!;
        }
        "#,
        "array-callback",
        "6000\n6006\n",
    );
}

#[test]
fn jit_user_iterator_fallback_matches_vm() {
    assert_jit_fallback_matches_vm(USER_ITERATOR_FALLBACK_SOURCE, "user-iterator", "18003000\n");
}

#[test]
fn jit_uint_arithmetic_and_unsigned_comparisons_match_vm() {
    assert_jit_matches_vm(UINT_SOURCE, "uint", "12000\n");
}

#[test]
fn jit_uint_overflow_fault_resumes_in_vm_with_the_same_error() {
    let profile_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!(
            "jit-branch-profile-{}-uint-overflow.json",
            std::process::id()
        ));
    let _ = std::fs::remove_file(&profile_path);

    let vm_result = execute_result(UINT_OVERFLOW_SOURCE, false, None);
    let jit_result = execute_result(UINT_OVERFLOW_SOURCE, true, Some(&profile_path));
    assert_eq!(vm_result, Ok("Error: integer arithmetic overflow\n".into()));
    assert_eq!(jit_result, vm_result);

    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist its successful compilation"),
    )
    .expect("APE profile should be valid JSON");
    assert!(
        profile["functions"]
            .as_object()
            .expect("profile should include function records")
            .values()
            .any(|function| function["compile_ns"].as_u64().unwrap_or(0) > 0)
    );

    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_float_arithmetic_and_nan_comparison_match_vm() {
    assert_jit_matches_vm(FLOAT_SOURCE, "float", "6000\n");
}

#[test]
fn jit_float_division_by_zero_fault_resumes_in_vm() {
    let profile_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!(
            "jit-float-divzero-profile-{}.json",
            std::process::id()
        ));
    let _ = std::fs::remove_file(&profile_path);
    let vm_result = execute_result(FLOAT_DIVISION_BY_ZERO_SOURCE, false, None);
    let jit_result = execute_result(FLOAT_DIVISION_BY_ZERO_SOURCE, true, Some(&profile_path));
    assert_eq!(vm_result, Ok("Error: division by zero\n".into()));
    assert_eq!(jit_result, vm_result);
    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist its successful compilation"),
    )
    .expect("APE profile should be valid JSON");
    assert!(
        profile["functions"]
            .as_object()
            .expect("profile should include function records")
            .values()
            .any(|function| function["compile_ns"].as_u64().unwrap_or(0) > 0)
    );
    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_float_remainder_resumes_in_vm() {
    let profile_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("jit-float-mod-profile-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&profile_path);
    assert_eq!(execute(FLOAT_MOD_SOURCE, false, None), "9000\n");
    assert_eq!(
        execute(FLOAT_MOD_SOURCE, true, Some(&profile_path)),
        "9000\n"
    );
    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("JIT should persist its successful compilation"),
    )
    .expect("APE profile should be valid JSON");
    assert!(
        profile["functions"]
            .as_object()
            .expect("profile should include function records")
            .values()
            .any(|function| function["compile_ns"].as_u64().unwrap_or(0) > 0)
    );
    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_spills_ssa_locals_before_falling_back_to_a_call() {
    assert_jit_fallback_matches_vm(
        JIT_DIRTY_LOCAL_FALLBACK_SOURCE,
        "dirty-locals",
        "18003000\n",
    );
}

#[test]
fn jit_char_ordering_and_bool_equality_match_vm() {
    assert_jit_matches_vm(CHAR_BOOL_COMPARE_SOURCE, "char-bool", "42000\n");
}

#[test]
fn jit_mixed_signed_unsigned_comparisons_match_vm_at_boundaries() {
    assert_jit_matches_vm(
        MIXED_INT_UINT_COMPARE_SOURCE,
        "mixed-int-uint",
        "17106000\n10140000\n",
    );
}

#[test]
fn jit_mixed_integer_float_comparisons_match_vm_at_exact_boundaries() {
    let artifact = project::compile_source(
        "jit-mixed-int-float.cog",
        MIXED_INTEGER_FLOAT_COMPARE_SOURCE,
    )
    .expect("mixed comparison fixture should compile");
    let expected_types = [
        Some(cog::value::RuntimeType::Int),
        Some(cog::value::RuntimeType::Float),
        Some(cog::value::RuntimeType::Int),
    ];
    let signed_score_entry = artifact
        .callables
        .iter()
        .find_map(|(&entry, signature)| {
            signature
                .parameter_types
                .as_ref()
                .is_some_and(|types| types.as_slice() == expected_types.as_slice())
                .then_some(entry)
        })
        .expect("signed_score should have int/float/int parameter types");
    let profile_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("jit-mixed-int-float-{}.json", std::process::id()));
    let _ = std::fs::remove_file(&profile_path);

    let vm_output = execute(MIXED_INTEGER_FLOAT_COMPARE_SOURCE, false, None);
    let jit_output = execute(
        MIXED_INTEGER_FLOAT_COMPARE_SOURCE,
        true,
        Some(&profile_path),
    );
    assert_eq!(jit_output, vm_output);

    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&profile_path).expect("signed_score should persist its JIT profile"),
    )
    .expect("APE profile should be valid JSON");
    assert!(
        profile["functions"][signed_score_entry.to_string()]["jit_ns_per_backedge"]
            .as_u64()
            .unwrap_or(0)
            > 0,
        "signed_score must execute in native code for this regression"
    );
    let _ = std::fs::remove_file(profile_path);
}

#[test]
fn jit_small_bigint_comparisons_match_vm_at_signed_and_float_boundaries() {
    assert_jit_matches_vm_computed(SMALL_BIGINT_COMPARE_SOURCE, "small-bigint");
}

#[test]
fn jit_large_bigint_comparisons_resume_in_the_vm() {
    let expected = execute(LARGE_BIGINT_COMPARE_FALLBACK_SOURCE, false, None);
    assert_jit_fallback_matches_vm(
        LARGE_BIGINT_COMPARE_FALLBACK_SOURCE,
        "large-bigint",
        &expected,
    );
}

#[test]
fn jit_roots_and_restores_string_and_aggregate_locals_across_osr() {
    assert_jit_matches_vm(
        ROOTED_AGGREGATE_LOCALS_SOURCE,
        "rooted-aggregate",
        "18009000\n",
    );
}

#[test]
fn jit_nullable_type_test_and_boolean_ops_match_vm() {
    assert_jit_matches_vm(JIT_TYPE_AND_BOOLEAN_SOURCE, "type-bool", "66000\n36000\n");
}

#[test]
fn jit_nil_equality_preserves_errors_in_both_operands() {
    let template = r#"
        @!include("std")
        use io;
        fn probe(count: int) {
            const failure: int? = "bad" as int;
            var index = 0;
            var swallowed = 0;
            while index < count {
                when COMPARISON is bool { swallowed += 1; }
                index += 1;
            }
            swallowed
        }
        fn main() { io::println(probe(6000)); }
    "#;
    for (suffix, comparison) in [
        ("eq-left", "failure == nil"),
        ("eq-right", "nil == failure"),
        ("ne-left", "failure != nil"),
        ("ne-right", "nil != failure"),
    ] {
        assert_jit_fallback_matches_vm(&template.replace("COMPARISON", comparison), suffix, "0\n");
    }
}

#[test]
fn jit_enum_and_variant_type_tests_match_vm() {
    assert_jit_matches_vm(JIT_ENUM_TYPE_SOURCE, "enum-types", "18000\n30000\n");
}

#[test]
fn jit_error_variant_type_test_falls_back_to_vm_matching() {
    assert_jit_fallback_matches_vm(JIT_ERROR_VARIANT_FALLBACK_SOURCE, "error-variant", "6000\n");
}
