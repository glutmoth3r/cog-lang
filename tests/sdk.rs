use cog::sdk::{Engine, RunContext, Signature, Type, Value};

#[test]
fn retained_task_cannot_cancel_a_task_from_a_later_run() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let saved = Rc::new(RefCell::new(None::<Value>));
    let mut engine = Engine::new();
    let retained = saved.clone();
    engine
        .register(
            "host::save",
            Signature::new(vec![Type::Any], Type::Nil),
            move |_, arguments| {
                *retained.borrow_mut() = Some(arguments[0].clone());
                Ok(Value::nil())
            },
        )
        .unwrap();
    let retained = saved.clone();
    engine
        .register(
            "host::load",
            Signature::new(vec![], Type::Any),
            move |_, _| Ok(retained.borrow().as_ref().unwrap().clone()),
        )
        .unwrap();
    let first = engine
        .compile(
            "save.cog",
            r#"
            @!include("std") use host;
            @async fn original() -> int { 11 }
            @async fn main() { host::save(original()); }
        "#,
        )
        .unwrap();
    engine
        .decode(&first.encode().unwrap())
        .unwrap()
        .run(&mut Vec::new())
        .unwrap();
    let second = engine
        .compile(
            "load.cog",
            r#"
            @!include("std") use host; use io; use err::Error;
            @async fn unrelated() -> int { 22 }
            @async fn main() {
                const fresh = unrelated();
                const stale = host::load();
                when cancel(stale) is Error { io::println("foreign"); }
                io::println(await fresh);
            }
        "#,
        )
        .unwrap();
    let mut output = Vec::new();
    engine
        .decode(&second.encode().unwrap())
        .unwrap()
        .run(&mut output)
        .unwrap();
    assert_eq!(output, b"foreign\n22\n");
}

#[test]
fn foreign_task_in_cold_task_does_not_hide_a_live_root_during_collection() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let retained = Rc::new(RefCell::new(None::<Value>));
    let mut engine = Engine::new();
    let saved = retained.clone();
    engine
        .register(
            "host::save",
            Signature::new(vec![Type::Any], Type::Nil),
            move |_, arguments| {
                *saved.borrow_mut() = Some(arguments[0].clone());
                Ok(Value::nil())
            },
        )
        .unwrap();
    let saved = retained.clone();
    engine
        .register(
            "host::load",
            Signature::new(vec![], Type::Any),
            move |_, _| Ok(saved.borrow().as_ref().unwrap().clone()),
        )
        .unwrap();
    let first = engine
        .compile(
            "save.cog",
            r#"
            @!include("std") use host;
            @async fn original() -> int { 11 }
            @async fn main() { host::save(original()); }
        "#,
        )
        .unwrap();
    engine
        .decode(&first.encode().unwrap())
        .unwrap()
        .run(&mut Vec::new())
        .unwrap();
    let second = engine
        .compile(
            "collect.cog",
            r#"
            @!include("std") use host; use io;
            @async fn active(value) -> int { 22 }
            @async fn main() {
                const fresh = active(host::load());
                var index = 0;
                while index < 600 {
                    const item = [index];
                    index = item[0] + 1;
                }
                io::println(await fresh);
            }
        "#,
        )
        .unwrap();
    let mut output = Vec::new();
    engine
        .decode(&second.encode().unwrap())
        .unwrap()
        .run(&mut output)
        .unwrap();
    assert_eq!(output, b"22\n");
}

#[test]
fn native_reference_result_cannot_make_a_const_alias_writable() {
    let mut engine = Engine::new();
    engine
        .register(
            "host::identity",
            Signature::new(vec![Type::Array], Type::Array),
            |_, arguments| Ok(arguments[0].clone()),
        )
        .unwrap();
    let source = r#"
        @!include("std") use host;
        fn main() {
            const protected = [10];
            var alias = host::identity(protected);
            alias << 99;
        }
    "#;
    let error = engine.compile("native-alias.cog", source).err().unwrap();
    assert!(error.message().contains("read-only"), "{error}");
}

#[test]
fn registered_callbacks_round_trip_and_outlive_the_engine() {
    let mut engine = Engine::new();
    engine
        .register(
            "host::add",
            Signature::new(vec![Type::Int, Type::Int], Type::Int),
            |_, arguments| Ok(Value::int(arguments[0].as_int()? + arguments[1].as_int()?)),
        )
        .unwrap();
    let program = engine
        .compile(
            "sdk.cog",
            "@!include(\"std\") use io; use host; fn main() { io::println(host::add(20, 22)); }",
        )
        .unwrap();
    let encoded = program.encode().unwrap();
    let program = engine.decode(&encoded).unwrap();
    drop(engine);
    let mut output = Vec::new();
    program.run(&mut output).unwrap();
    assert_eq!(output, b"42\n");
}

#[test]
fn foreign_callable_cannot_invoke_same_entry_in_another_program() {
    use cog::backend::{CallableSignature, CompiledProgram};
    use cog::instruction::Instruction as I;
    use cog::runtime::NativeImport;
    use cog::template::{ArrayItem, ArrayTemplate};
    use cog::value::Value as VmValue;
    use std::cell::RefCell;
    use std::rc::Rc;

    // Preserve compiler-produced native metadata while constructing the
    // low-level callable operations that Cog source does not yet expose.
    fn import_section(bytes: &[u8]) -> (usize, Vec<String>) {
        fn index(bytes: &[u8], offset: &mut usize) -> usize {
            let value = u32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
            *offset += 4;
            value as usize
        }
        let mut offset = 8;
        let count = index(bytes, &mut offset);
        let mut names = Vec::new();
        for _ in 0..count {
            let length = index(bytes, &mut offset);
            names.push(String::from_utf8(bytes[offset..offset + length].to_vec()).unwrap());
            offset += length + 2;
            let parameters = index(bytes, &mut offset);
            offset += parameters + 2;
        }
        (offset, names)
    }

    for case in 0..3 {
        let retained = Rc::new(RefCell::new(None::<Value>));
        let observed = Rc::new(RefCell::new(Vec::new()));
        let mut engine = Engine::new();
        let saved = retained.clone();
        engine
            .register(
                "host::save",
                Signature::new(vec![Type::Any], Type::Any),
                move |_, arguments| {
                    let mut saved = saved.borrow_mut();
                    if saved.is_none() {
                        let value = if case == 2 {
                            arguments[0].array_get(0)?.array_get(0)?
                        } else {
                            arguments[0].clone()
                        };
                        *saved = Some(value);
                    }
                    Ok(Value::nil())
                },
            )
            .unwrap();
        let saved = retained.clone();
        engine
            .register(
                "host::load",
                Signature::new(vec![Type::Any], Type::Any),
                move |_, _| {
                    let value = saved.borrow().as_ref().unwrap().clone();
                    if case == 2 {
                        Value::array(vec![Value::array(vec![value])?])
                    } else {
                        Ok(value)
                    }
                },
            )
            .unwrap();
        let values = observed.clone();
        engine
            .register(
                "host::observe",
                Signature::new(vec![Type::Any], Type::Any),
                move |_, arguments| {
                    if arguments[0].is_error() {
                        return Ok(arguments[0].clone());
                    }
                    values.borrow_mut().push(arguments[0].as_int()?);
                    Ok(Value::nil())
                },
            )
            .unwrap();
        let metadata = engine
            .compile(
                "native-metadata.cog",
                "@!include(\"std\") use host; fn main() { host::save(0); host::load(nil); host::observe(0); }",
            )
            .unwrap()
            .encode()
            .unwrap();
        let (metadata_end, imports) = import_section(&metadata);
        let import = |name: &str| {
            imports
                .iter()
                .position(|identity| identity.ends_with(name))
                .unwrap()
        };
        let program = |answer| {
            let mut instructions = if case == 1 {
                vec![I::Push(VmValue::Int(answer)), I::MakeClosure(0, 1)]
            } else {
                vec![I::Push(VmValue::Nil)]
            };
            instructions.extend([
                I::CallNative(import("::host::save")),
                I::Pop,
                I::Push(VmValue::Nil),
                I::CallNative(import("::host::load")),
            ]);
            if case == 2 {
                instructions.extend([
                    I::Push(VmValue::Int(0)),
                    I::Index,
                    I::Push(VmValue::Int(0)),
                    I::Index,
                ]);
            }
            instructions.extend([
                I::CallValue(0),
                I::CallNative(import("::host::observe")),
                I::Halt,
            ]);
            let entry = instructions.len();
            let captures = usize::from(case == 1);
            if case == 1 {
                instructions[1] = I::MakeClosure(entry, captures);
                instructions.push(I::LoadCapture(0));
            } else {
                instructions[0] = if case == 2 {
                    I::PushArray(ArrayTemplate::Array(vec![ArrayItem {
                        template: ArrayTemplate::Array(vec![ArrayItem {
                            template: ArrayTemplate::Scalar(VmValue::Function(entry)),
                            repeat: 1,
                        }]),
                        repeat: 1,
                    }]))
                } else {
                    I::Push(VmValue::Function(entry))
                };
                instructions.push(I::Push(VmValue::Int(answer)));
            }
            instructions.push(I::Return);
            let mut artifact = CompiledProgram::new(instructions);
            artifact
                .callables
                .insert(entry, CallableSignature::dynamic(0, captures));
            artifact.imports = ["bytes::render", "hash::of", "str::to_str"]
                .map(|name| NativeImport::named(name).unwrap())
                .to_vec();
            let encoded = cog::bytecode::encode(&artifact).unwrap();
            let (imports_end, _) = import_section(&encoded);
            let mut bytes = metadata[..metadata_end].to_vec();
            bytes.extend_from_slice(&encoded[imports_end..]);
            engine.decode(&bytes).unwrap()
        };
        let owner = program(11);
        let same_bytes_other_program = engine.decode(&owner.encode().unwrap()).unwrap();
        let recipient = program(22);

        owner.run(&mut Vec::new()).unwrap();
        owner.run(&mut Vec::new()).unwrap();
        assert_eq!(*observed.borrow(), [11, 11], "callable case {case}");
        observed.borrow_mut().clear();
        drop(owner);
        for foreign in [same_bytes_other_program, recipient] {
            let error = foreign.run(&mut Vec::new()).unwrap_err();
            assert!(
                error.message().contains("foreign callable"),
                "callable case {case}: {error}"
            );
            assert!(observed.borrow().is_empty(), "callable case {case}");
        }
    }
}

#[test]
fn run_context_supplies_arguments_to_the_typed_main_entry_point() {
    let engine = Engine::new();
    let program = engine
        .compile(
            "sdk-args.cog",
            "@!include(\"std\") use io; fn main(args: str[]) { io::println(args[0]); }",
        )
        .unwrap();
    let mut output = Vec::new();
    let mut context = RunContext::new(&mut output);
    context.set_process_arguments(vec![std::ffi::OsString::from("two words")]);

    program.run_in(&mut context).unwrap();
    drop(context);

    assert_eq!(output, b"two words\n");
}

#[test]
fn registrations_are_engine_scoped_and_check_typed_arguments() {
    let mut engine = Engine::new();
    engine
        .register(
            "host::identity",
            Signature::new(vec![Type::Int], Type::Int),
            |_, arguments| Ok(arguments[0].clone()),
        )
        .unwrap();
    let source = "@!include(\"std\") use host; fn main() { host::identity(1); }";
    assert!(engine.compile("sdk.cog", source).is_ok());
    assert!(Engine::new().compile("sdk.cog", source).is_err());
    assert!(
        engine
            .compile(
                "sdk.cog",
                "@!include(\"std\") use host; fn main() { host::identity(\"bad\"); }"
            )
            .is_err()
    );
}

#[test]
fn callback_panics_become_catchable_native_failures() {
    let mut engine = Engine::new();
    engine
        .register(
            "host::fail",
            Signature::new(vec![], Type::Nil),
            |_, _| -> Result<Value, cog::sdk::Error> {
                panic!("test panic");
            },
        )
        .unwrap();
    let program = engine.compile("panic.cog", "@!include(\"std\") use io; use host; use err; fn main() { try { host::fail(); } catch(e) { io::println(e.error_kind()); } }").unwrap();
    let mut output = Vec::new();
    program.run(&mut output).unwrap();
    assert_eq!(output, b"NativeFailure\n");
}

#[test]
fn metadata_cannot_execute_and_contract_changes_cannot_link() {
    use cog::sdk::Effects;
    let source = "@!include(\"std\") use io; use host; fn main() { io::println(1); host::work(); }";
    let signature = Signature::new(vec![], Type::Nil);
    let mut compiler = Engine::new();
    compiler
        .register_metadata("host::work", signature.clone())
        .unwrap();
    let program = compiler.compile("metadata.cog", source).unwrap();
    let bytes = program.encode().unwrap();
    let mut output = Vec::new();
    assert!(
        program
            .run(&mut output)
            .unwrap_err()
            .message()
            .contains("not linked")
    );
    assert!(
        output.is_empty(),
        "linking must fail before executing any instruction"
    );
    let mut changed = Engine::new();
    changed
        .register(
            "host::work",
            signature.clone().with_effects(Effects::OUTPUT),
            |_, _| Ok(Value::nil()),
        )
        .unwrap();
    assert!(changed.decode(&bytes).is_err());
    assert!(Engine::new().decode(&bytes).is_err());
    let mut linked = Engine::new();
    linked
        .register("host::work", signature, |_, _| Ok(Value::nil()))
        .unwrap();
    linked.decode(&bytes).unwrap().run(&mut output).unwrap();
    assert_eq!(output, b"1\n");
}

#[test]
fn wrong_results_and_callback_errors_follow_the_runtime_error_path() {
    let mut engine = Engine::new();
    engine
        .register("host::wrong", Signature::new(vec![], Type::Int), |_, _| {
            Ok(Value::string("bad"))
        })
        .unwrap();
    engine
        .register("host::fail", Signature::new(vec![], Type::Nil), |_, _| {
            Err(cog::sdk::Error::new("failed"))
        })
        .unwrap();
    let source = "@!include(\"std\") use io; use host; use err; fn main() { try { host::wrong(); } catch(e) { io::println(e.error_kind()); } try { host::fail(); } catch(e) { io::println(e.error_kind()); } }";
    let program = engine
        .decode(
            &engine
                .compile("errors.cog", source)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap();
    let mut output = Vec::new();
    program.run(&mut output).unwrap();
    assert_eq!(output, b"NativeFailure\nNativeFailure\n");
}

#[test]
fn optional_effect_policy_checks_before_invoking_callbacks() {
    use cog::sdk::{Effects, Permissions, RunContext};
    use std::cell::Cell;
    use std::rc::Rc;
    let calls = Rc::new(Cell::new(0));
    let retained = calls.clone();
    let mut engine = Engine::new();
    engine
        .register(
            "host::write",
            Signature::new(vec![], Type::Nil).with_effects(Effects::OUTPUT),
            move |context, _| {
                retained.set(retained.get() + 1);
                context.write_output("called\n")?;
                Ok(Value::nil())
            },
        )
        .unwrap();
    let program = engine
        .compile(
            "effects.cog",
            "@!include(\"std\") use host; fn main() { host::write()!!; }",
        )
        .unwrap();
    let mut output = Vec::new();
    assert!(
        program
            .run_in(&mut RunContext::new(&mut output).with_permissions(Permissions::none()))
            .is_err()
    );
    assert_eq!(calls.get(), 0);
    program.run(&mut output).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(output, b"called\n");
}

#[test]
fn reentry_is_rejected_and_callback_arguments_are_owned_roots() {
    use std::cell::RefCell;
    use std::rc::Rc;
    let nested = Rc::new(Engine::new().compile("nested.cog", "fn main() {}").unwrap());
    let saved = Rc::new(RefCell::new(None));
    let retained = saved.clone();
    let mut engine = Engine::new();
    engine
        .register(
            "host::retain",
            Signature::new(vec![Type::Array], Type::Nil),
            move |_, values| {
                assert_eq!(
                    nested.run(&mut Vec::new()).unwrap_err().kind(),
                    cog::sdk::ErrorKind::Reentrant
                );
                *retained.borrow_mut() = Some(values[0].clone());
                Ok(Value::nil())
            },
        )
        .unwrap();
    engine
        .compile(
            "roots.cog",
            "@!include(\"std\") use host; fn main() { host::retain([42]); }",
        )
        .unwrap()
        .run(&mut Vec::new())
        .unwrap();
    drop(engine);
    assert_eq!(
        saved
            .borrow()
            .as_ref()
            .unwrap()
            .array_get(0)
            .unwrap()
            .as_int()
            .unwrap(),
        42
    );
    Engine::new()
        .compile("after.cog", "fn main() {}")
        .unwrap()
        .run(&mut Vec::new())
        .unwrap();
}

#[test]
fn invalid_registration_is_rejected_without_losing_prior_entries() {
    let mut engine = Engine::new();
    engine
        .register("host::keep", Signature::new(vec![], Type::Nil), |_, _| {
            Ok(Value::nil())
        })
        .unwrap();
    for name in [
        "host::keep",
        "io::print",
        "err::custom",
        "host",
        "host::deep::name",
        "host::bad\n",
    ] {
        assert!(
            engine
                .register_metadata(name, Signature::new(vec![], Type::Nil))
                .is_err(),
            "{name:?}"
        );
    }
    assert!(
        engine
            .register_metadata(
                "host::bad",
                Signature::new(vec![Type::Array], Type::Nil).with_mutating_parameters(vec![1])
            )
            .is_err()
    );
    engine
        .compile(
            "after.cog",
            "@!include(\"std\") use host; fn main() { host::keep(); }",
        )
        .unwrap()
        .run(&mut Vec::new())
        .unwrap();
}

#[test]
fn typed_native_parameters_propagate_errors_and_any_can_handle_them() {
    let mut engine = Engine::new();
    engine
        .register(
            "host::typed",
            Signature::new(vec![Type::Int], Type::Int),
            |_, _| panic!("typed Error must bypass the callback"),
        )
        .unwrap();
    engine
        .register(
            "host::dynamic",
            Signature::new(vec![Type::Any], Type::Bool),
            |_, arguments| Ok(Value::boolean(arguments[0].is_error())),
        )
        .unwrap();
    let source = "@!include(\"std\") use host; use io; use err; fn main() { try { host::typed(1 / 0); } catch(e) { io::println(e.error_kind()); } io::println(host::dynamic(1 / 0)); }";
    let program = engine
        .decode(
            &engine
                .compile("error-arguments.cog", source)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap();
    let mut output = Vec::new();
    program.run(&mut output).unwrap();
    assert_eq!(output, b"DivisionByZero\ntrue\n");
}

#[test]
fn mutation_contracts_check_source_aliases_and_portable_identity() {
    let signature = Signature::new(vec![Type::Array], Type::Nil).with_mutating_parameters(vec![0]);
    let mut engine = Engine::new();
    engine
        .register("host::change", signature, |_, arguments| {
            arguments[0].array_set(0, Value::int(42))?;
            Ok(Value::nil())
        })
        .unwrap();
    assert!(
        engine
            .compile(
                "const.cog",
                "@!include(\"std\") use host; fn main() { const a = [1]; host::change(a); }"
            )
            .is_err()
    );
    let source = "@!include(\"std\") use host; use io; fn main() { var a = [1]; host::change(a); io::println(a); }";
    let encoded = engine
        .compile("mutate.cog", source)
        .unwrap()
        .encode()
        .unwrap();
    let mut output = Vec::new();
    engine.decode(&encoded).unwrap().run(&mut output).unwrap();
    assert_eq!(output, b"[42]\n");
    let mut changed = Engine::new();
    changed
        .register(
            "host::change",
            Signature::new(vec![Type::Array], Type::Nil),
            |_, _| Ok(Value::nil()),
        )
        .unwrap();
    assert!(changed.decode(&encoded).is_err());
}

#[test]
fn run_context_applies_optional_buffer_limits() {
    use cog::sdk::{Limits, RunContext};

    let program = Engine::new()
        .compile(
            "limits.cog",
            "@!include(\"std\") use io; use bytes; fn main() { try { bytes::new(3)!!; io::println(\"ok\"); } catch(error) { io::println(\"R\"); } }",
        )
        .unwrap();

    let mut output = Vec::new();
    program.run(&mut output).unwrap();
    assert_eq!(output, b"ok\n");

    let mut output = Vec::new();
    let limits = Limits {
        max_buffer_bytes: Some(2),
        ..Limits::default()
    };
    let error = program
        .run_in(&mut RunContext::new(&mut output).with_limits(limits))
        .unwrap_err();
    assert_eq!(error.kind(), cog::sdk::ErrorKind::Runtime);
    assert!(error.message().contains("runtime resource limit: bytes"));
    assert!(output.is_empty(), "the failing allocation precedes output");
}
