use cog::{bytecode, interpreter, project};

fn source(body: &str) -> String {
    format!("@!include(\"std\") use io; use err::Error; {body}")
}
fn run(body: &str) -> String {
    let program =
        project::compile_source("mutability.cog", &source(body)).unwrap_or_else(|e| panic!("{e}"));
    let program = bytecode::decode(&bytecode::encode(&program).unwrap()).unwrap();
    let mut out = vec![];
    interpreter::run_with(&program, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}
fn rejects(body: &str, expected: &str) {
    let error = project::compile_source("mutability.cog", &source(body))
        .expect_err("must fail semantic checking")
        .to_string();
    assert!(!error.contains("E0002"), "{error}");
    assert!(error.contains(expected), "expected {expected}: {error}");
}

#[test]
fn shared_aliases_keep_independent_access_paths_without_moves() {
    assert_eq!(
        run("fn main() { var a = [1]; const b = a; a << 2; io::println(a); io::println(b); }"),
        "[1, 2]\n[1, 2]\n"
    );
    rejects(
        "fn main() { var a = [1]; const b = a; a << 2; b << 3; }",
        "cannot mutate",
    );
    rejects(
        "fn main() { const a = [1]; var b = a; b << 2; }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "fn main() { const a = [1]; var b = a; b = [10, 20]; b << 30; io::println(a); io::println(b); }"
        ),
        "[1]\n[10, 20, 30]\n"
    );
    assert_eq!(
        run("fn main() { var a = [1]; var b = a; b << 2; io::println(a); }"),
        "[1, 2]\n"
    );
}

#[test]
fn capabilities_follow_alias_reassignment_and_control_flow_joins() {
    rejects(
        "fn main() { var a = [1]; const b = [2]; a = b; a << 3; }",
        "cannot mutate",
    );
    rejects(
        "fn f(flag: bool) { const a = [1]; var b = [2]; if flag { b = a; } b << 3; } fn main() { f(false); }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "fn f(flag: bool) { const a = [1]; var b = a; if flag { b = [2]; } else { b = [3]; } b << 4; io::println(b); } fn main() { f(true); f(false); }"
        ),
        "[2, 4]\n[3, 4]\n"
    );
    rejects(
        "fn main() { const a = [1]; var b = [2]; if false { b << 3; } }",
        "requires a reachable mutation",
    );
}

#[test]
fn nested_paths_and_heap_stores_do_not_launder_readonly_capabilities() {
    rejects("fn main() { const a = [[1]]; a[0] << 2; }", "cannot mutate");
    rejects(
        "fn main() { const a = [1]; var holder = { item: [2] }; holder.item = a; holder.item << 3; }",
        "cannot mutate",
    );
    rejects(
        "fn main() { const a = [1]; var holder = { item: a }; holder.item << 3; }",
        "cannot mutate",
    );
    rejects(
        "fn main() { var holder = { item: [2] }; var other = holder; const a = [1]; holder.item = a; other.item << 3; }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "fn main() { var holder = { item: [2] }; const other = holder; holder.item << 3; io::println(other.item); }"
        ),
        "[2, 3]\n"
    );
}

#[test]
fn effects_cross_helpers_aliases_returns_and_recursive_cycles() {
    rejects(
        "fn change(a) { a << 2; } fn wrap(a) { var alias = a; change(alias); } fn main() { const a = [1]; wrap(a); }",
        "cannot mutate",
    );
    rejects(
        "fn identity(a) { a } fn main() { const a = [1]; identity(a) << 2; }",
        "cannot mutate",
    );
    rejects(
        "fn boxed(a) { { item: a } } fn main() { const a = [1]; var b = boxed(a); b.item << 2; }",
        "cannot mutate",
    );
    rejects(
        "fn one(a, n: int) { if n > 0 { two(a, n - 1); } } fn two(a, n: int) { a << 2; one(a, n); } fn main() { const a = [1]; one(a, 1); }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "fn change(a) { a << 2; } fn identity(a) { a } fn main() { var a = [1]; change(identity(a)); io::println(a); }"
        ),
        "[1, 2]\n"
    );
    // Rebinding a parameter changes its slot, not the caller's object.
    assert_eq!(
        run(
            "fn replace(a) { a = [2]; a << 3; a } fn main() { const a = [1]; io::println(replace(a)); io::println(a); }"
        ),
        "[2, 3]\n[1]\n"
    );
}

#[test]
fn local_constants_and_default_arguments_are_checked_in_declaration_context() {
    rejects(
        "fn mutate(a) { const b = a; b << 2; } fn main() { var a = [1]; mutate(a); }",
        "cannot mutate",
    );
    rejects(
        "const DEFAULT = [1]; fn mutate(a = DEFAULT) { a << 2; } fn main() { mutate(); }",
        "cannot mutate",
    );
    rejects(
        "fn mutate(a) { a << 2; } fn main() { const a = [1]; mutate(a: a); }",
        "cannot mutate",
    );
    assert_eq!(
        run("fn mutate(a = [1]) { a << 2; a } fn main() { io::println(mutate()); }"),
        "[1, 2]\n"
    );
}

#[test]
fn method_default_arguments_preserve_mutation_and_return_capabilities() {
    rejects(
        "const DEFAULT = [1]; struct Box { value: int } extend Box { fn change(self, values: int[] = DEFAULT) { values << 2; } } fn main() { const box = Box(1); box.change(); }",
        "cannot mutate",
    );
    rejects(
        "const DEFAULT = [1]; struct Box { value: int } extend Box { fn identity(self, values: int[] = DEFAULT) -> int[] { values } } fn main() { const box = Box(1); box.identity() << 2; }",
        "cannot mutate",
    );
    rejects(
        "const DEFAULT = [1]; fn change(values: int[]) -> int[] { values << 2; values } struct Box { value: int } extend Box { fn consume(self, values: int[] = change(DEFAULT)) {} } fn main() { const box = Box(1); box.consume(); }",
        "cannot mutate",
    );
    rejects(
        "const DEFAULT = [1]; struct Box[T: type] { value: T } extend[T: type] Box[T] { fn change(self, values: int[] = DEFAULT) { values << 2; } } fn main() { const box = Box(1); box.change(); }",
        "cannot mutate",
    );
    rejects(
        "const DEFAULT = [1]; struct Box { value: int } extend Box { @async fn change(self, values: int[] = DEFAULT) { values << 2; } } @async fn main() { const box = Box(1); await box.change(); }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "struct Box { value: int } extend Box { fn change(self, values: int[] = [1]) -> int[] { values << 2; values } } fn main() { const box = Box(1); io::println(box.change()); }"
        ),
        "[1, 2]\n"
    );
}

#[test]
fn custom_iterator_items_preserve_capabilities_in_single_and_zip_loops() {
    let declarations = r#"
        use iter;
        const protected = [10];
        struct Source { values: int[] }
        struct Cursor { values: int[], position: int }
        extend Source with iter::IntoIterator[Cursor, int[]] {
            fn into_iter(self) -> Cursor { Cursor(self.values, 0) }
        }
        extend Cursor with iter::Iterator[int[]] {
            fn next(mut self) -> bool {
                if self.position > 0 { return false; }
                self.position += 1;
                true
            }
            fn current(self) -> int[] { self.values }
        }
    "#;
    for loop_body in [
        "for value in Source(protected) { value << 99; }",
        "for index in 0..1, value in Source(protected) { value << index; }",
    ] {
        rejects(
            &format!("{declarations} fn main() {{ {loop_body} }}"),
            "cannot mutate",
        );
    }
    assert_eq!(
        run(&format!(
            "{declarations} fn main() {{ for value in Source([10]) {{ value << 99; io::println(value); }} }}"
        )),
        "[10, 99]\n"
    );
}

#[test]
fn custom_iterator_protocol_effects_are_checked_before_loop_body() {
    let declarations = r#"
        use iter;
        const protected = [10];
        struct Source { values: int[] }
        struct Cursor { values: int[], position: int }
        extend Source with iter::IntoIterator[Cursor, int[]] {
            fn into_iter(self) -> Cursor { Cursor(self.values, 0) }
        }
        extend Cursor with iter::Iterator[int[]] {
            fn next(mut self) -> bool {
                if self.position > 0 { return false; }
                self.position += 1;
                self.values = protected;
                true
            }
            fn current(self) -> int[] { self.values }
        }
    "#;
    rejects(
        &format!("{declarations} fn main() {{ for value in Source([1]) {{ value << 99; }} }}"),
        "cannot mutate",
    );
    rejects(
        &format!(
            "{declarations} fn main() {{ for index in 0..1, value in Source([1]) {{ value << index; }} }}"
        ),
        "cannot mutate",
    );
}

#[test]
fn custom_iterator_current_cannot_launder_a_global_const() {
    rejects(
        r#"
        use iter;
        const protected = [10];
        struct Source { value: int }
        struct Cursor { position: int }
        extend Source with iter::IntoIterator[Cursor, int[]] {
            fn into_iter(self) -> Cursor { Cursor(self.value) }
        }
        extend Cursor with iter::Iterator[int[]] {
            fn next(mut self) -> bool {
                if self.position > 0 { return false; }
                self.position += 1;
                true
            }
            fn current(self) -> int[] { protected }
        }
        fn main() { for value in Source(0) { value << 99; } }
        "#,
        "cannot mutate",
    );
}

#[test]
fn custom_iterator_protocol_mutations_respect_captured_capabilities() {
    for (into_iter, next, current) in [
        ("self.values << 2;", "", ""),
        ("", "self.values << 2;", ""),
        ("", "", "self.values << 2;"),
    ] {
        let program = format!(
            r#"
            use iter;
            const protected = [10];
            struct Source {{ values: int[] }}
            struct Cursor {{ values: int[], position: int }}
            extend Source with iter::IntoIterator[Cursor, int] {{
                fn into_iter(self) -> Cursor {{ {into_iter} Cursor(self.values, 0) }}
            }}
            extend Cursor with iter::Iterator[int] {{
                fn next(mut self) -> bool {{
                    {next}
                    if self.position > 0 {{ return false; }}
                    self.position += 1;
                    true
                }}
                fn current(self) -> int {{ {current} self.position }}
            }}
            fn main() {{ for value in Source(protected) {{ value; }} }}
            "#
        );
        rejects(&program, "cannot mutate");
    }
}

#[test]
fn generic_iterator_specialization_retains_item_capabilities() {
    let declarations = r#"
        use iter;
        const protected = [10];
        struct Source[T: type] { value: T }
        struct Cursor[T: type] { value: T, position: int }
        extend[T: type] Source[T] with iter::IntoIterator[Cursor[T], T] {
            fn into_iter(self) -> Cursor[T] { Cursor(self.value, 0) }
        }
        extend[T: type] Cursor[T] with iter::Iterator[T] {
            fn next(mut self) -> bool {
                if self.position > 0 { return false; }
                self.position += 1;
                true
            }
            fn current(self) -> T { self.value }
        }
    "#;
    rejects(
        &format!(
            "{declarations} fn main() {{ for value in Source(protected) {{ value << 99; }} }}"
        ),
        "cannot mutate",
    );
    assert_eq!(
        run(&format!(
            "{declarations} fn main() {{ for value in Source([10]) {{ value << 99; io::println(value); }} }}"
        )),
        "[10, 99]\n"
    );
}

#[test]
fn iterator_effects_remain_visible_after_break() {
    rejects(
        r#"
        use iter;
        var shared = [1];
        const protected = [10];
        struct Source { value: int }
        struct Cursor { position: int }
        extend Source with iter::IntoIterator[Cursor, int] {
            fn into_iter(self) -> Cursor { Cursor(self.value) }
        }
        extend Cursor with iter::Iterator[int] {
            fn next(mut self) -> bool { self.position += 1; true }
            fn current(self) -> int { shared = protected; self.position }
        }
        fn main() { for value in Source(0) { break; } shared << 99; }
        "#,
        "cannot mutate",
    );
}

#[test]
fn zip_skips_iterator_steps_after_a_known_empty_range() {
    let declarations = r#"
        use iter;
        const protected = [10];
        struct Source { values: int[] }
        struct Cursor { values: int[] }
        extend Source with iter::IntoIterator[Cursor, int] {
            fn into_iter(self) -> Cursor { Cursor(self.values) }
        }
        extend Cursor with iter::Iterator[int] {
            fn next(mut self) -> bool { self.values << 99; self.values = [1]; false }
            fn current(self) -> int { 0 }
        }
    "#;
    run(&format!(
        "{declarations} fn main() {{ for index in 0..0, value in Source(protected) {{ index; value; }} }}"
    ));
    rejects(
        &format!(
            "{declarations} fn main() {{ for value in Source(protected), index in 0..0 {{ value; index; }} }}"
        ),
        "cannot mutate",
    );
}

#[test]
fn iteration_rechecks_elements_after_the_collection_changes() {
    for loop_body in [
        "for value in arrays { value << 99; arrays[1] = protected; }",
        "for index in 0..2, value in arrays { value << index; arrays[1] = protected; }",
    ] {
        rejects(
            &format!(
                "fn main() {{ const protected = [10]; var arrays = [[1], [2]]; {loop_body} }}"
            ),
            "cannot mutate",
        );
    }
    assert_eq!(
        run(
            "fn main() { var arrays = [[1], [2]]; for value in arrays { value << 3; } io::println(arrays); }"
        ),
        "[[1, 3], [2, 3]]\n"
    );
}

#[test]
fn native_heap_mutations_are_distinct_from_resource_operations() {
    rejects(
        "use bytes; fn main() { const b = bytes::new(1); bytes::set(b, 0, 65); }",
        "cannot mutate",
    );
    rejects(
        "use bytes; fn main() { const b = bytes::new(1); io::read_into(io::stdin(), b, 0, 1); }",
        "cannot mutate",
    );
    rejects(
        "use str; use fmt; fn main() { const b = str::builder(); fmt::write(b, \"%\", 1); }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "use bytes; fn main() { var b = bytes::new(1); bytes::set(b, 0, 65); const out = io::stdout(); io::write_all(out, b, 0, 1); io::flush(out); }"
        ),
        "A"
    );
}

#[test]
fn reachable_mutations_are_counted_with_shadowing_lazy_paths_and_terminal_calls() {
    for body in [
        "fn main() { var a = [1]; while false { a << 2; } }",
        "fn main() { var a = [1]; for i in 0..0 { a << i; } }",
        "fn main() { var a = [1]; crash(\"stop\"); a << 2; }",
        "fn main() { var a = 1; if true { var a = 2; a = 3; } io::println(a); }",
        "fn main() { var a = [1]; false && (if true { a << 2; true } else { false }); }",
        "fn main() { var a = [1]; 1 ?? (if true { a << 2; 2 } else { 2 }); }",
        "fn main() { var a = [1]; const x: int = 2; x ?? (if true { a << 2; 2 } else { 2 }); }",
    ] {
        rejects(body, "requires a reachable mutation");
    }
    assert_eq!(
        run("fn main() { var a = [1]; defer { a << 2; io::println(a); } }"),
        "[1, 2]\n"
    );
}

#[test]
fn defer_uses_capability_at_exit_and_retains_lexical_capture() {
    rejects(
        "fn main() { var a = [1]; const b = [2]; defer { a << 3; } a = b; }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "fn main() { var a = [1]; defer { a << 3; io::println(a); } if true { const a = [2]; io::println(a); } }"
        ),
        "[2]\n[1, 3]\n"
    );
}

#[test]
fn named_sequence_methods_evaluate_once_and_return_the_mutated_array() {
    assert_eq!(
        run(
            "var calls = 0; var a = [2]; fn get() { calls += 1; a } fn main() { io::println(get().append(3).prepend(1)); io::println(calls); io::println(a); }"
        ),
        "[1, 2, 3]\n1\n[1, 2, 3]\n"
    );
    rejects(
        "fn main() { const a = [1]; a.prepend(2); }",
        "cannot mutate",
    );
    rejects("fn main() { 1.append(2); }", "array receiver");
}

#[test]
fn capabilities_follow_global_reassignment_across_calls() {
    rejects(
        "var global = [1]; fn set(a) { global = a; } fn change() { global << 2; } fn main() { const a = [3]; set(a); change(); }",
        "cannot mutate",
    );
    rejects(
        "const a = [1]; var global = a; fn main() { global << 2; }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "var global = [1]; fn change() { global << 2; } fn main() { const view = global; change(); io::println(view); }"
        ),
        "[1, 2]\n"
    );
}

#[test]
fn closure_heap_and_global_stores_preserve_readonly_aliases() {
    for body in [
        r#"
        fn main() {
            const protected = [10];
            var holder = {item: [1]};
            const install = fn[var holder, protected]() -> nil { holder.item = protected; };
            (install)();
            holder.item << 99;
        }
        "#,
        r#"
        fn make(holder, value) -> fn() -> nil {
            var current = holder;
            const install = fn[var current, value]() -> nil { current.item = value; };
            install
        }
        fn main() {
            const protected = [10];
            var holder = {item: [1]};
            const install = make(holder, protected);
            (install)();
            holder.item << 99;
        }
        "#,
        r#"
        var shared = [1];
        fn main() {
            const protected = [10];
            const install = fn[protected]() -> nil { shared = protected; };
            (install)();
            shared << 99;
        }
        "#,
        r#"
        fn main() {
            const protected = [10];
            var holder = {item: [1]};
            const install = fn[var holder, protected](flag: bool) -> nil {
                if flag { holder.item = protected; return; }
            };
            (install)(true);
            holder.item << 99;
        }
        "#,
    ] {
        rejects(body, "cannot mutate");
    }
    assert_eq!(
        run(r#"
        fn main() {
            var holder = {item: [1]};
            var value = [10];
            const install = fn[var holder, var value]() -> nil { holder.item = value; };
            (install)();
            holder.item << 99;
            io::println(value);
        }
        "#),
        "[10, 99]\n"
    );
}

#[test]
fn terminal_helpers_and_infinite_loops_do_not_justify_unreachable_writes() {
    rejects(
        "fn stop() { crash(\"stop\"); } fn main() { var a = [1]; stop(); a << 2; }",
        "requires a reachable mutation",
    );
    rejects(
        "fn main() { var a = [1]; while true {} a << 2; }",
        "requires a reachable mutation",
    );
    rejects(
        "fn main() { var a = [1]; defer { a << 2; } while true {} }",
        "requires a reachable mutation",
    );
    assert_eq!(
        run("fn main() { var a = [1]; while true { a << 2; break; } io::println(a); }"),
        "[1, 2]\n"
    );
}

#[test]
fn aliases_stored_by_helpers_keep_capabilities_and_parameter_aliasing_is_checked() {
    rejects(
        "fn save(holder, value) { holder.item = value; } fn main() { var holder = { item: [1] }; const value = [2]; save(holder, value); holder.item << 3; }",
        "cannot mutate",
    );
    rejects(
        "fn save_and_change(p, q, value) { p.item = value; q.item << 3; } fn main() { var holder = { item: [1] }; const value = [2]; save_and_change(holder, holder, value); }",
        "cannot mutate",
    );
    rejects(
        "fn boxed(a, flag: bool) { if flag { return { item: a }; } { item: [1] } } fn main() { const a = [2]; var b = boxed(a, true); b.item << 3; }",
        "cannot mutate",
    );
}

#[test]
fn catch_entry_keeps_capabilities_from_interrupted_prefixes() {
    rejects(
        "fn fail() { 1 / 0 } fn main() { var b = [1]; const a = [2]; try { b = a; fail(); b = [3]; } catch(e) { b << 4; } }",
        "cannot mutate",
    );
}

#[test]
fn scalar_copies_do_not_inherit_readonly_reference_capability() {
    assert_eq!(
        run(
            "fn increase(x: int) -> int { x += 1; x } fn main() { const a = 1; var b = a; b = increase(b); io::println(a); io::println(b); }"
        ),
        "1\n2\n"
    );
}

#[test]
fn recursive_projection_widening_keeps_nested_readonly_access() {
    rejects(
        "fn walk(node, depth: int) { if depth > 0 { walk(node.next, depth - 1); } else { node.item << 2; } } fn main() { const a = [1]; var node = { next: { next: { next: { next: { next: { item: a } } } } } }; walk(node, 5); }",
        "cannot mutate",
    );
}

#[test]
fn sequence_compound_assignments_are_removed_and_array_addition_retains_child_capabilities() {
    for op in ["<<=", ">>="] {
        let body = format!("fn main() {{ var values = [1]; values {op} 2; }}");
        let error = project::compile_source("mutability.cog", &source(&body))
            .expect_err("removed compound syntax")
            .to_string();
        assert!(error.contains("E0002"), "{error}");
    }
    rejects(
        "fn main() { const a = [[1]]; var b = a; b += []; b[0] << 2; }",
        "cannot mutate",
    );
    assert_eq!(
        run(
            "fn main() { const a = [1]; var b = a; b += [2]; b << 3; io::println(a); io::println(b); }"
        ),
        "[1]\n[1, 2, 3]\n"
    );
}

#[test]
fn invalid_initializers_do_not_cascade_into_required_mutation_errors() {
    let error = project::compile_source(
        "mutability.cog",
        &source("fn main() { var bad: int = \"text\"; const a = [1]; a << 2; missing(); }"),
    )
    .expect_err("independent semantic errors")
    .to_string();
    for message in ["type mismatch", "cannot mutate", "unresolved name"] {
        assert!(error.contains(message), "{error}");
    }
    assert!(
        !error.contains("E1006"),
        "invalid binding must not cascade: {error}"
    );
    rejects(
        "fn main() { const a = [1]; if false { var b = [2]; b = a; b << 3; } }",
        "cannot mutate",
    );
}

#[test]
fn returns_in_loop_conditions_and_known_nonempty_ranges_stop_successor_mutations() {
    assert_eq!(
        run(
            "fn main() { var a = [1]; for i in 0..1 { const a = [2]; io::println(a); break; } a << 3; io::println(a); }"
        ),
        "[2]\n[1, 3]\n"
    );
    rejects(
        "fn main() { var a = [1]; while (if true { return; } else { true }) {} a << 2; }",
        "requires a reachable mutation",
    );
    rejects(
        "fn main() { var a = [1]; for i in 0..1 { return; } a << 2; }",
        "requires a reachable mutation",
    );
}

#[test]
fn repeated_allocations_do_not_clear_capabilities_of_escaped_older_objects() {
    rejects(
        "fn main() { var saved = { item: [0] }; const a = [1]; for i in 0..2 { var fresh = { item: [2] }; if i == 0 { fresh.item = a; saved = fresh; } else { saved.item << 3; } } }",
        "cannot mutate",
    );
    rejects(
        "fn make() { { item: [2] } } fn main() { var saved = { item: [0] }; const a = [1]; for i in 0..2 { var fresh = make(); if i == 0 { fresh.item = a; saved = fresh; } else { saved.item << 3; } } }",
        "cannot mutate",
    );
}
