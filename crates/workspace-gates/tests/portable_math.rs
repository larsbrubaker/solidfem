// Portable math gate (CLAUDE.md, "Exactness"): the crates listed in `PORTABLE_CRATES` give the
// same bits on every system, so their `src/` never calls the platform's transcendental functions.
//
// `f64::acos`, `sin`, `exp` and the rest come from each system's own math library, and those
// libraries differ in the last bit of some results. In the mesher one such bit moves a layer point
// or flips an improvement decision, and the mesh comes out different on macOS than on Windows or
// Linux from the same input. The `libm` crate's functions are the same pure-Rust code on every
// target, wasm32 included, so they are what these crates call. `sqrt` and `+ - * /` are exactly
// rounded by IEEE 754 everywhere and stay allowed. `powi` is refused because Rust leaves its
// precision to the platform; `mul_add` is refused too, as `clippy.toml` refuses it
// workspace-wide, because whether it fuses depends on the target.
//
// A use is `.name(` (a method call), `f64::name` or `f32::name` (a path, called or passed as a
// function, as in `.map(f64::acos)`), or `>::name` (a qualified path, `<f64>::acos`), with
// `name` a whole identifier, so `libm::acos(` and a name that merely starts the same
// (`.cosine(`) pass. Comments, string and byte-string literals (raw ones too) and character
// literals are blanked before matching, keeping line numbers, so prose about `acos` is never a
// finding. Files are those `support/repo_walk.rs` walks under each crate's `src/`.

#[path = "support/repo_walk.rs"]
mod repo_walk;

use std::path::Path;

use repo_walk::{repo_files, workspace_root, TempTree};

/// The crates whose output must be bit-identical on every system.
const PORTABLE_CRATES: [&str; 1] = ["solidfem-tetra"];

/// The platform functions whose last bit can differ between systems.
const PLATFORM_FUNCTIONS: [&str; 27] = [
    "acos", "asin", "atan", "atan2", "sin", "cos", "tan", "sin_cos", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "exp2", "exp_m1", "ln", "ln_1p", "log", "log2", "log10",
    "powf", "powi", "cbrt", "hypot", "mul_add",
];

const RULE: &str =
    "These crates must give the same mesh, bit for bit, on every system (CLAUDE.md, \
                    \"Exactness\"). The platform's transcendental functions come from each \
                    system's own math library, which differ in the last bit, so call the `libm` \
                    crate's function instead (`libm::acos(x)` for `x.acos()`): it is the same \
                    pure-Rust code on every target. Write `a * b + c` for `mul_add`, and a \
                    power by repeated multiplication where that is exact.";

/// `text` with every comment, string literal and character literal replaced by spaces, newlines
/// kept, so a match's line number is its line in the file. Nested block comments nest, as Rust's
/// do; a raw string's `#` count is honoured, and a byte string (`b"`, `br"`) is a string.
fn code_only(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            let mut depth = 0usize;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(chars[i]));
                    i += 1;
                }
            }
        } else if c == 'r' && matches!(next, Some('"') | Some('#')) && is_raw_string(&chars, i) {
            let hashes = chars[i + 1..].iter().take_while(|&&h| h == '#').count();
            let body = i + 2 + hashes;
            for _ in i..body {
                out.push(' ');
            }
            i = body;
            while i < chars.len() {
                let closes = chars[i] == '"'
                    && chars[i + 1..]
                        .iter()
                        .take(hashes)
                        .filter(|&&h| h == '#')
                        .count()
                        == hashes;
                if closes {
                    for _ in 0..=hashes {
                        out.push(' ');
                    }
                    i += 1 + hashes;
                    break;
                }
                out.push(blank(chars[i]));
                i += 1;
            }
        } else if c == '"' {
            out.push(' ');
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                let step = if chars[i] == '\\' { 2 } else { 1 };
                for &escaped in chars.iter().skip(i).take(step) {
                    out.push(blank(escaped));
                }
                i += step;
            }
            if i < chars.len() {
                out.push(' ');
                i += 1;
            }
        } else if c == '\'' {
            // A character literal is `'x'` or `'\…'`; anything else is a lifetime or label.
            let length = match (next, chars.get(i + 2)) {
                // The escaped character itself may be a quote (`'\''`), so the closing quote is
                // searched for after it.
                (Some('\\'), _) => chars
                    .get(i + 3..)
                    .and_then(|rest| rest.iter().position(|&q| q == '\''))
                    .map(|p| p + 4),
                (Some(_), Some('\'')) => Some(3),
                _ => None,
            };
            match length {
                Some(length) => {
                    for _ in 0..length.min(chars.len() - i) {
                        out.push(' ');
                    }
                    i += length;
                }
                None => {
                    out.push(c);
                    i += 1;
                }
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// True when the `r` at `i` opens a raw string (`r"`, `r#"`, or after a `b`, `br"`) rather than
/// ending an identifier.
fn is_raw_string(chars: &[char], i: usize) -> bool {
    let token_start = |k: usize| k == 0 || !is_identifier(chars[k - 1]);
    let starts_token = token_start(i) || (i > 0 && chars[i - 1] == 'b' && token_start(i - 1));
    let hashes = chars[i + 1..].iter().take_while(|&&h| h == '#').count();
    starts_token && chars.get(i + 1 + hashes) == Some(&'"')
}

fn is_identifier(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Every use of a platform function in one file's code, as `(line, use)` with the use written
/// `.name(`, `f64::name`, `f32::name` or `>::name`, line counted from 1.
fn platform_calls(text: &str) -> Vec<(usize, String)> {
    let code = code_only(text);
    let mut found = Vec::new();
    for (index, line) in code.lines().enumerate() {
        for name in PLATFORM_FUNCTIONS {
            let mut from = 0;
            while let Some(at) = line[from..].find(name) {
                let start = from + at;
                from = start + name.len();
                let (before, after) = (&line[..start], &line[from..]);
                let whole = !before.chars().next_back().is_some_and(is_identifier)
                    && !after.chars().next().is_some_and(is_identifier);
                if !whole {
                    continue;
                }
                let form = if before.ends_with('.') {
                    after.starts_with('(').then(|| format!(".{name}("))
                } else if let Some(path) = before.strip_suffix("::") {
                    // The path segment just before `::`, as a whole identifier.
                    let segment_start = path
                        .char_indices()
                        .rev()
                        .take_while(|&(_, c)| is_identifier(c))
                        .last()
                        .map_or(path.len(), |(k, _)| k);
                    let segment = &path[segment_start..];
                    if path.ends_with('>') {
                        Some(format!(">::{name}"))
                    } else if segment == "f64" || segment == "f32" {
                        Some(format!("{segment}::{name}"))
                    } else {
                        None
                    }
                } else {
                    None
                };
                found.extend(form.map(|form| (index + 1, form)));
            }
        }
    }
    found.sort();
    found
}

/// Every finding under `crates/<krate>/src/` of the repository at `root`, one line each.
fn findings(root: &Path, krate: &str) -> Vec<String> {
    let prefix = format!("crates/{krate}/src/");
    let files: Vec<String> = repo_files(root)
        .into_iter()
        .filter(|path| path.starts_with(&prefix) && path.ends_with(".rs"))
        .collect();
    assert!(!files.is_empty(), "no Rust sources found under {prefix}");
    let mut out = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(root.join(&path))
            .unwrap_or_else(|e| panic!("read {path}: {e}"));
        for (line, call) in platform_calls(&text) {
            out.push(format!("  {path}:{line}: {call}"));
        }
    }
    out
}

#[test]
fn portable_crates_call_no_platform_math() {
    let root = workspace_root();
    let offenders: Vec<String> = PORTABLE_CRATES
        .iter()
        .flat_map(|krate| findings(&root, krate))
        .collect();
    assert!(
        offenders.is_empty(),
        "{} call(s) to the platform's math library:\n{}\n{RULE}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn platform_calls_are_found_in_code_only() {
    assert_eq!(
        platform_calls("let a = x.acos();"),
        [(1, ".acos(".to_string())]
    );
    assert_eq!(
        platform_calls("fn f() {}\nlet a = f64::atan2(y, x);"),
        [(2, "f64::atan2".to_string())]
    );
    assert_eq!(
        platform_calls("let s = t.sin() + t.cos();"),
        [(1, ".cos(".to_string()), (1, ".sin(".to_string())]
    );
    assert_eq!(platform_calls("let m = a.mul_add(b, c);").len(), 1);
    assert_eq!(platform_calls("let h = 2.0_f64.powi(-k);").len(), 1);
    // A function passed by path, and a qualified path, are uses too.
    assert_eq!(
        platform_calls("let a: Vec<f64> = c.iter().copied().map(f64::acos).collect();"),
        [(1, "f64::acos".to_string())]
    );
    assert_eq!(
        platform_calls("let a = <f64>::acos(x); let s = core::f32::sin;"),
        [(1, ">::acos".to_string()), (1, "f32::sin".to_string())]
    );
    // Portable or merely similar names.
    assert!(platform_calls("let a = libm::acos(x);").is_empty());
    assert!(platform_calls("let c = v.cosine(); let s = x.sqrt();").is_empty());
    assert!(platform_calls("let a = x.asin_like();").is_empty());
    assert!(platform_calls("let d = x.to_degrees() * y.powi2(2);").is_empty());
    assert!(platform_calls("let e = f64::EPSILON; let p = my_f64::acos(x);").is_empty());
    assert!(platform_calls("let l = self.log; let c = x.cos;").is_empty());
    // Prose and text.
    assert!(platform_calls("// the angle is x.acos()\nlet y = 1;").is_empty());
    assert!(platform_calls("/* x.sin() /* nested */ y.cos() */ let z = 1;").is_empty());
    assert!(platform_calls("let s = \"x.exp() \\\" y.ln()\";").is_empty());
    assert!(platform_calls("let s = r#\"x.tan() \"y\" \"#;").is_empty());
    assert!(platform_calls("let s = br#\"a\"b.tan()\"#; let t = b\"y.cos()\";").is_empty());
    // A lifetime is not a character literal, and a quote character does not open a string.
    assert_eq!(
        platform_calls("fn f<'a>(x: &'a f64) -> f64 { x.exp() }").len(),
        1
    );
    assert_eq!(platform_calls("let q = '\"'; let e = x.exp();").len(), 1);
    // An escaped quote character neither ends the literal early nor swallows the code after it.
    assert_eq!(
        platform_calls("let q = ['\\'','\"']; let s = \"x.sin()\"; let e = x.exp();"),
        [(1, ".exp(".to_string())]
    );
    assert_eq!(platform_calls("let b = b'\\''; let e = x.exp();").len(), 1);
    // Line numbers survive a block comment that spans lines.
    assert_eq!(
        platform_calls("/* one\ntwo */\nlet a = x.ln();"),
        [(3, ".ln(".to_string())]
    );
}

#[test]
fn findings_name_the_file_and_line() {
    let tree = TempTree::new("portable-math");
    tree.write(
        "crates/demo/src/lib.rs",
        "// header: x.acos() in prose\nfn f(x: f64) -> f64 {\n    x.acos()\n}\n",
    );
    tree.write(
        "crates/demo/tests/t.rs",
        "// a test may\nfn g() { 1.0f64.sin(); }\n",
    );
    assert_eq!(
        findings(&tree.0, "demo"),
        ["  crates/demo/src/lib.rs:3: .acos("]
    );
}
