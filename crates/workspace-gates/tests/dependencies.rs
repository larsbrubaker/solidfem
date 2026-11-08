// Dependency gate: every crate solidfem builds on is pure Rust and permissively licensed
// (CLAUDE.md, "Never"). Every package outside the workspace (crates.io and the sibling checkouts
// any sibling checkouts) is checked in one of two ways. Both read the whole
// workspace through cargo, so a crate that moves in, and everything it depends on, is checked
// with nothing named here.
//
// (a) Licenses, over the whole lockfile graph (`cargo metadata --all-features`, every target): a
//     license binds whatever the lockfile carries. Each `license` expression must be satisfied by
//     PERMISSIVE: `OR` (and the old `/` spelling) passes when any branch passes, `AND` needs every
//     branch, and `A WITH exception` is judged by `A`, since an exception only adds permissions. A
//     package with no `license` field, only a license file, is named in LICENSE_FILES with the
//     license that file holds (checked here, and stale once the package leaves the graph or gains
//     a `license` field).
//
// (b) Native code, per supported target (TARGETS): `cargo tree --all-features --target <t>
//     -e normal,build,dev` lists exactly the packages and features that target's builds and
//     tests compile. The lockfile graph is not enough here: it merges every target's packages
//     and features, so it holds packages no build compiles (an optional dependency nobody turns
//     on, a crate only another platform reaches) and features asked for on another platform. A
//     package compiles native code when it has a `links` key or an active build-dependency on
//     `cc`, `cmake`, `bindgen` or `pkg-config` (the native reasons: `links` comes from the
//     metadata, the active build-dependencies from the target's tree). It is allowed only
//     through a named list, each entry saying which native reasons it expects and why they are
//     harmless: NO_NATIVE_CODE (the build script compiles and links nothing; FEATURE_CONDITIONS
//     pins, on every target that builds it, a feature that makes it so) and KNOWN_EXCEPTIONS
//     (breaks the contract; only Lars adds to it). A reason an entry does not expect is refused,
//     and an entry, expected reason or feature condition that no target needs any more fails as
//     stale, so the lists hold only what today's graph needs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

/// SPDX ids solidfem accepts: MIT/BSD/Apache/zlib-compatible permissive licenses and public
/// domain dedications. A permissive license missing here may be added with its SPDX id; a
/// copyleft or source-available one may not.
const PERMISSIVE: &[&str] = &[
    "0BSD",
    "Apache-2.0",
    "BSD-1-Clause",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSL-1.0",
    "CC0-1.0",
    "ISC",
    "MIT",
    "MIT-0",
    "Unicode-3.0",
    "Unicode-DFS-2016",
    "Unlicense",
    "Zlib",
];

/// Packages without a `license` field: (name, the SPDX id of the license file it ships, a phrase
/// that file contains).
const LICENSE_FILES: &[(&str, &str, &str)] = &[];

/// The targets solidfem supports: macOS on both processors, Windows, Linux and the browser.
const TARGETS: [&str; 5] = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
    "wasm32-unknown-unknown",
];

/// Build-dependencies that exist to compile or link native code; with `links`, the native
/// reasons an allowed entry may expect.
const NATIVE_BUILD_TOOLS: [&str; 4] = ["cc", "cmake", "bindgen", "pkg-config"];

/// One allowed package: its name, the native reasons it is expected to have ("links" or a
/// NATIVE_BUILD_TOOLS name) and why they compile no C.
struct Allowed {
    package: &'static str,
    reasons: &'static [&'static str],
    why: &'static str,
}

/// Packages that break the letter of the rule but whose build scripts compile and link nothing.
const NO_NATIVE_CODE: &[Allowed] = &[];

/// Features that keep a NO_NATIVE_CODE entry free of native code on every supported target that
/// builds it: (package, feature, whether it must be on). Only a tool its build script always
/// depends on needs one; an optional tool that a feature turns on shows in the target's tree and
/// is refused there.
const FEATURE_CONDITIONS: &[(&str, &str, bool)] = &[];

/// Packages a supported target compiles that break the pure-Rust contract, kept only until Lars
/// decides; nobody else adds to this list.
const KNOWN_EXCEPTIONS: &[Allowed] = &[];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/workspace-gates sits two levels below the workspace root")
        .to_path_buf()
}

/// Runs cargo at the workspace root and returns its standard output.
fn cargo(args: &[&str]) -> String {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = std::process::Command::new(cargo)
        .current_dir(workspace_root())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("run cargo {args:?}: {e}"));
    assert!(
        output.status.success(),
        "cargo {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo prints UTF-8")
}

/// The lockfile graph: `cargo metadata` with every feature of every member, all targets.
fn metadata() -> Value {
    let json = cargo(&[
        "metadata",
        "--format-version",
        "1",
        "--all-features",
        "--locked",
    ]);
    serde_json::from_str(&json).expect("cargo metadata prints JSON")
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key].as_str().unwrap_or("")
}

/// The packages outside the workspace, sorted by name, then version, so the failure lists read
/// the same on every run.
fn external_packages(meta: &Value) -> Vec<&Value> {
    let members: Vec<&str> = meta["workspace_members"]
        .as_array()
        .expect("workspace_members")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut packages: Vec<&Value> = meta["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .filter(|package| !members.contains(&text(package, "id")))
        .collect();
    packages.sort_by(|a, b| {
        (text(a, "name"), text(a, "version")).cmp(&(text(b, "name"), text(b, "version")))
    });
    packages
}

/// Splits an SPDX expression into identifiers, `(`, `)` and the `/` legacy OR.
fn tokens(expression: &str) -> Vec<String> {
    let spaced = expression
        .replace('(', " ( ")
        .replace(')', " ) ")
        .replace('/', " OR ");
    spaced.split_whitespace().map(str::to_string).collect()
}

/// Whether an SPDX license expression is satisfied by PERMISSIVE; an expression that does not
/// parse is not.
fn is_permissive(expression: &str) -> bool {
    let tokens = tokens(expression);
    let mut at = 0;
    let result = or_expression(&tokens, &mut at);
    result == Some(true) && at == tokens.len()
}

fn or_expression(tokens: &[String], at: &mut usize) -> Option<bool> {
    let mut any = and_expression(tokens, at)?;
    while tokens.get(*at).map(String::as_str) == Some("OR") {
        *at += 1;
        any |= and_expression(tokens, at)?;
    }
    Some(any)
}

fn and_expression(tokens: &[String], at: &mut usize) -> Option<bool> {
    let mut all = license(tokens, at)?;
    while tokens.get(*at).map(String::as_str) == Some("AND") {
        *at += 1;
        all &= license(tokens, at)?;
    }
    Some(all)
}

fn license(tokens: &[String], at: &mut usize) -> Option<bool> {
    let token = tokens.get(*at)?.as_str();
    *at += 1;
    if token == "(" {
        let inner = or_expression(tokens, at)?;
        (tokens.get(*at).map(String::as_str) == Some(")")).then_some(())?;
        *at += 1;
        return Some(inner);
    }
    if matches!(token, ")" | "OR" | "AND" | "WITH") {
        return None;
    }
    if tokens.get(*at).map(String::as_str) == Some("WITH") {
        // The exception only adds permissions; the base license decides.
        tokens.get(*at + 1)?;
        *at += 2;
    }
    Some(PERMISSIVE.contains(&token))
}

/// Why a package's `license` expression fails, or None when it is permissive.
fn license_expression_problem(expression: &str) -> Option<String> {
    (!is_permissive(expression)).then(|| format!("`{expression}`"))
}

/// Why a license-file-only package fails, or None when its file still holds the named license.
fn license_file_problem(package: &Value, spdx: &str, phrase: &str) -> Option<String> {
    let manifest = Path::new(text(package, "manifest_path"));
    let file = manifest
        .parent()
        .expect("a manifest has a folder")
        .join(text(package, "license_file"));
    let body = std::fs::read_to_string(&file).unwrap_or_default();
    (!(body.contains(phrase) && PERMISSIVE.contains(&spdx)))
        .then(|| format!("its license file {} is no longer {spdx}", file.display()))
}

#[test]
fn every_dependency_is_permissively_licensed() {
    let meta = metadata();
    let packages = external_packages(&meta);
    let mut refused = Vec::new();
    for package in &packages {
        let name = text(package, "name");
        let version = text(package, "version");
        let problem = match package["license"].as_str() {
            Some(expression) => license_expression_problem(expression),
            None => match LICENSE_FILES.iter().find(|(n, _, _)| *n == name) {
                Some((_, spdx, phrase)) => license_file_problem(package, spdx, phrase),
                None => Some("no `license` field".to_string()),
            },
        };
        if let Some(problem) = problem {
            refused.push(format!("  {name} {version}: {problem}"));
        }
    }
    assert!(
        refused.is_empty(),
        "{} dependency(ies) without a permissive license:\n{}\nsolidfem is BSD-2 and depends \
         only on MIT/BSD/Apache/zlib-compatible code (CLAUDE.md, \"Never\"). Pick a permissively \
         licensed alternative, or ask Lars. A permissive license this gate does not know yet goes \
         in PERMISSIVE by its SPDX id; a package with only a license file goes in LICENSE_FILES. \
         Third-party code adapted into the repo is recorded in THIRD_PARTY_NOTICES.md.",
        refused.len(),
        refused.join("\n")
    );
    let stale: Vec<&str> = LICENSE_FILES
        .iter()
        .map(|(name, _, _)| *name)
        .filter(|name| {
            !packages
                .iter()
                .any(|p| text(p, "name") == *name && p["license"].is_null())
        })
        .collect();
    assert!(
        stale.is_empty(),
        "LICENSE_FILES names {stale:?}, which left the graph or now has a `license` field that \
         this gate reads instead: delete the entry"
    );
}

/// A package version as cargo prints it: (name, version without the `v`).
type Key = (String, String);

/// What one supported target compiles: each package's active features and the names of its active
/// dependencies (normal, build and, for workspace members, dev).
#[derive(Default)]
struct TargetGraph {
    features: BTreeMap<Key, BTreeSet<String>>,
    children: BTreeMap<Key, BTreeSet<String>>,
}

/// Reads `cargo tree --prefix depth --format '{p}|{f}'`: each line is the depth, the package
/// (`name vVERSION` and maybe a path or `(proc-macro)`), `|`, its comma-separated features, and
/// ` (*)` when its dependencies were already listed above. A line's parent is the closest line
/// above it one level up.
fn parse_tree(tree: &str) -> TargetGraph {
    let mut graph = TargetGraph::default();
    let mut path: Vec<Key> = Vec::new();
    for line in tree.lines() {
        let digits = line.chars().take_while(char::is_ascii_digit).count();
        let Ok(depth) = line[..digits].parse::<usize>() else {
            continue;
        };
        let (package, features) = line[digits..]
            .split_once('|')
            .unwrap_or((&line[digits..], ""));
        let mut words = package.split_whitespace();
        let name = words.next().unwrap_or("").to_string();
        let version = words
            .next()
            .unwrap_or("")
            .trim_start_matches('v')
            .to_string();
        let key = (name.clone(), version);
        let features = features.trim().trim_end_matches("(*)").trim();
        graph.features.entry(key.clone()).or_default().extend(
            features
                .split(',')
                .filter(|f| !f.is_empty())
                .map(str::to_string),
        );
        path.truncate(depth);
        if let Some(parent) = depth.checked_sub(1).and_then(|d| path.get(d)) {
            graph
                .children
                .entry(parent.clone())
                .or_default()
                .insert(name);
        }
        path.push(key);
    }
    graph
}

/// What `target`'s builds and tests compile.
fn target_graph(target: &str) -> TargetGraph {
    parse_tree(&cargo(&[
        "tree",
        "--locked",
        "--workspace",
        "--all-features",
        "--target",
        target,
        "-e",
        "normal,build,dev",
        "--prefix",
        "depth",
        "--format",
        "{p}|{f}",
    ]))
}

/// Every supported target's graph, in TARGETS order.
fn target_graphs() -> Vec<(&'static str, TargetGraph)> {
    TARGETS
        .iter()
        .map(|target| (*target, target_graph(target)))
        .collect()
}

/// From the metadata, per package version outside the workspace: whether it has a `links` key,
/// and the NATIVE_BUILD_TOOLS it declares as build-dependencies.
fn native_declarations(meta: &Value) -> BTreeMap<Key, (bool, BTreeSet<String>)> {
    external_packages(meta)
        .into_iter()
        .map(|package| {
            let key = (
                text(package, "name").to_string(),
                text(package, "version").to_string(),
            );
            let tools = package["dependencies"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|dep| text(dep, "kind") == "build")
                .map(|dep| text(dep, "name"))
                .filter(|name| NATIVE_BUILD_TOOLS.contains(name))
                .map(str::to_string)
                .collect();
            (key, (!package["links"].is_null(), tools))
        })
        .collect()
}

/// The native reasons each compiled package has on one target: "links", or a NATIVE_BUILD_TOOLS
/// name it build-depends on and the target compiles under it.
fn native_reasons(
    graph: &TargetGraph,
    declared: &BTreeMap<Key, (bool, BTreeSet<String>)>,
) -> BTreeMap<Key, BTreeSet<String>> {
    let mut found = BTreeMap::new();
    for key in graph.features.keys() {
        let Some((links, tools)) = declared.get(key) else {
            continue; // a workspace member
        };
        let active = graph.children.get(key);
        let mut reasons: BTreeSet<String> = tools
            .iter()
            .filter(|tool| active.is_some_and(|names| names.contains(*tool)))
            .cloned()
            .collect();
        if *links {
            reasons.insert("links".to_string());
        }
        if !reasons.is_empty() {
            found.insert(key.clone(), reasons);
        }
    }
    found
}

fn allowed_lists() -> [(&'static str, &'static [Allowed]); 2] {
    [
        ("NO_NATIVE_CODE", NO_NATIVE_CODE),
        ("KNOWN_EXCEPTIONS", KNOWN_EXCEPTIONS),
    ]
}

fn allowed_entry(package: &str) -> Option<(&'static str, &'static Allowed)> {
    allowed_lists().into_iter().find_map(|(list, entries)| {
        entries
            .iter()
            .find(|entry| entry.package == package)
            .map(|entry| (list, entry))
    })
}

#[test]
fn no_dependency_compiles_native_code() {
    let meta = metadata();
    let declared = native_declarations(&meta);
    // (package, version, reason) -> the targets that compile it so.
    let mut refused: BTreeMap<(String, String, String), Vec<&str>> = BTreeMap::new();
    // (package, reason) seen on some target, for the stale check.
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
    for (target, graph) in target_graphs() {
        for ((name, version), reasons) in native_reasons(&graph, &declared) {
            for reason in reasons {
                let expected = allowed_entry(&name)
                    .is_some_and(|(_, entry)| entry.reasons.contains(&reason.as_str()));
                seen.insert((name.clone(), reason.clone()));
                if !expected {
                    let key = (name.clone(), version.clone(), reason);
                    refused.entry(key).or_default().push(target);
                }
            }
        }
    }
    let lines: Vec<String> = refused
        .iter()
        .map(|((name, version, reason), targets)| {
            let why = if reason == "links" {
                "has a `links` key".to_string()
            } else {
                format!("build-depends on `{reason}`")
            };
            let note = match allowed_entry(name) {
                Some((list, _)) => format!(", which its {list} entry does not expect"),
                None => String::new(),
            };
            format!(
                "  {name} {version} {why}{note}; compiled for {}\n    see why: cargo tree -i \
                 {name}@{version} --target {} -e normal,build",
                targets.join(", "),
                targets[0]
            )
        })
        .collect();
    assert!(
        lines.is_empty(),
        "{} way(s) a dependency may compile native code:\n{}\nsolidfem is pure Rust: no C/C++, \
         no `-sys` crates, no native build scripts (CLAUDE.md, \"Never\"). First run the `see why` \
         command beside each to learn what pulls it in (`-e features` adds which features turn \
         it on). Then pick a pure-Rust alternative or a feature that \
         compiles nothing, or ask Lars. A package whose build script provably compiles and links \
         nothing goes in NO_NATIVE_CODE with its expected reasons and why (and FEATURE_CONDITIONS \
         when a feature decides it); only Lars adds to KNOWN_EXCEPTIONS.",
        lines.len(),
        lines.join("\n")
    );
    let stale: Vec<String> = allowed_lists()
        .iter()
        .flat_map(|(list, entries)| entries.iter().map(move |entry| (*list, entry)))
        .flat_map(|(list, entry)| entry.reasons.iter().map(move |r| (list, entry, *r)))
        .filter(|(_, entry, reason)| {
            !seen.contains(&(entry.package.to_string(), reason.to_string()))
        })
        .map(|(list, entry, reason)| {
            format!(
                "  {list}: {} ({reason}), allowed as {}",
                entry.package, entry.why
            )
        })
        .collect();
    assert!(
        stale.is_empty(),
        "no supported target compiles these allowed packages with these reasons any more; delete \
         them so the lists hold only what is needed:\n{}",
        stale.join("\n")
    );
}

#[test]
fn allowed_native_packages_keep_the_features_that_make_them_pure() {
    let mut wrong = Vec::new();
    let mut built: BTreeSet<&str> = BTreeSet::new();
    for (target, graph) in target_graphs() {
        for (package, feature, on) in FEATURE_CONDITIONS {
            for ((name, version), features) in &graph.features {
                if name != package {
                    continue;
                }
                built.insert(package);
                if features.contains(*feature) != *on {
                    let state = if *on { "off" } else { "on" };
                    wrong.push(format!(
                        "  {package} {version} for {target}: feature `{feature}` is {state}"
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a feature that kept an allowed package free of native code changed:\n{}\nIt now \
         compiles or links native code, which solidfem's pure-Rust rule refuses (CLAUDE.md, \
         \"Never\"). Find what changed the feature (`cargo tree -e features -i <package> \
         --target <target>`) and restore it, or ask Lars. The package's NO_NATIVE_CODE entry \
         says which feature keeps it pure.",
        wrong.join("\n")
    );
    let stale: Vec<&str> = FEATURE_CONDITIONS
        .iter()
        .map(|(package, _, _)| *package)
        .filter(|package| !built.contains(package))
        .collect();
    assert!(
        stale.is_empty(),
        "FEATURE_CONDITIONS names {stale:?}, which no supported target compiles: delete the entry"
    );
}

#[test]
fn cargo_tree_lines_parse_examples() {
    let graph = parse_tree(
        "0app v0.1.0 (/x/app)|default\n\
         1aegis v0.9.20|default,pure-rust\n\
         2cc v1.2.0|\n\
         1serde v1.0.0|std (*)\n\
         \n\
         0tool v0.1.0 (/x/tool)|\n\
         1aegis v0.9.20|default,pure-rust (*)\n\
         1derive v1.0.0 (proc-macro)|\n",
    );
    let key = |name: &str, version: &str| (name.to_string(), version.to_string());
    let children = |name, version| graph.children.get(&key(name, version)).cloned();
    assert_eq!(
        children("app", "0.1.0"),
        Some(["aegis", "serde"].map(String::from).into())
    );
    assert_eq!(children("aegis", "0.9.20"), Some(["cc".to_string()].into()));
    assert_eq!(
        children("tool", "0.1.0"),
        Some(["aegis", "derive"].map(String::from).into())
    );
    assert_eq!(
        graph.features.get(&key("aegis", "0.9.20")),
        Some(&["default", "pure-rust"].map(String::from).into())
    );
    assert_eq!(
        graph.features.get(&key("serde", "1.0.0")),
        Some(&["std".to_string()].into())
    );
    assert!(graph.features.contains_key(&key("cc", "1.2.0")));
}

#[test]
fn license_expressions_parse_examples() {
    for permissive in [
        "MIT",
        "MIT OR Apache-2.0",
        "MIT/Apache-2.0",
        "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT",
        "(MIT OR Apache-2.0) AND Unicode-3.0",
        "MIT OR Apache-2.0 OR LGPL-2.1-or-later",
        "Apache-2.0 AND ISC",
    ] {
        assert!(is_permissive(permissive), "{permissive} is permissive");
    }
    for refused in [
        "GPL-3.0",
        "MIT AND GPL-2.0",
        "LGPL-2.1-or-later WITH LLVM-exception",
        "(MIT OR Apache-2.0) AND MPL-2.0",
        "MIT OR",
        "(MIT",
        "MIT Apache-2.0",
        "",
    ] {
        assert!(!is_permissive(refused), "{refused} is refused");
    }
}
