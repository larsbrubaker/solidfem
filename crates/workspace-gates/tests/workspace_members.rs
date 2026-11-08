// Workspace membership gate: every crate in the repository is a workspace member, and every member
// takes the workspace's settings, so the other gates, CI and the lints reach a crate the moment
// it moves in.
//
//   (a) every folder `crates/<dir>` holding a `Cargo.toml` is a member of the root workspace
//       (`members` in the root Cargo.toml): a crate left out would build, lint and test only when
//       someone remembers to name it;
//   (b) every member takes the workspace lints (`[lints] workspace = true`), which refuse stubs
//       (`todo!()`, `unimplemented!()`) and undocumented `unsafe`;
//   (c) every member inherits the workspace's version, edition, license, authors and repository
//       (`<key>.workspace = true`), so the crates publish as one release under one license.
//
// Members are read from `cargo metadata --no-deps`, not by parsing the root manifest, so a glob
// in `members` or an `exclude` is read the way Cargo reads it. Each member manifest is read by
// hand (no TOML dependency): comments are stripped and the `[lints]` and `[package]` tables are
// located by their headers.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The `[package]` keys every member inherits from `[workspace.package]`.
const INHERITED_KEYS: [&str; 5] = ["version", "edition", "license", "authors", "repository"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/workspace-gates sits two levels below the workspace root")
        .to_path_buf()
}

/// The manifest paths of the workspace members, as Cargo reads them.
fn member_manifests() -> Vec<PathBuf> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = std::process::Command::new(cargo)
        .current_dir(workspace_root())
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .expect("run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let meta: Value = serde_json::from_slice(&output.stdout).expect("cargo metadata prints JSON");
    let members: Vec<&str> = meta["workspace_members"]
        .as_array()
        .expect("workspace_members")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let mut manifests: Vec<PathBuf> = meta["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .filter(|p| members.contains(&p["id"].as_str().unwrap_or("")))
        .filter_map(|p| p["manifest_path"].as_str().map(PathBuf::from))
        .collect();
    manifests.sort();
    manifests
}

/// Every `crates/<dir>/Cargo.toml` in the repository.
fn crate_manifests() -> Vec<PathBuf> {
    let crates = workspace_root().join("crates");
    let mut manifests: Vec<PathBuf> = fs::read_dir(&crates)
        .unwrap_or_else(|e| panic!("read {}: {e}", crates.display()))
        .map(|e| e.expect("crate entry").path().join("Cargo.toml"))
        .filter(|manifest| manifest.is_file())
        .collect();
    manifests.sort();
    manifests
}

/// `path` as a comparable key: canonical where it exists, so `\\?\` prefixes and `/` vs `\`
/// cannot make one file look like two.
fn key(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Removes a `#` comment from one TOML line, leaving `#` inside double-quoted strings alone.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

/// The comment-free, space-free lines of `table` (e.g. `[lints]`) in a manifest. A line that
/// starts with `[` begins another table.
fn table_lines(manifest: &str, table: &str) -> Vec<String> {
    let mut inside = false;
    let mut lines = Vec::new();
    for raw in manifest.lines() {
        let line = strip_comment(raw).trim().replace([' ', '\t'], "");
        if line.starts_with('[') {
            inside = line == table;
        } else if inside && !line.is_empty() {
            lines.push(line);
        }
    }
    lines
}

/// Whether a member manifest opts into the workspace lints: a `[lints]` table holding
/// `workspace = true`.
fn uses_workspace_lints(manifest: &str) -> bool {
    table_lines(manifest, "[lints]")
        .iter()
        .any(|l| l == "workspace=true")
}

/// The INHERITED_KEYS a member's `[package]` table does not take from the workspace.
fn uninherited_keys(manifest: &str) -> Vec<&'static str> {
    let lines = table_lines(manifest, "[package]");
    INHERITED_KEYS
        .into_iter()
        .filter(|k| !lines.iter().any(|l| *l == format!("{k}.workspace=true")))
        .collect()
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn every_crate_folder_is_a_workspace_member() {
    let members: Vec<PathBuf> = member_manifests().iter().map(|m| key(m)).collect();
    let root = workspace_root();
    let outside: Vec<String> = crate_manifests()
        .iter()
        .filter(|m| !members.contains(&key(m)))
        .map(|m| {
            m.parent()
                .and_then(|dir| dir.strip_prefix(&root).ok())
                .map(|dir| dir.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default()
        })
        .collect();
    assert!(
        outside.is_empty(),
        "these crates are not workspace members, so no build, lint, gate or CI job reaches them: \
         {outside:?}. Add each to `members` in the root Cargo.toml"
    );
    assert!(
        members.contains(&key(&root.join("crates/workspace-gates/Cargo.toml"))),
        "cargo metadata did not list crates/workspace-gates among the members"
    );
}

/// Every member takes the workspace lints, so the stub refusal (`todo!()`, `unimplemented!()`)
/// reaches a new crate too: a member without `[lints] workspace = true` would compile stubs.
#[test]
fn every_member_refuses_stubs_through_the_workspace_lints() {
    let missing: Vec<String> = member_manifests()
        .iter()
        .filter(|m| !uses_workspace_lints(&read(m)))
        .map(|m| m.display().to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "these members don't take the workspace lints: {missing:?}. Add\n\n[lints]\n\
         workspace = true\n\nto each Cargo.toml, so clippy refuses `todo!()` and \
         `unimplemented!()` there too: nothing lands half-built (the root Cargo.toml's \
         [workspace.lints.clippy])"
    );
}

#[test]
fn every_member_inherits_the_workspace_package_keys() {
    let wrong: Vec<String> = member_manifests()
        .iter()
        .filter_map(|m| {
            let keys = uninherited_keys(&read(m));
            (!keys.is_empty()).then(|| format!("  {}: {keys:?}", m.display()))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "these members set their own value for keys the workspace owns:\n{}\nWrite \
         `<key>.workspace = true` in [package] instead, so every crate publishes as one release \
         under one license (the root Cargo.toml's [workspace.package])",
        wrong.join("\n")
    );
}

#[test]
fn manifest_tables_parse_examples() {
    assert!(uses_workspace_lints(
        "[package]\n[lints]\nworkspace = true # all\n"
    ));
    assert!(uses_workspace_lints("[lints]\n workspace=true\n[deps]\n"));
    assert!(!uses_workspace_lints(
        "[package]\nworkspace = true\n[lints]\n"
    ));
    assert!(!uses_workspace_lints("[lints]\n# workspace = true\n"));
    let manifest = "[package]\nname = \"x\"\nversion.workspace = true\nedition.workspace = true\n\
                    license = \"MIT\"\nauthors.workspace = true\n\
                    [lints]\nrepository.workspace = true\n";
    assert_eq!(uninherited_keys(manifest), ["license", "repository"]);
    assert_eq!(strip_comment("a = \"#x\" # y"), "a = \"#x\" ");
}
