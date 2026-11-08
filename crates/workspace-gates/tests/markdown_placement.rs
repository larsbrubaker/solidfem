// Markdown placement gate (CLAUDE.md, "Code and docs"). Documentation lives in the code, in the
// header comment of the file, module or crate it describes, where whoever works on that code reads
// it, and in the rustdoc a library user reads. A Markdown file is allowed only where it is not
// documentation of code:
//
//   - `README.md`, `CLAUDE.md`, `THIRD_PARTY_NOTICES.md` and `CHANGELOG.md` at the repository root;
//   - a crate's own `README.md`, `CHANGELOG.md` and `THIRD_PARTY_NOTICES.md`
//     (`crates/<crate>/README.md`), which crates.io shows, a release updates and a published
//     crate's licenses require (`notices.rs` keeps each crate's notices the root's section);
//   - license files (`LICENSE*`, `COPYING*`, `NOTICE*`, any case) at the root or a crate's root;
//   - anything under `.claude/` (agent instructions and skills);
//   - `docs/later/<idea>.md`, an idea for later; nothing deeper under `docs/later/`, and nothing
//     anywhere else under `docs/`;
//   - anything under a `fixtures` folder inside a crate's `tests/` (`crates/<crate>/tests/**/
//     fixtures/**`): test input, not documentation.
//
// Every other `.md` or `.markdown` file (any case) fails the gate, named with what to do about
// it. The files are those `support/repo_walk.rs` walks, the same as `file_compliance.rs`'s, and
// untracked files count, so a new doc fails before it is committed. Crates are matched by
// position (`crates/<any>/`), so a crate that moves in needs nothing added here.

#[path = "support/repo_walk.rs"]
mod repo_walk;

use repo_walk::{repo_files, workspace_root, TempTree};

/// What the gate makes of one repository path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// Not a Markdown file: the gate has nothing to say about it.
    NotMarkdown,
    /// A Markdown file in one of the allowed places.
    Allowed,
    /// A Markdown file anywhere else.
    Refused,
}

/// True for a Markdown file by its extension, whatever its case.
fn is_markdown(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => {
            ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("markdown")
        }
        _ => false,
    }
}

/// True for a license file's name: `LICENSE`, `LICENSE-MIT.md`, `COPYING.md`, `NOTICE.md`...
fn is_license_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    ["LICENSE", "LICENCE", "COPYING", "NOTICE"]
        .iter()
        .any(|prefix| upper.starts_with(prefix))
}

/// True for a file a crate's own folder may hold at its root: its README, its CHANGELOG, its
/// third-party notices (the `notices` gate keeps them the root's section for the crate) or a
/// license file. `rest` is the path below `crates/<crate>/`.
fn is_crate_root_doc(rest: &str) -> bool {
    !rest.contains('/')
        && (matches!(
            rest,
            "README.md" | "CHANGELOG.md" | "THIRD_PARTY_NOTICES.md"
        ) || is_license_name(rest))
}

/// True for a path inside a `fixtures` folder of a crate's tests: `rest` is the path below
/// `crates/<crate>/`.
fn is_test_fixture(rest: &str) -> bool {
    let mut parts = rest.split('/');
    parts.next() == Some("tests") && {
        let folders: Vec<&str> = parts.collect();
        folders.len() > 1 && folders[..folders.len() - 1].contains(&"fixtures")
    }
}

/// Where `path` (relative to the root, `/`-separated) stands.
fn placement(path: &str) -> Placement {
    if !is_markdown(path) {
        return Placement::NotMarkdown;
    }
    let at_root = !path.contains('/');
    let in_crate = path
        .strip_prefix("crates/")
        .and_then(|rest| rest.split_once('/'))
        .filter(|(krate, _)| !krate.is_empty())
        .map(|(_, rest)| rest);
    let allowed = matches!(
        path,
        "README.md" | "CLAUDE.md" | "THIRD_PARTY_NOTICES.md" | "CHANGELOG.md"
    ) || (at_root && is_license_name(path))
        || in_crate.is_some_and(|rest| is_crate_root_doc(rest) || is_test_fixture(rest))
        || path
            .strip_prefix(".claude/")
            .is_some_and(|rest| !rest.is_empty())
        || path
            .strip_prefix("docs/later/")
            .is_some_and(|idea| !idea.contains('/'));
    if allowed {
        Placement::Allowed
    } else {
        Placement::Refused
    }
}

/// The gate's failure message for the refused paths, empty when there are none.
fn failure_message(refused: &[&str]) -> String {
    if refused.is_empty() {
        return String::new();
    }
    let mut message = format!(
        "{} Markdown file(s) outside the places Markdown may live:\n",
        refused.len()
    );
    for path in refused {
        message.push_str(&format!("  {path}\n"));
    }
    message.push_str(
        "Documentation lives in the code. Move what it explains into the header comment or \
         rustdoc of the code it is about (CLAUDE.md, \"Code and docs\"), then delete it. An idea \
         for later goes in docs/later/<idea>.md.\n",
    );
    message
}

/// The refused paths among the files the walk keeps under `root`.
fn misplaced(root: &std::path::Path) -> Vec<String> {
    repo_files(root)
        .into_iter()
        .filter(|path| placement(path) == Placement::Refused)
        .collect()
}

#[test]
fn markdown_lives_only_in_the_allowed_places() {
    let refused = misplaced(&workspace_root());
    let refused: Vec<&str> = refused.iter().map(String::as_str).collect();
    let message = failure_message(&refused);
    assert!(message.is_empty(), "{message}");
}

#[test]
fn markdown_gate_sees_the_allowed_files() {
    // Guards against a walk that silently checks nothing (wrong root, over-broad skips).
    let files = repo_files(&workspace_root());
    for expected in ["CLAUDE.md", "README.md", "THIRD_PARTY_NOTICES.md"] {
        assert!(
            files.iter().any(|p| p == expected),
            "the walk missed {expected}"
        );
    }
}

#[test]
fn markdown_placement_allows_the_listed_places() {
    for path in [
        "README.md",
        "CLAUDE.md",
        "THIRD_PARTY_NOTICES.md",
        "CHANGELOG.md",
        "LICENSE.md",
        "LICENSE-APACHE.md",
        "crates/solidfem-mesh/README.md",
        "crates/solidfem-mesh/CHANGELOG.md",
        "crates/solidfem-mesh/THIRD_PARTY_NOTICES.md",
        "crates/solidfem-mesh/LICENSE.md",
        "crates/solidfem-elastic/tests/bevel/fixtures/notes.md",
        "crates/solidfem-mesh/tests/fixtures/sample.md",
        ".claude/agents/implementer.md",
        ".claude/skills/release/SKILL.md",
        "docs/later/idea.md",
    ] {
        assert_eq!(placement(path), Placement::Allowed, "{path}");
    }
}

#[test]
fn markdown_placement_refuses_docs_anywhere_else() {
    for path in [
        "crates/README.md",
        "crates/solidfem-mesh/src/README.md",
        "crates/solidfem-mesh/docs/design.md",
        "crates/solidfem-mesh/tests/notes.md",
        "crates/solidfem-mesh/tests/fixtures.md",
        "crates/solidfem-mesh/fixtures/notes.md",
        "docs/ROADMAP.md",
        "docs/plan.md",
        "docs/later/sub/idea.md",
        "crates/x/docs/later/idea.md",
        "CONTRIBUTING.md",
        "scripts/README.md",
    ] {
        assert_eq!(placement(path), Placement::Refused, "{path}");
    }
    // Not Markdown, so not this gate's concern.
    for path in [
        "crates/x/src/lib.rs",
        "docs/later",
        "README",
        ".md",
        "LICENSE",
    ] {
        assert_eq!(placement(path), Placement::NotMarkdown, "{path}");
    }
}

#[test]
fn markdown_placement_matches_extensions_in_any_case() {
    for path in [
        "notes.MD",
        "notes.Md",
        "notes.markdown",
        "scripts/guide.MARKDOWN",
    ] {
        assert_eq!(placement(path), Placement::Refused, "{path}");
    }
    assert_eq!(placement("docs/later/Idea.MD"), Placement::Allowed);
}

#[test]
fn markdown_gate_reads_untracked_files_and_skips_build_trees() {
    let tree = TempTree::new("markdown");
    tree.write("Cargo.toml", "[workspace]\n");
    tree.write("README.md", "readme\n");
    tree.write("docs/later/idea.md", "later\n");
    tree.write("docs/old-plan.md", "done\n");
    tree.write("crates/x/NOTES.md", "a stray doc\n");
    // Build output and VCS trees are never walked.
    for skipped in [
        "target/doc/README.md",
        "node_modules/pkg/README.md",
        ".claude/worktrees/slot1/notes.md",
        ".git/info/notes.md",
        "build-out/CACHEDIR.TAG",
        "build-out/README.md",
    ] {
        tree.write(skipped, "skipped\n");
    }
    assert_eq!(
        misplaced(&tree.0),
        ["crates/x/NOTES.md", "docs/old-plan.md"]
    );
}

#[test]
fn markdown_gate_failure_message_says_what_to_do() {
    let message = failure_message(&["crates/x/src/NOTES.md"]);
    assert!(message.contains("  crates/x/src/NOTES.md\n"));
    assert!(message.contains("into the header comment or rustdoc of the code it is about"));
    assert!(failure_message(&[]).is_empty());
}
