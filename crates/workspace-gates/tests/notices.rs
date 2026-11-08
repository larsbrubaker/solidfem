// Notices gate: every published crate ships the repository's license and its own third-party
// notices, and neither can drift from the root's.
//
// crates.io packages a crate's own folder only, so the root `LICENSE` and `THIRD_PARTY_NOTICES.md`
// would never reach a user of the published crate, and the licenses of the code the crates port
// (MIT, BSL-1.0, ...) require their notices to travel with the source. So each published crate
// (a `crates/<crate>` whose `Cargo.toml` does not say `publish = false`) holds:
//
//   - `LICENSE`, byte for byte the root's;
//   - `THIRD_PARTY_NOTICES.md`: a short header of its own, then exactly the root file's section
//     for that crate, from its `# <package> (`crates/<crate>`)` heading to the next top-level
//     heading. The root file stays the union of every section, and it is the one to edit: a new
//     notice is added there and copied into the crate's file in the same change.
//
// Line endings are compared as text (a checkout may turn LF into CRLF); the license is compared
// as bytes, which a checkout treats the same way for both copies.

use std::fs;
use std::path::Path;

/// The workspace root: this crate sits at `crates/workspace-gates`.
fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/workspace-gates sits two levels below the workspace root")
        .to_path_buf()
}

/// `text` with CRLF read as LF and trailing blank lines dropped.
fn normalized(text: &str) -> String {
    let lf = text.replace("\r\n", "\n");
    format!("{}\n", lf.trim_end())
}

/// One top-level section of a notices file: the crate folder its heading names and its text,
/// heading included.
#[derive(Debug, PartialEq)]
struct Section {
    folder: String,
    text: String,
}

/// The crate folder a top-level heading names, `# <package> (`crates/<folder>`)`, if it is one.
fn folder_of(heading: &str) -> Option<String> {
    let rest = heading.strip_prefix("# ")?;
    let (_, after) = rest.split_once("(`crates/")?;
    let (folder, _) = after.split_once("`)")?;
    Some(folder.to_string())
}

/// Every crate section of a notices file, in order. A section runs from its heading to the next
/// top-level heading; a top-level heading that names no crate (the title) ends the one before it.
fn sections(text: &str) -> Vec<Section> {
    let mut found: Vec<Section> = Vec::new();
    let mut current: Option<Section> = None;
    for line in normalized(text).lines() {
        if line.starts_with("# ") {
            found.extend(current.take());
            current = folder_of(line).map(|folder| Section {
                folder,
                text: String::new(),
            });
        }
        if let Some(section) = current.as_mut() {
            section.text.push_str(line);
            section.text.push('\n');
        }
    }
    found.extend(current);
    for section in &mut found {
        section.text = normalized(&section.text);
    }
    found
}

/// The crate folders under `crates/` that publish (no `publish = false` in their manifest).
fn published_crates(root: &Path) -> Vec<String> {
    let mut crates: Vec<String> = fs::read_dir(root.join("crates"))
        .expect("the crates folder")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().join("Cargo.toml").is_file())
        .filter(|entry| {
            let manifest = fs::read_to_string(entry.path().join("Cargo.toml")).unwrap_or_default();
            !manifest
                .lines()
                .any(|line| line.trim().replace(' ', "") == "publish=false")
        })
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    crates.sort();
    crates
}

#[test]
fn every_published_crate_carries_the_root_license() {
    let root = workspace_root();
    let license = fs::read(root.join("LICENSE")).expect("the root LICENSE");
    for folder in published_crates(&root) {
        let copy = fs::read(root.join("crates").join(&folder).join("LICENSE")).unwrap_or_default();
        assert!(
            copy == license,
            "crates/{folder}/LICENSE is missing or differs from the root LICENSE; copy the root's \
             (crates.io packages only the crate's folder)"
        );
    }
}

#[test]
fn every_published_crate_carries_exactly_its_root_notices() {
    let root = workspace_root();
    let union = fs::read_to_string(root.join("THIRD_PARTY_NOTICES.md")).expect("root notices");
    let root_sections = sections(&union);
    let published = published_crates(&root);

    for folder in &published {
        let at_root: Vec<&Section> = root_sections
            .iter()
            .filter(|section| &section.folder == folder)
            .collect();
        assert_eq!(
            at_root.len(),
            1,
            "THIRD_PARTY_NOTICES.md should have one `# <package> (`crates/{folder}`)` section"
        );
        let path = root
            .join("crates")
            .join(folder)
            .join("THIRD_PARTY_NOTICES.md");
        let own = fs::read_to_string(&path).unwrap_or_default();
        let own_sections = sections(&own);
        assert_eq!(
            own_sections.iter().collect::<Vec<_>>(),
            at_root,
            "crates/{folder}/THIRD_PARTY_NOTICES.md must hold exactly the root file's section for \
             it; edit the root file and copy that section across"
        );
    }

    for section in &root_sections {
        assert!(
            published.contains(&section.folder),
            "THIRD_PARTY_NOTICES.md has a section for crates/{}, which is not a published crate",
            section.folder
        );
    }
}

#[test]
fn the_gate_sees_the_published_crates_and_their_sections() {
    // Guards against a gate that passes by reading nothing.
    let root = workspace_root();
    let published = published_crates(&root);
    for folder in [
        "solidfem",
        "solidfem-elastic",
        "solidfem-mesh",
        "solidfem-solve",
    ] {
        assert!(published.iter().any(|f| f == folder), "missed {folder}");
    }
    assert!(!published.iter().any(|f| f == "workspace-gates"));
    let union = fs::read_to_string(root.join("THIRD_PARTY_NOTICES.md")).expect("root notices");
    assert_eq!(sections(&union).len(), published.len());
}

#[test]
fn a_section_runs_to_the_next_top_level_heading() {
    let text = "# Title\r\nintro\r\n\r\n# a (`crates/x`)\r\n\r\n## one\r\ntext\r\n\r\n# b (`crates/y`)\r\nmore\r\n";
    assert_eq!(
        sections(text),
        vec![
            Section {
                folder: "x".to_string(),
                text: "# a (`crates/x`)\n\n## one\ntext\n".to_string(),
            },
            Section {
                folder: "y".to_string(),
                text: "# b (`crates/y`)\nmore\n".to_string(),
            },
        ]
    );
}
