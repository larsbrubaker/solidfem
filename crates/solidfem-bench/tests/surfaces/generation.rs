//! Remeshing a real part: regenerating the same surface gives the same bytes, so a benchmark
//! result can be reproduced by anyone. A bad request is a named error, not a panic. Uses the
//! snap-fit beam, the smallest part.

use solidfem_bench::surfaces::{
    all_parts, demo_only_parts, remesh_density, surface_text, SurfaceError,
};
use solidmesh_mesh::CancelToken;

/// The surface written at `snap_h1.0` regenerates identically, and its header line states the
/// counts of what follows it, which the benchmark reader relies on.
#[test]
fn a_surface_regenerates_byte_for_byte_and_reports_its_own_counts() {
    let parts = all_parts();
    let part = parts.iter().find(|p| p.name == "snap").expect("snap");
    let density = &part.densities[0];
    assert_eq!(density.stem, "snap_h1.0");
    let cancel = CancelToken::new();
    let mesh = part.mesh(&cancel).expect("the part builds");
    let first = remesh_density(&mesh, part, density, &cancel).expect("the part remeshes");
    let second = remesh_density(&mesh, part, density, &cancel).expect("the part remeshes");
    let text = surface_text(&first.surface);
    assert_eq!(text, surface_text(&second.surface), "two runs differed");
    let header = text.lines().next().expect("a header line");
    assert_eq!(
        header,
        format!(
            "{} {}",
            first.surface.vertices.len(),
            first.surface.faces.len()
        )
    );
    // A closed triangle surface has 2 V - 4 faces when it has genus zero, as this beam does.
    assert_eq!(
        first.surface.faces.len(),
        2 * first.surface.vertices.len() - 4,
        "the beam's surface is not a closed genus-zero solid"
    );
}

#[test]
fn a_size_the_remesher_cannot_meet_is_a_named_refusal() {
    use solidfem_bench::benchmarks::cases::Sizing;
    use solidfem_bench::surfaces::SurfaceDensity;
    let parts = all_parts();
    let part = parts.iter().find(|p| p.name == "snap").expect("snap");
    let density = SurfaceDensity {
        stem: "snap_impossible".to_string(),
        // A non-positive size is refused by the remesher; the stem should reach the error.
        size: Sizing::Uniform(0.0),
    };
    let cancel = CancelToken::new();
    let mesh = part.mesh(&cancel).expect("the part builds");
    let Err(error) = remesh_density(&mesh, part, &density, &cancel) else {
        panic!("a zero edge length must be refused");
    };
    assert!(matches!(&error, SurfaceError::Refused { stem, .. } if stem == "snap_impossible"));
    assert!(error.to_string().contains("snap_impossible"), "{error}");
}

/// The demo page's surfaces (`demo/public/parts/`) are exactly what the generator writes, so the
/// page's parts are the benchmark's parts and the file is not a hand-kept copy. Regenerate them
/// with `solidfem-surfaces demo/public/parts <stem> ...` when this fails after a generator change.
#[test]
fn the_demo_surfaces_are_what_the_generator_produces() {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/../../demo/public/parts");
    let cancel = CancelToken::new();
    let parts: Vec<_> = solidfem_bench::surfaces::all_parts()
        .into_iter()
        .chain(demo_only_parts())
        .collect();
    let mut checked = 0;
    for entry in std::fs::read_dir(directory).expect("the demo's surfaces") {
        let path = entry.expect("a directory entry").path();
        let stem = path
            .file_stem()
            .expect("a surface file has a stem")
            .to_string_lossy()
            .into_owned();
        let (part, density) = parts
            .iter()
            .find_map(|part| {
                part.densities
                    .iter()
                    .find(|density| density.stem == stem)
                    .map(|density| (part, density))
            })
            .unwrap_or_else(|| panic!("{stem} is not a surface the generator writes"));
        let mesh = part.mesh(&cancel).expect("the part builds");
        let remeshed = remesh_density(&mesh, part, density, &cancel).expect("the part remeshes");
        let committed = std::fs::read_to_string(&path).expect("the committed surface");
        assert_eq!(
            committed,
            surface_text(&remeshed.surface),
            "{stem} differs from what the generator writes; rerun solidfem-surfaces"
        );
        checked += 1;
    }
    assert!(checked > 0, "no demo surfaces were found");
}
