// Comparing a part's peak stress between a coarser and a finer mesh: exact where the elements
// represent the stress, falling with refinement where the stress is smooth, and never settling at
// a clamp's edge, where the model's stress is unbounded; and meshes too alike where the peak is,
// or not finer in the region read, are flagged or refused whatever their change.

use solidfem::elastic::{
    ComparedMesh, Material, MeshComparison, Model, ModelError, PeakStress, Solution, Verdict,
    MIN_ELEMENT_RATIO,
};
use solidfem::mesh::{box_tet_mesh, QuadraticTetMesh};

use crate::support::{bar, faces_at, fix_root, run};

/// A bar along x of `size`, clamped at x = 0, with `load` spread over its end face as a traction.
fn clamped_bar(
    size: [f64; 3],
    divisions: [usize; 3],
    material: Material,
    load: [f64; 3],
) -> (QuadraticTetMesh, Solution) {
    solve_clamped(bar(size, divisions), size, material, load)
}

/// `mesh`, a bar along x of `size`, clamped at x = 0, with `load` spread over its end face.
fn solve_clamped(
    mesh: QuadraticTetMesh,
    size: [f64; 3],
    material: Material,
    load: [f64; 3],
) -> (QuadraticTetMesh, Solution) {
    let area = size[1] * size[2];
    let mut model = Model::new(mesh.clone(), material);
    fix_root(&mut model);
    for face in faces_at(&mesh, size[0]) {
        model.add_traction(face, [load[0] / area, load[1] / area, load[2] / area]);
    }
    let solution = run(model);
    (mesh, solution)
}

const CANTILEVER: [f64; 3] = [4.0, 1.0, 1.0];
const CANTILEVER_STEEL: Material = Material {
    youngs_modulus: 1000.0,
    poisson_ratio: 0.3,
};

/// The cantilever under a unit tip shear, at one, two, four and eight elements through the depth:
/// each mesh halves the last one's element size.
fn cantilevers() -> Vec<(QuadraticTetMesh, Solution)> {
    [1usize, 2, 4, 8]
        .into_iter()
        .map(|n| {
            clamped_bar(
                CANTILEVER,
                [4 * n, n, n],
                CANTILEVER_STEEL,
                [0.0, 0.0, -1.0],
            )
        })
        .collect()
}

fn compare_pairs(
    meshes: &[(QuadraticTetMesh, Solution)],
    inside: impl Fn([f64; 3]) -> bool + Copy,
) -> Vec<MeshComparison> {
    meshes
        .windows(2)
        .map(|pair| {
            MeshComparison::between(&pair[0].0, &pair[0].1, &pair[1].0, &pair[1].1, inside)
                .expect("the meshes compare")
        })
        .collect()
}

/// The safety factors are yield over each peak, coarse first, and their change is
/// `(fine - coarse) / fine` of the factors, of the opposite sign to the peak's change: a finer
/// mesh that finds more stress leaves a smaller margin.
fn assert_safety_factors(comparison: &MeshComparison) {
    let yield_strength = 100.0;
    let (coarse, fine) = comparison
        .safety_factors(yield_strength)
        .expect("a valid yield");
    assert_eq!(
        coarse.to_bits(),
        (yield_strength / comparison.coarse().peak.von_mises).to_bits()
    );
    assert_eq!(
        fine.to_bits(),
        (yield_strength / comparison.fine().peak.von_mises).to_bits()
    );
    let change = comparison
        .safety_factor_change(yield_strength)
        .expect("a valid yield");
    assert_eq!(change.to_bits(), ((fine - coarse) / fine).to_bits());
    assert!(
        change * comparison.peak_change() < 0.0,
        "safety factor change {change} has the sign of the peak change {}",
        comparison.peak_change()
    );
}

fn print_pairs(name: &str, comparisons: &[MeshComparison]) {
    for c in comparisons {
        println!(
            "COMPARISON {name}: elements {} -> {}, peak element ratio {:?}, peak {} at {:?} -> {} at {:?}, change {:.4e}, distance {:.3}",
            c.coarse().elements,
            c.fine().elements,
            c.peak_element_ratio(),
            c.coarse().peak.von_mises,
            c.coarse().peak.position,
            c.fine().peak.von_mises,
            c.fine().peak.position,
            c.peak_change(),
            c.peak_distance()
        );
    }
}

#[test]
fn a_bar_in_uniform_tension_reads_the_same_peak_on_both_meshes() {
    // With nu = 0 the clamp restrains nothing and sigma = F / A everywhere, inside the quadratic
    // element's space, so both meshes give F / A up to the solver: `tension.rs` derives a 1e-5
    // relative bound on every stress at the default residual. Two peaks within 1e-5 of F / A
    // differ by at most 2e-5 of the fine one, to first order, and each safety factor is within
    // 1e-5 of yield / (F / A) the same way; 1.001 covers the second-order terms.
    let force = 1000.0;
    let material = Material {
        youngs_modulus: 2100.0,
        poisson_ratio: 0.0,
    };
    let size = [20.0, 1.0, 1.0];
    let (coarse_mesh, coarse) = clamped_bar(size, [10, 1, 1], material, [force, 0.0, 0.0]);
    let (fine_mesh, fine) = clamped_bar(size, [20, 2, 2], material, [force, 0.0, 0.0]);
    let comparison = MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &fine, |_| true)
        .expect("the meshes compare");
    print_pairs("tension", &[comparison]);
    let stress = force / (size[1] * size[2]);
    for peak in [comparison.coarse().peak, comparison.fine().peak] {
        assert!(
            ((peak.von_mises - stress) / stress).abs() < 1e-5,
            "peak {peak:?}, expected {stress}"
        );
    }
    assert_eq!(comparison.coarse().elements, 60);
    assert_eq!(comparison.fine().elements, 480);
    // Both peaks are integration points (positions inside elements), so each peak's volume is its
    // own element's. Every element of a structured box mesh has a sixth of its cell's volume, and
    // the fine cell (1 x 0.5 x 0.5) is the coarse one (2 x 1 x 1) halved, a power of two, so the
    // volumes are exact multiples and the ratio is exactly 8.
    assert_eq!(comparison.peak_element_ratio(), 8.0);
    assert_eq!(
        comparison.verdict(2e-5 * 1.001),
        Verdict::Settled,
        "change {}",
        comparison.peak_change()
    );
    // A NaN tolerance is never met, however small the change.
    assert_eq!(comparison.verdict(f64::NAN), Verdict::StillMoving);
    let yield_strength = 250.0;
    let expected = yield_strength / stress;
    let (coarse_factor, fine_factor) = comparison
        .safety_factors(yield_strength)
        .expect("a valid yield");
    for factor in [coarse_factor, fine_factor] {
        assert!(
            ((factor - expected) / expected).abs() < 1e-5 * 1.001,
            "safety factor {factor}, expected {expected}"
        );
    }
    let factor_change = comparison
        .safety_factor_change(yield_strength)
        .expect("a valid yield");
    assert!(
        factor_change.abs() < 2e-5 * 1.001,
        "safety factor change {factor_change}"
    );
}

#[test]
fn away_from_the_clamp_the_change_falls_as_the_mesh_refines() {
    // Read from two depths beyond the clamp, where its disturbance has decayed, the stress is the
    // smooth bending stress F (L - x) (h / 2) / I = 12 at x = 2, highest at the section's corners.
    // The peak is the recovered value at a corner node there: a linear fit over a patch that is
    // one-sided in y and z, which reads the transverse shear (zero at the surface, rising inward
    // as a parabola) as above zero at the surface, so von Mises reads high by an amount of order
    // h^2. Each finer mesh therefore reads less (a negative change), and quadratic elements make
    // the change fall by about four per halving of the element size. Limit: it falls by more
    // than half at each step, a margin on the measured 0.266 and 0.248.
    //
    // Beam theory puts the region's highest stress on the four corners of the section at x = 2,
    // its edge nearest the clamp. Which corner reads highest is set by the direction every cell's
    // diagonals are cut, the same on each of these nested meshes; measured, it is the node
    // (2, 1, 1) on all four, so the distance between the peaks is exactly zero.
    let meshes = cantilevers();
    let away = |p: [f64; 3]| p[0] >= 2.0 - 1e-9;
    let comparisons = compare_pairs(&meshes, away);
    print_pairs("cantilever_away", &comparisons);
    for comparison in &comparisons {
        assert!(
            comparison.peak_change() < 0.0,
            "change {}: the finer mesh read more",
            comparison.peak_change()
        );
        assert_safety_factors(comparison);
        assert_eq!(comparison.coarse().peak.position, [2.0, 1.0, 1.0]);
        assert_eq!(comparison.fine().peak.position, [2.0, 1.0, 1.0]);
        assert_eq!(comparison.peak_distance(), 0.0);
        // Half the bar is in the region: each mesh has 8 times the last one's elements there.
        assert_eq!(comparison.element_ratio(), 8.0);
        // A node peak's volume is the mean of the elements holding it. The node (2, 1, 1) has the
        // same elements around it on each of these nested meshes, every one a sixth of its cell,
        // and each cell is the last one's halved, a power of two: the ratio is exactly 8.
        assert_eq!(comparison.peak_element_ratio(), 8.0);
        assert_eq!(comparison.verdict(f64::NAN), Verdict::StillMoving);
    }
    for pair in comparisons.windows(2) {
        assert!(
            pair[1].peak_change().abs() < 0.5 * pair[0].peak_change().abs(),
            "change {} did not fall by half from {}",
            pair[1].peak_change(),
            pair[0].peak_change()
        );
    }
}

#[test]
fn including_the_clamp_the_peak_keeps_rising_and_never_settles() {
    // Where the clamped face meets the free surface the model's stress is unbounded, so the peak
    // read there can grow without limit as the elements shrink. How fast it grows at these sizes
    // is measured, not derived, and the peak sits on different things as the mesh refines: at
    // one and two elements through the depth, the recovered value at the midside node half an
    // element from the clamp face on its edge y = 1, z = 0; at four, an integration point next to
    // the clamp face's corner (0, 0, 1); at eight, that face's corner vertex (0, 1, 0). The limits
    // are margins on the measured changes, 0.126, 0.096 and 0.197: every pair's peak rises, none
    // is settled at 5%, and none falls below a quarter of the first pair's change, where a smooth
    // field's change falls by about four per halving (the test above), to about a sixteenth of
    // the first by the third pair.
    let meshes = cantilevers();
    let comparisons = compare_pairs(&meshes, |_| true);
    print_pairs("cantilever_whole", &comparisons);
    let first = comparisons[0].peak_change();
    for comparison in &comparisons {
        assert!(
            comparison.peak_change() > 0.0,
            "the peak fell, {}",
            comparison.peak_change()
        );
        assert_eq!(comparison.verdict(0.05), Verdict::StillMoving);
        assert_safety_factors(comparison);
        assert!(
            comparison.peak_change() > 0.25 * first,
            "change {} fell toward zero from {first}",
            comparison.peak_change()
        );
    }
    // The first two peaks are midside nodes on the same edge, half an element from the clamp
    // face: x = 0.5 at one element through the depth and 0.25 at two, 0.25 apart.
    assert_eq!(comparisons[0].coarse().peak.position, [0.5, 1.0, 0.0]);
    assert_eq!(comparisons[0].fine().peak.position, [0.25, 1.0, 0.0]);
    assert_eq!(comparisons[0].peak_distance(), 0.25);
}

#[test]
fn meshes_too_alike_say_so_whatever_the_change() {
    // With quadratic elements the peak's error falls as h^2, so refining the element edge by r
    // moves the peak by 1 - 1/r^2 of the coarse mesh's error e: 0.75 e for a halving, 0.36 e for
    // the step from 16 x 4 x 4 to 20 x 5 x 5 cells (r = 1.25). Away from the clamp the
    // 16 x 4 x 4 mesh reads e = 0.66% above beam theory (measured, the test above), so a halving
    // moves the peak about 0.49% and the small step about 0.24%. A tolerance of 0.3% lies between
    // the two: the small step's change is within it, yet says nothing, while the halving shows
    // the peak still moving.
    let tolerance = 0.003;
    let away = |p: [f64; 3]| p[0] >= 2.0 - 1e-9;
    let base = clamped_bar(CANTILEVER, [16, 4, 4], CANTILEVER_STEEL, [0.0, 0.0, -1.0]);
    let near = clamped_bar(CANTILEVER, [20, 5, 5], CANTILEVER_STEEL, [0.0, 0.0, -1.0]);
    let halved = clamped_bar(CANTILEVER, [32, 8, 8], CANTILEVER_STEEL, [0.0, 0.0, -1.0]);
    let small_step = MeshComparison::between(&base.0, &base.1, &near.0, &near.1, away)
        .expect("the meshes compare");
    let halving = MeshComparison::between(&base.0, &base.1, &halved.0, &halved.1, away)
        .expect("the meshes compare");
    print_pairs("too_alike", &[small_step, halving]);
    assert_eq!(small_step.coarse().elements, 768);
    // Both peaks are the node (2, 1, 1), with the same elements around it: the volume ratio is
    // the cells' (0.25 / 0.2)^3 = 1.953125, up to the rounding of 0.2, which is not a binary
    // fraction.
    let expected_ratio = 1.953125;
    assert!(
        ((small_step.peak_element_ratio() - expected_ratio) / expected_ratio).abs() < 1e-12,
        "peak element ratio {}",
        small_step.peak_element_ratio()
    );
    assert_eq!(small_step.fine().elements, 1500);
    assert!(small_step.element_ratio() < MIN_ELEMENT_RATIO);
    assert!(
        small_step.peak_change().abs() <= tolerance,
        "the small step moved the peak {}",
        small_step.peak_change()
    );
    assert_eq!(small_step.verdict(tolerance), Verdict::MeshesTooAlike);
    assert_eq!(small_step.verdict(f64::NAN), Verdict::MeshesTooAlike);
    assert_eq!(halving.element_ratio(), 8.0);
    assert_eq!(halving.peak_element_ratio(), 8.0);
    assert_eq!(halving.verdict(tolerance), Verdict::StillMoving);
}

/// The cantilever on cells whose x widths are `widths` in order (one cell through the depth),
/// clamped at x = 0 and loaded at its tip as in [`cantilevers`]. `box_tet_mesh` cuts unit cells,
/// whose x coordinates are the exact integers 0..=n, and each is moved to its cell boundary.
fn graded_cantilever(widths: &[f64]) -> (QuadraticTetMesh, Solution) {
    let mut planes = vec![0.0];
    for width in widths {
        let last = planes[planes.len() - 1];
        planes.push(last + width);
    }
    let length = planes[planes.len() - 1];
    let cells = widths.len();
    let linear = box_tet_mesh([0.0; 3], [cells as f64, 1.0, 1.0], [cells, 1, 1])
        .expect("box")
        .with_mapped_positions(|p| [planes[p[0] as usize], p[1], p[2]])
        .expect("a positive mesh");
    solve_clamped(
        QuadraticTetMesh::from_linear(&linear),
        [length, 1.0, 1.0],
        CANTILEVER_STEEL,
        [0.0, 0.0, -1.0],
    )
}

#[test]
fn a_mesh_refined_only_outside_the_region_read_is_not_finer_there() {
    // The coarse cantilever has unit cells; each fine one cuts x < 2 into 16 cells of 0.125. One
    // leaves the region x >= 2 at its two unit cells, the same 12 elements as the coarse mesh, so
    // the comparison is refused though the fine mesh has 108 elements to 24. The other halves
    // the cells there in x only: twice the elements in the region, and 5 times in the whole mesh.
    //
    // The verdict reads the volume at each peak. A cell's six elements walk its diagonal from
    // its lowest corner to its highest, so those two corners are in all six and every other
    // corner in two. Read over x >= 2 both peaks are the node (2, 1, 1): the highest corner of
    // the cell before it (6 elements) and a corner of the cell after it (2). On the coarse mesh
    // all eight are unit cells' elements; on the fine one six are in 0.125-wide cells and two in
    // 0.5-wide ones, a mean width of (6 x 0.125 + 2 x 0.5) / 8 = 7/32, so the ratio is 32/7, over
    // the minimum: the node's value is fitted to all eight, and six of them were refined. Read
    // over the whole part the peaks lie on the clamp side, the fine one inside a 0.125-wide
    // cell, so the ratio is the cells' 1 / 0.125 = 8, though only x was refined: a volume ratio
    // counts a thinning in one direction as fully as one in all three.
    let away = |p: [f64; 3]| p[0] >= 2.0 - 1e-9;
    let mut outside_only = vec![0.125; 16];
    outside_only.extend([1.0, 1.0]);
    let mut halved_in_x = vec![0.125; 16];
    halved_in_x.extend([0.5; 4]);
    let coarse = graded_cantilever(&[1.0; 4]);
    let same_inside = graded_cantilever(&outside_only);
    let twice_inside = graded_cantilever(&halved_in_x);
    assert_eq!(coarse.0.tets().len(), 24);
    assert_eq!(same_inside.0.tets().len(), 108);
    assert_eq!(twice_inside.0.tets().len(), 120);
    assert_eq!(
        MeshComparison::between(&coarse.0, &coarse.1, &same_inside.0, &same_inside.1, away),
        Err(ModelError::FineMeshNotFiner {
            coarse_elements: 12,
            fine_elements: 12,
        })
    );
    let in_region =
        MeshComparison::between(&coarse.0, &coarse.1, &twice_inside.0, &twice_inside.1, away)
            .expect("the meshes compare");
    let whole = MeshComparison::between(
        &coarse.0,
        &coarse.1,
        &twice_inside.0,
        &twice_inside.1,
        |_| true,
    )
    .expect("the meshes compare");
    print_pairs("graded", &[in_region, whole]);
    assert_eq!(in_region.element_ratio(), 2.0);
    assert_eq!(in_region.fine().peak.position, [2.0, 1.0, 1.0]);
    let expected = 32.0 / 7.0;
    assert!(
        ((in_region.peak_element_ratio() - expected) / expected).abs() < 1e-12,
        "peak element ratio {}",
        in_region.peak_element_ratio()
    );
    assert_eq!(in_region.verdict(f64::NAN), Verdict::StillMoving);
    assert_eq!(whole.element_ratio(), 5.0);
    assert_eq!(whole.fine().peak.position, [0.3125, 1.0, 0.0]);
    assert_eq!(whole.peak_element_ratio(), 8.0);
    assert_eq!(whole.verdict(f64::NAN), Verdict::StillMoving);
}

#[test]
fn a_region_refined_away_from_its_peak_reads_too_alike() {
    // A host that halves only its global size while a feature stays at a floor size: the region
    // x >= 2 gains elements toward the tip (4.5 times as many, over the minimum), but the peak at
    // (2, 1, 1) is held by the same elements, all of unit cells, on both meshes (the six of the
    // cell before it and two of the cell after it, see the test above). Its volume ratio is
    // exactly 1, so the comparison reads too alike whatever the change, which is small: the
    // peak's own elements did not change.
    let away = |p: [f64; 3]| p[0] >= 2.0 - 1e-9;
    let mut tip_only = vec![1.0; 3];
    tip_only.extend([0.125; 8]);
    let coarse = graded_cantilever(&[1.0; 4]);
    let fine = graded_cantilever(&tip_only);
    let comparison = MeshComparison::between(&coarse.0, &coarse.1, &fine.0, &fine.1, away)
        .expect("the meshes compare");
    print_pairs("tip_only", &[comparison]);
    assert_eq!(comparison.coarse().elements, 12);
    assert_eq!(comparison.fine().elements, 54);
    assert!(comparison.element_ratio() >= MIN_ELEMENT_RATIO);
    assert_eq!(comparison.coarse().peak.position, [2.0, 1.0, 1.0]);
    assert_eq!(comparison.fine().peak.position, [2.0, 1.0, 1.0]);
    assert_eq!(comparison.peak_element_ratio(), 1.0);
    // Within a 1% tolerance, a margin on the measured +0.26%, so a region-wide count would have
    // read it as settled.
    assert!(comparison.peak_change().abs() <= 0.01);
    assert_eq!(comparison.verdict(0.01), Verdict::MeshesTooAlike);
}

#[test]
fn each_refusal_is_named() {
    let material = Material {
        youngs_modulus: 1000.0,
        poisson_ratio: 0.0,
    };
    let size = [2.0, 1.0, 1.0];
    let (coarse_mesh, coarse) = clamped_bar(size, [2, 1, 1], material, [0.0, 0.0, -1.0]);
    let (fine_mesh, fine) = clamped_bar(size, [4, 2, 2], material, [0.0, 0.0, -1.0]);

    let swapped = MeshComparison::between(&fine_mesh, &fine, &coarse_mesh, &coarse, |_| true);
    assert_eq!(
        swapped,
        Err(ModelError::FineMeshNotFiner {
            coarse_elements: 96,
            fine_elements: 12,
        })
    );
    let same = MeshComparison::between(&coarse_mesh, &coarse, &coarse_mesh, &coarse, |_| true);
    assert!(matches!(same, Err(ModelError::FineMeshNotFiner { .. })));

    // The clamp face holds nodes but no integration point: a face alone has no volume to read.
    assert_eq!(
        MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &fine, |p| p[0] == 0.0),
        Err(ModelError::NothingInsideRegion {
            mesh: ComparedMesh::Coarse
        })
    );
    // Nothing of the bar lies beyond x = 10.
    let outside =
        MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &fine, |p| p[0] > 10.0);
    assert_eq!(
        outside,
        Err(ModelError::NothingInsideRegion {
            mesh: ComparedMesh::Coarse
        })
    );
    // Only the coarse mesh has an integration point exactly there.
    let coarse_point = coarse.point_stresses[0].position;
    let only_coarse = MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &fine, |p| {
        p == coarse_point
    });
    assert_eq!(
        only_coarse,
        Err(ModelError::NothingInsideRegion {
            mesh: ComparedMesh::Fine
        })
    );

    let mut unstressed = fine.clone();
    unstressed.nodal_von_mises.iter_mut().for_each(|v| *v = 0.0);
    unstressed
        .point_stresses
        .iter_mut()
        .for_each(|p| p.von_mises = 0.0);
    assert_eq!(
        MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &unstressed, |_| true),
        Err(ModelError::ZeroPeak {
            mesh: ComparedMesh::Fine,
        })
    );
    let mut broken = coarse.clone();
    broken.nodal_von_mises[0] = f64::INFINITY;
    assert_eq!(
        MeshComparison::between(&coarse_mesh, &broken, &fine_mesh, &fine, |_| true),
        Err(ModelError::NonFinitePeak {
            mesh: ComparedMesh::Coarse,
            von_mises: f64::INFINITY,
        })
    );
    // A NaN anywhere in the region is the peak, never dropped behind a finite one.
    let mut not_a_number = coarse.clone();
    not_a_number.point_stresses[5].von_mises = f64::NAN;
    match MeshComparison::between(&coarse_mesh, &not_a_number, &fine_mesh, &fine, |_| true) {
        Err(ModelError::NonFinitePeak {
            mesh: ComparedMesh::Coarse,
            von_mises,
        }) => assert!(von_mises.is_nan()),
        other => panic!("a NaN peak was not refused: {other:?}"),
    }

    // The coarse solution passed with the fine mesh: its arrays are the coarse mesh's.
    assert_eq!(
        MeshComparison::between(&fine_mesh, &coarse, &fine_mesh, &fine, |_| true),
        Err(ModelError::SolutionMeshMismatch {
            mesh: ComparedMesh::Coarse
        })
    );
    let mut short = fine.clone();
    short.point_stresses.pop();
    assert_eq!(
        MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &short, |_| true),
        Err(ModelError::SolutionMeshMismatch {
            mesh: ComparedMesh::Fine
        })
    );

    let comparison = MeshComparison::between(&coarse_mesh, &coarse, &fine_mesh, &fine, |_| true)
        .expect("the meshes compare");
    for yield_strength in [0.0, -250.0, f64::INFINITY] {
        assert_eq!(
            comparison.safety_factors(yield_strength),
            Err(ModelError::InvalidYieldStrength { yield_strength })
        );
        assert_eq!(
            comparison.safety_factor_change(yield_strength),
            Err(ModelError::InvalidYieldStrength { yield_strength })
        );
    }
    assert!(matches!(
        comparison.safety_factors(f64::NAN),
        Err(ModelError::InvalidYieldStrength { .. })
    ));
}

#[test]
fn the_peak_with_its_position_is_the_reported_peak() {
    // The peak with a position must be the very value `peak_von_mises` reports, over the whole
    // part and over a region, and its position must hold that value.
    let material = Material {
        youngs_modulus: 1000.0,
        poisson_ratio: 0.3,
    };
    let (mesh, solution) = clamped_bar([2.0, 1.0, 1.0], [4, 2, 2], material, [0.0, 0.0, -1.0]);
    let regions: [fn([f64; 3]) -> bool; 2] = [|_| true, |p| p[0] >= 1.0];
    for inside in regions {
        let value = solution.peak_von_mises(&mesh, inside).expect("a peak");
        let PeakStress {
            von_mises,
            position,
        } = solution.peak_stress(&mesh, inside).expect("a peak");
        assert_eq!(von_mises.to_bits(), value.to_bits());
        assert!(inside(position));
        let at_node = mesh
            .nodes()
            .iter()
            .zip(&solution.nodal_von_mises)
            .any(|(p, v)| *p == position && *v == von_mises);
        let at_point = solution
            .point_stresses
            .iter()
            .any(|p| p.position == position && p.von_mises == von_mises);
        assert!(
            at_node || at_point,
            "no candidate {von_mises} at {position:?}"
        );
    }
}

#[test]
fn equal_peaks_report_the_first_in_node_then_point_order() {
    let material = Material {
        youngs_modulus: 1000.0,
        poisson_ratio: 0.0,
    };
    let (mesh, mut solution) = clamped_bar([2.0, 1.0, 1.0], [2, 1, 1], material, [0.0, 0.0, -1.0]);
    solution.nodal_von_mises.iter_mut().for_each(|v| *v = 1.0);
    solution
        .point_stresses
        .iter_mut()
        .for_each(|p| p.von_mises = 1.0);
    let peak = solution.peak_stress(&mesh, |_| true).expect("a peak");
    assert_eq!(peak.position, mesh.nodes()[0]);
    // With every node outside, the first integration point holds the tie.
    let first_point = solution.point_stresses[0].position;
    let on_nodes = |p: [f64; 3]| mesh.nodes().contains(&p);
    let peak = solution
        .peak_stress(&mesh, |p| !on_nodes(p))
        .expect("a peak");
    assert_eq!(peak.position, first_point);
}
