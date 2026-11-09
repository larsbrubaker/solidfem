// Every surface contract from recovery still holds after refinement, on a block, a plate with
// holes, and a thin hollow box, including the surface unchanged face for face.

use crate::shapes::{hollow_box, plate_with_holes, rotated, subdivided_box, Surface};
use crate::support::{assert_contracts, constant_size, options_with_layers, recovered, run_refine};
use solidfem_tetra::RefineOptions;

fn check_case(surface: &Surface, size: f64, options: RefineOptions) {
    let unrefined = recovered(surface);
    let (refined, report) =
        run_refine(unrefined.clone(), constant_size(size), options).expect("refines");
    assert_contracts(&refined, &unrefined, surface);
    assert_eq!(
        report.after.tets,
        refined.tets.len(),
        "the report counts the final mesh"
    );
}

#[test]
fn a_block_keeps_every_contract() {
    check_case(
        &subdivided_box([0.0; 3], [40.0; 3], 3),
        9.0,
        RefineOptions::default(),
    );
}

#[test]
fn a_plate_with_holes_keeps_every_contract_with_two_layers() {
    let holes = [(1, 1, true), (3, 1, false)];
    check_case(
        &plate_with_holes(20.0, 12.0, 0.8, 5, 3, 2, &holes),
        3.0,
        options_with_layers(2),
    );
}

#[test]
fn a_thin_hollow_box_keeps_every_contract_with_two_layers() {
    check_case(&hollow_box(30.0, 1.0, 6, 5), 8.0, options_with_layers(2));
}

#[test]
fn a_rotated_plate_keeps_every_contract() {
    // Nearly coplanar faces, nothing axis aligned: the layer rays and the exact tests must
    // still agree the points are strictly inside.
    let plate = plate_with_holes(20.0, 12.0, 0.8, 4, 2, 2, &[]);
    let (points, triangles) = rotated(&plate);
    let spun = Surface {
        points,
        triangles,
        ..Surface::default()
    };
    check_case(&spun, 3.0, options_with_layers(2));
}
