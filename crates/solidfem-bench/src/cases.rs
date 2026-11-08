//! The printable-part benchmark cases, in millimetres and PLA-like material, at grid spacing t/3 of
//! the thinnest wall t, the finest spacing the mesher's thin-wall limit needs. Each holds one region
//! and loads another with a total force independent of the mesh.

use solidfem_elastic::Material;

use crate::pipeline::Case;
use crate::shapes::{box_sdf, hollow_box, l_bracket, ribbed_plate, tube};

/// A PLA-like material.
pub const PLA: Material = Material {
    youngs_modulus: 3500.0,
    poisson_ratio: 0.35,
};

/// The printable-part cases, in the order the tables list them.
pub fn part_cases() -> Vec<Case> {
    vec![
        l_bracket_case(),
        hollow_box_case(2.0),
        hollow_box_case(1.2),
        hollow_box_case(0.8),
        tube_case(),
        ribbed_plate_case(),
    ]
}

/// An L-bracket with 3 mm walls, about 60 mm long: base bolted down, wall top pushed sideways.
pub fn l_bracket_case() -> Case {
    let (length, width, wall) = (60.0, 30.0, 3.0);
    Case {
        name: "L-bracket, 3 mm walls".to_string(),
        sdf: Box::new(move |p| l_bracket(p, length, width, wall)),
        low: [0.0; 3],
        high: [length, width, length],
        dx: wall / 3.0,
        fixed: Box::new(move |p, _| p[0] > length - 10.0 && p[2] <= wall + 1e-9),
        loaded: Box::new(move |p, tolerance| (p[2] - length).abs() < tolerance),
        total_force: [100.0, 0.0, 0.0],
        material: PLA,
        grid_shift_cells: 0.0,
    }
}

/// A closed 60 mm hollow box with the given wall, held by its base and pressed on the top.
pub fn hollow_box_case(wall: f64) -> Case {
    let size = 60.0;
    Case {
        name: format!("hollow box, {wall} mm walls"),
        sdf: Box::new(move |p| hollow_box(p, size, wall)),
        low: [0.0; 3],
        high: [size; 3],
        dx: wall / 3.0,
        fixed: Box::new(|p, tolerance| p[2].abs() < tolerance),
        loaded: Box::new(move |p, tolerance| (p[2] - size).abs() < tolerance),
        total_force: [0.0, 0.0, -100.0],
        material: PLA,
        grid_shift_cells: 0.0,
    }
}

/// A smaller hollow box the browser's 32-bit memory can hold, 30 mm with a 2 mm wall.
pub fn small_hollow_box_case() -> Case {
    let (size, wall) = (30.0, 2.0);
    Case {
        name: format!("hollow box {size} mm, {wall} mm walls"),
        sdf: Box::new(move |p| hollow_box(p, size, wall)),
        low: [0.0; 3],
        high: [size; 3],
        dx: wall / 3.0,
        fixed: Box::new(|p, tolerance| p[2].abs() < tolerance),
        loaded: Box::new(move |p, tolerance| (p[2] - size).abs() < tolerance),
        total_force: [0.0, 0.0, -100.0],
        material: PLA,
        grid_shift_cells: 0.0,
    }
}

/// A 1.2 mm wall tube about 80 mm long: one end clamped, a transverse force on the other.
pub fn tube_case() -> Case {
    let (length, outer, wall) = (80.0, 10.0, 1.2);
    let center = [outer, outer];
    Case {
        name: "tube, 1.2 mm wall".to_string(),
        sdf: Box::new(move |p| tube(p, length, outer, wall, center)),
        low: [0.0; 3],
        high: [length, 2.0 * outer, 2.0 * outer],
        dx: wall / 3.0,
        fixed: Box::new(|p, tolerance| p[0].abs() < tolerance),
        loaded: Box::new(move |p, tolerance| (p[0] - length).abs() < tolerance),
        total_force: [0.0, 0.0, -50.0],
        material: PLA,
        grid_shift_cells: 0.0,
    }
}

/// A ribbed plate with 2 mm ribs, 80 mm square: one edge clamped, a force on the opposite edge.
pub fn ribbed_plate_case() -> Case {
    let (size, plate, rib, height, pitch) = (80.0, 2.0, 2.0, 8.0, 20.0);
    Case {
        name: "ribbed plate, 2 mm ribs".to_string(),
        sdf: Box::new(move |p| ribbed_plate(p, size, plate, rib, height, pitch)),
        low: [0.0; 3],
        high: [size, size, plate + height],
        dx: rib / 3.0,
        fixed: Box::new(|p, tolerance| p[0].abs() < tolerance),
        loaded: Box::new(move |p, tolerance| (p[0] - size).abs() < tolerance),
        total_force: [0.0, 0.0, -50.0],
        material: PLA,
        grid_shift_cells: 0.0,
    }
}

/// A solid block used by the browser page as a fast smoke case.
pub fn block_case() -> Case {
    let size = 10.0;
    Case {
        name: "block (smoke)".to_string(),
        sdf: Box::new(move |p| box_sdf(p, [0.0; 3], [size; 3])),
        low: [0.0; 3],
        high: [size; 3],
        dx: 2.0,
        fixed: Box::new(|p, tolerance| p[2].abs() < tolerance),
        loaded: Box::new(move |p, tolerance| (p[2] - size).abs() < tolerance),
        total_force: [0.0, 0.0, -10.0],
        material: PLA,
        grid_shift_cells: 0.0,
    }
}
