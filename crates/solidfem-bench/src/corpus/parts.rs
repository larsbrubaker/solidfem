//! Synthetic parts typical of what MatterCAD produces, built through solidmesh's real operations:
//! booleans on manifold-rust, extrudes of 2D outlines, a rounded-corner box and a real fillet.
//! Each builder is deterministic, so a run measures the same meshes every time.

use std::error::Error;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

use solidmesh::math::{Matrix4X4, Vector2, Vector3};
use solidmesh::mesh::csg::boolean_processing::do_array;
use solidmesh::mesh::csg::csg_operations::{subtract, union};
use solidmesh::mesh::csg::{BooleanOptions, CsgModes};
use solidmesh::mesh::processors::outline_to_mesh::extrude_contours;
use solidmesh::mesh::solids::platonic_solids::create_cube;
use solidmesh::mesh::solids::RoundedCornerBox;
use solidmesh::mesh::{CancelToken, Mesh};
use solidmesh::{
    bevel_part, pick, BevelPartRequest, BevelSelectionKind, FaceGroup, FeatureEdgeSet,
};

/// Circle facets for a through-hole or boss, the count a modest CAD tessellation produces.
const CIRCLE_SIDES: usize = 48;

/// One named part.
pub struct Part {
    pub name: &'static str,
    pub mesh: Mesh,
}

type Build = Result<Mesh, Box<dyn Error>>;

type Builder = (&'static str, fn() -> Build);

fn circle(radius: f64, sides: usize) -> Vec<Vector2> {
    (0..sides)
        .map(|i| {
            let angle = TAU * i as f64 / sides as f64;
            Vector2::new(radius * angle.cos(), radius * angle.sin())
        })
        .collect()
}

/// A cylinder of `radius` along +z from z=0 to `height`, centred on the origin.
fn cylinder(radius: f64, height: f64) -> Build {
    Ok(extrude_contours(&[circle(radius, CIRCLE_SIDES)], height)?)
}

fn translated(mut mesh: Mesh, x: f64, y: f64, z: f64) -> Mesh {
    mesh.transform(Matrix4X4::create_translation(x, y, z));
    mesh
}

fn box_with_hole() -> Build {
    let body = create_cube(40.0, 40.0, 20.0);
    let hole = translated(cylinder(6.0, 40.0)?, 0.0, 0.0, -20.0);
    Ok(subtract(&body, &hole, &CancelToken::new())?)
}

fn bracket() -> Build {
    let cancel = CancelToken::new();
    let base = translated(create_cube(80.0, 40.0, 5.0), 0.0, 0.0, 2.5);
    let upright = translated(create_cube(5.0, 40.0, 50.0), -37.5, 0.0, 30.0);
    let mut part = union(&base, &upright, &cancel)?;
    for (x, y) in [(10.0, -12.0), (10.0, 12.0), (30.0, -12.0), (30.0, 12.0)] {
        let hole = translated(cylinder(3.0, 30.0)?, x, y, -10.0);
        part = subtract(&part, &hole, &cancel)?;
    }
    for (y, z) in [(-12.0, 25.0), (12.0, 25.0), (0.0, 42.0)] {
        let mut hole = cylinder(3.0, 30.0)?;
        hole.transform(Matrix4X4::create_rotation_y(FRAC_PI_2));
        let hole = translated(hole, -55.0, y, z);
        part = subtract(&part, &hole, &cancel)?;
    }
    Ok(part)
}

fn hollow_enclosure_with_bosses() -> Build {
    let cancel = CancelToken::new();
    let outer = create_cube(60.0, 40.0, 30.0);
    let cavity = translated(create_cube(56.0, 36.0, 26.0), 0.0, 0.0, 0.0);
    let mut part = subtract(&outer, &cavity, &cancel)?;
    for (x, y) in [(-22.0, -12.0), (22.0, -12.0), (-22.0, 12.0), (22.0, 12.0)] {
        let boss = translated(cylinder(4.0, 16.0)?, x, y, -13.0);
        let pilot = translated(cylinder(1.5, 16.0)?, x, y, -8.0);
        part = union(&part, &boss, &cancel)?;
        part = subtract(&part, &pilot, &cancel)?;
    }
    Ok(part)
}

fn rounded_box() -> Build {
    Ok(RoundedCornerBox::create(
        6,
        Vector3::new(30.0, 20.0, 10.0),
        3.0,
    ))
}

fn filleted_box() -> Build {
    let cube = Arc::new(create_cube(30.0, 20.0, 10.0));
    let edges = FeatureEdgeSet::create_with_angle(&cube, 20.0);
    let picked = pick::group_from_edge(
        &edges,
        0,
        BevelSelectionKind::Edge,
        FaceGroup::DEFAULT_CURVE_TOLERANCE_DEGREES,
    )?;
    let request = BevelPartRequest::new(Some(cube), vec![picked], 2.0, 8, true, 20.0);
    let outcome = bevel_part(request, &CancelToken::new(), None)?;
    Ok(outcome.mesh.ok_or("the fillet produced no mesh")?)
}

fn plate_with_hole_grid() -> Build {
    let plate = translated(create_cube(120.0, 120.0, 4.0), 0.0, 0.0, 2.0);
    let drill = translated(cylinder(2.0, 20.0)?, 0.0, 0.0, -8.0);
    let mut offsets = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            offsets.push((-54.0 + 12.0 * i as f64, -54.0 + 12.0 * j as f64));
        }
    }
    let matrices: Vec<Matrix4X4> = offsets
        .iter()
        .map(|&(x, y)| Matrix4X4::create_translation(x, y, 0.0))
        .collect();
    let mut items: Vec<(&Mesh, Matrix4X4)> = vec![(&plate, Matrix4X4::IDENTITY)];
    items.extend(matrices.iter().map(|&m| (&drill, m)));
    Ok(do_array(
        &items,
        CsgModes::Subtract,
        &CancelToken::new(),
        &BooleanOptions::default(),
    )?)
}

fn extruded_curved_outline() -> Build {
    let outer: Vec<Vector2> = (0..360)
        .map(|i| {
            let angle = TAU * i as f64 / 360.0;
            let radius = 30.0 + 6.0 * (angle * 7.0).sin() + 2.5 * (angle * 13.0).cos();
            Vector2::new(radius * angle.cos(), radius * angle.sin())
        })
        .collect();
    let hole: Vec<Vector2> = circle(8.0, 96).into_iter().rev().collect();
    Ok(extrude_contours(&[outer, hole], 6.0)?)
}

fn thin_walled_shell() -> Build {
    let cancel = CancelToken::new();
    let outer = create_cube(60.0, 40.0, 20.0);
    let cavity = translated(create_cube(58.8, 38.8, 19.0), 0.0, 0.0, 0.5);
    Ok(subtract(&outer, &cavity, &cancel)?)
}

/// Every synthetic part, or the build error of the one that failed.
pub fn all_parts() -> Result<Vec<Part>, Box<dyn Error>> {
    let builders: [Builder; 8] = [
        ("box-with-hole", box_with_hole),
        ("bracket-holes", bracket),
        ("enclosure-bosses", hollow_enclosure_with_bosses),
        ("rounded-box", rounded_box),
        ("filleted-box", filleted_box),
        ("plate-hole-grid", plate_with_hole_grid),
        ("extruded-curves", extruded_curved_outline),
        ("thin-shell", thin_walled_shell),
    ];
    builders
        .into_iter()
        .map(|(name, build)| {
            Ok(Part {
                name,
                mesh: build()?,
            })
        })
        .collect()
}
