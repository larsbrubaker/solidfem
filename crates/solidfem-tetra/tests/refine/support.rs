// Reading refinement output from the outside: boundary faces, a walk along a straight line through
// the output tetrahedra using their neighbour links, and a minimum dihedral angle computed here,
// separately from the library's.

use std::collections::BTreeSet;

use robust::{orient3d, Coord3D};
use solidfem_tetra::{RefineOptions, RefineReport, Refiner, SizeFn, SolidMesh, StepStatus};

use crate::check::{assert_solid, mesh};
use crate::shapes::Surface;

fn coord(p: [f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

pub fn recovered(surface: &Surface) -> SolidMesh {
    let (solid, _) = mesh(&surface.points, &surface.triangles).expect("surface recovers");
    solid
}

pub fn constant_size(size: f64) -> SizeFn {
    Box::new(move |_| size)
}

/// Runs refinement to the end, returning the mesh, the report, or the error it stopped with.
pub fn run_refine(
    solid: SolidMesh,
    size: SizeFn,
    options: RefineOptions,
) -> Result<(SolidMesh, RefineReport), solidfem_tetra::TetraError> {
    let mut refiner = Refiner::new(solid, size, options)?;
    assert_eq!(refiner.step(usize::MAX, &mut || false)?, StepStatus::Done);
    Ok((refiner.result()?, refiner.report()))
}

pub fn options_with_layers(layers: u32) -> RefineOptions {
    RefineOptions {
        layers_through_walls: layers,
        ..RefineOptions::default()
    }
}

/// Every prior surface contract, and the surface unchanged by refinement face for face.
pub fn assert_contracts(refined: &SolidMesh, unrefined: &SolidMesh, surface: &Surface) {
    assert_solid(refined, &surface.points, &surface.triangles);
    assert_eq!(
        boundary_faces(refined),
        boundary_faces(unrefined),
        "refinement changed a boundary face"
    );
    assert_eq!(
        &refined.points[..surface.points.len()],
        &unrefined.points[..surface.points.len()],
        "a surface vertex moved"
    );
}

/// Each boundary face as its three sorted point indices.
pub fn boundary_faces(mesh: &SolidMesh) -> BTreeSet<[u32; 3]> {
    let mut faces = BTreeSet::new();
    for (t, tet) in mesh.tets.iter().enumerate() {
        for i in 0..4 {
            if mesh.neighbors[t][i] == solidfem_tetra::HULL {
                let mut face: Vec<u32> = (0..4).filter(|&j| j != i).map(|j| tet[j]).collect();
                face.sort_unstable();
                faces.insert([face[0], face[1], face[2]]);
            }
        }
    }
    faces
}

fn crosses(mesh: &SolidMesh, tet: [u32; 4], skip: usize, low: [f64; 3], high: [f64; 3]) -> bool {
    let vertices: Vec<[f64; 3]> = (0..4)
        .filter(|&i| i != skip)
        .map(|i| mesh.points[tet[i] as usize])
        .collect();
    let (a, b, c) = (coord(vertices[0]), coord(vertices[1]), coord(vertices[2]));
    let (s1, s2) = (
        orient3d(a, b, c, coord(low)),
        orient3d(a, b, c, coord(high)),
    );
    if s1 * s2 >= 0.0 {
        return false;
    }
    let sides = [
        orient3d(coord(low), coord(high), a, b),
        orient3d(coord(low), coord(high), b, c),
        orient3d(coord(low), coord(high), c, a),
    ];
    sides.iter().all(|&s| s > 0.0) || sides.iter().all(|&s| s < 0.0)
}

/// Walks the segment `low..high` through the mesh from where it first enters, returning the
/// tetrahedra traversed in order. Panics if the walk does not close, which would mean broken
/// adjacency. Sample positions must not pass through mesh edges.
pub fn column_walk(mesh: &SolidMesh, low: [f64; 3], high: [f64; 3]) -> Vec<usize> {
    let mut entry: Option<(usize, usize, f64)> = None;
    for (t, tet) in mesh.tets.iter().enumerate() {
        for i in 0..4 {
            if mesh.neighbors[t][i] != solidfem_tetra::HULL || !crosses(mesh, *tet, i, low, high) {
                continue;
            }
            let height = (0..4)
                .filter(|&j| j != i)
                .map(|j| mesh.points[tet[j] as usize][2])
                .fold(f64::INFINITY, f64::min);
            if entry.is_none_or(|(_, _, best)| height < best) {
                entry = Some((t, i, height));
            }
        }
    }
    let (mut t, mut through, _) = entry.expect("the column enters the solid");
    let mut seen = vec![t];
    for _ in 0..mesh.tets.len() {
        let tet = mesh.tets[t];
        let exit = (0..4)
            .find(|&i| i != through && crosses(mesh, tet, i, low, high))
            .expect("the walk leaves this tetrahedron through a face");
        let next = mesh.neighbors[t][exit];
        if next == solidfem_tetra::HULL {
            return seen;
        }
        let back = (0..4)
            .find(|&i| !tet.contains(&mesh.tets[next as usize][i]))
            .expect("neighbours share a face");
        t = next as usize;
        through = back;
        seen.push(t);
    }
    panic!("the walk did not leave the mesh");
}

/// The smallest of a tetrahedron's six dihedral angles, in degrees, computed along each edge.
pub fn min_dihedral(mesh: &SolidMesh, tet: &[u32; 4]) -> f64 {
    let p = tet.map(|v| mesh.points[v as usize]);
    let mut smallest = 180.0_f64;
    for (i, j, k, l) in [
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (1, 3, 0, 2),
        (2, 3, 0, 1),
    ] {
        // The dihedral along edge (i, j), between the half-planes towards k and l.
        let e = unit(sub(p[j], p[i]));
        let perpendicular = |v: [f64; 3]| {
            let d = sub(v, p[i]);
            let along = dot(d, e);
            unit([
                d[0] - e[0] * along,
                d[1] - e[1] * along,
                d[2] - e[2] * along,
            ])
        };
        let cosine = dot(perpendicular(p[k]), perpendicular(p[l])).clamp(-1.0, 1.0);
        smallest = smallest.min(cosine.acos().to_degrees());
    }
    smallest
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

pub fn edge_length(mesh: &SolidMesh, a: u32, b: u32) -> f64 {
    let d = sub(mesh.points[a as usize], mesh.points[b as usize]);
    dot(d, d).sqrt()
}
