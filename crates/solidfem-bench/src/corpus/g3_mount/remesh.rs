//! One isotropic remesh of a soup with the existing remesher: sharp edges held (may split, never
//! flip or collapse, and their end vertices do not move), open boundaries pinned, every vertex
//! projected back onto the input surface after each pass. A closed input that the remesher's
//! `DMesh3` cannot hold (a non-manifold edge) is reported, not silently patched.

use std::rc::Rc;

use super::dmesh3::DMesh3;
use super::dmesh_aabb_tree3::DMeshAABBTree3;
use super::mesh_constraints::{
    edge_refine_flags, EdgeConstraint, MeshConstraints, ProjectionTarget, VertexConstraint,
};
use super::mesh_projection_target::MeshProjectionTarget;
use super::remesher::TargetProjectionMode;
use super::remesher_pro::RemesherPro;
use crate::corpus::soup::TriSoup;

/// Remesh passes: enough to reach the target edge length from a coarser or finer input, and the
/// remesher's own convergence tests stop at about this many.
const ITERATIONS: i32 = 12;

/// Why a remesh did not produce a surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemeshRefusal {
    /// The input has faces `DMesh3` refuses (a non-manifold edge), so the remesher never sees
    /// the whole part.
    NonManifoldInput { refused_faces: usize },
    /// The remesher stopped with a broken-topology error.
    BrokenTopology(&'static str),
}

/// The remesher's output and how many input sharp edges it was told to keep.
#[derive(Clone, Debug)]
pub struct RemeshOutput {
    pub soup: TriSoup,
    pub constrained_sharp_edges: usize,
}

/// Remeshes `soup` towards `target_edge_length`, holding `sharp_edges`.
pub fn remesh(
    soup: &TriSoup,
    sharp_edges: &[(u32, u32)],
    target_edge_length: f64,
) -> Result<RemeshOutput, RemeshRefusal> {
    let (mut working, refused) = to_dmesh(soup);
    if refused > 0 {
        return Err(RemeshRefusal::NonManifoldInput {
            refused_faces: refused,
        });
    }
    let target = projection_target(working.clone());
    let constrained = {
        let mut remesher = RemesherPro::new(&mut working);
        let constrained = constrain_sharp_edges(&mut remesher, sharp_edges);
        remesher.fix_all_boundary_edges();
        remesher.remesher.set_target_edge_length(target_edge_length);
        // Half-way smoothing: the default 0.1 moves vertices too little to even out a CAD mesh's
        // long sliver fans within this many passes.
        remesher.remesher.smooth_speed_t = 0.5;
        remesher.remesher.prevent_normal_flips = true;
        remesher.remesher.projection_mode = TargetProjectionMode::AfterRefinement;
        remesher.remesher.set_projection_target(Some(target));
        remesher.remesher.precompute();
        remesher
            .fastest_remesh(ITERATIONS, true)
            .map_err(|e| RemeshRefusal::BrokenTopology(e.0))?;
        constrained
    };
    Ok(RemeshOutput {
        soup: from_dmesh(&working),
        constrained_sharp_edges: constrained,
    })
}

/// The target outlives the remesher and must be `'static` behind the `Rc`, so its tree and the
/// source copy are leaked. A remesh is one short-lived bench step, so the leak is bounded by the
/// size of the input.
fn projection_target(source: DMesh3) -> Rc<dyn ProjectionTarget> {
    let source: &'static DMesh3 = Box::leak(Box::new(source));
    let tree: &'static DMeshAABBTree3<'static> = Box::leak(Box::new(DMeshAABBTree3::new(source)));
    Rc::new(MeshProjectionTarget::new(tree))
}

/// Marks each sharp edge may-split-only and pins its ends. Returns how many edges were found in
/// the mesh (each input sharp edge is, unless the mesh dropped one of its faces).
fn constrain_sharp_edges(remesher: &mut RemesherPro<'_>, sharp_edges: &[(u32, u32)]) -> usize {
    let mut constraints = MeshConstraints::default();
    let mut found = 0;
    for &(a, b) in sharp_edges {
        let eid = remesher.remesher.base.mesh.find_edge(a as i32, b as i32);
        if eid < 0 {
            continue;
        }
        found += 1;
        constraints.set_or_update_edge_constraint(
            eid,
            EdgeConstraint::new(edge_refine_flags::NO_FLIP | edge_refine_flags::NO_COLLAPSE),
        );
        constraints.set_or_update_vertex_constraint(a as i32, VertexConstraint::pinned());
        constraints.set_or_update_vertex_constraint(b as i32, VertexConstraint::pinned());
    }
    remesher.remesher.base.constraints = Some(constraints);
    found
}

/// Copies a soup into a `DMesh3`, and how many faces it refused.
fn to_dmesh(soup: &TriSoup) -> (DMesh3, usize) {
    let mut mesh = DMesh3::default();
    for v in &soup.vertices {
        mesh.append_vertex(*v);
    }
    let mut refused = 0;
    for t in &soup.triangles {
        if mesh.append_triangle(t[0] as i32, t[1] as i32, t[2] as i32) < 0 {
            refused += 1;
        }
    }
    (mesh, refused)
}

/// The live vertices and triangles of a `DMesh3` as a compacted soup.
fn from_dmesh(mesh: &DMesh3) -> TriSoup {
    let mut remap = vec![0_u32; mesh.max_vertex_id() as usize];
    let mut vertices = Vec::with_capacity(mesh.vertex_count());
    for (new, vid) in mesh.vertex_indices().enumerate() {
        remap[vid as usize] = new as u32;
        vertices.push(mesh.vertex(vid));
    }
    let triangles = mesh
        .triangle_indices()
        .map(|t| {
            let [a, b, c] = mesh.get_triangle(t);
            [remap[a as usize], remap[b as usize], remap[c as usize]]
        })
        .collect();
    TriSoup {
        vertices,
        triangles,
    }
}
