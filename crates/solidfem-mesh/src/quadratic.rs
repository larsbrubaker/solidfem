//! The 10-node quadratic tetrahedral mesh: a [`TetMesh`](crate::TetMesh) with one new node per
//! shared edge. Edges are numbered from a sort of their sorted node pair, so numbering depends
//! only on the input mesh, never on map iteration order; `boundary_faces` sorts face keys for
//! the same reason.

use crate::boundary::{face_nodes_from_corners, sorted_key, BoundaryFace, FACE_NODES};
use crate::tet::TetMesh;

/// The corner pair of each midpoint node 4..=9, in local node order.
const EDGE_CORNERS: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];

/// A validated mesh of 10-node tetrahedra with straight edges.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticTetMesh {
    nodes: Vec<[f64; 3]>,
    tets: Vec<[u32; 10]>,
    corner_tets: Vec<[u32; 4]>,
    vertex_count: usize,
}

impl QuadraticTetMesh {
    /// Upgrades a linear mesh, adding a midpoint node per edge. Corner nodes keep their indices;
    /// midpoints follow, numbered in the ascending order of their sorted corner pair.
    pub fn from_linear(mesh: &TetMesh) -> Self {
        let vertex_count = mesh.positions().len();
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(mesh.tets().len() * 6);
        for tet in mesh.tets() {
            for (a, b) in EDGE_CORNERS {
                let (x, y) = (tet[a], tet[b]);
                edges.push((x.min(y), x.max(y)));
            }
        }
        edges.sort_unstable();
        edges.dedup();

        let mut nodes = mesh.positions().to_vec();
        nodes.reserve(edges.len());
        for &(a, b) in &edges {
            let (p, q) = (nodes[a as usize], nodes[b as usize]);
            nodes.push([
                0.5 * (p[0] + q[0]),
                0.5 * (p[1] + q[1]),
                0.5 * (p[2] + q[2]),
            ]);
        }
        let midpoint = |x: u32, y: u32| -> u32 {
            let key = (x.min(y), x.max(y));
            // Every edge key was pushed above, so the search always hits; the position fits in
            // u32 because the mesh's node indices are u32.
            let at = edges.binary_search(&key).unwrap_or_else(|i| i);
            (vertex_count + at) as u32
        };
        let tets = mesh
            .tets()
            .iter()
            .map(|t| {
                let mut quad = [0u32; 10];
                quad[..4].copy_from_slice(t);
                for (slot, (a, b)) in EDGE_CORNERS.iter().enumerate() {
                    quad[4 + slot] = midpoint(t[*a], t[*b]);
                }
                quad
            })
            .collect();
        QuadraticTetMesh {
            nodes,
            tets,
            corner_tets: mesh.tets().to_vec(),
            vertex_count,
        }
    }

    /// Node positions: the corners first, then edge midpoints.
    pub fn nodes(&self) -> &[[f64; 3]] {
        &self.nodes
    }

    /// Elements as ten node indices each, in the crate's node order.
    pub fn tets(&self) -> &[[u32; 10]] {
        &self.tets
    }

    /// How many leading nodes are element corners.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// The faces belonging to exactly one tetrahedron, ascending by their sorted corner nodes.
    pub fn boundary_faces(&self) -> Vec<BoundaryFace> {
        let mut keyed: Vec<([u32; 3], usize, usize)> = Vec::with_capacity(self.tets.len() * 4);
        for (tet, corners) in self.corner_tets.iter().enumerate() {
            for face in 0..4 {
                keyed.push((
                    sorted_key(face_nodes_from_corners(corners, face)),
                    tet,
                    face,
                ));
            }
        }
        keyed.sort_unstable();
        let mut boundary = Vec::new();
        let mut at = 0;
        while at < keyed.len() {
            let mut end = at + 1;
            while end < keyed.len() && keyed[end].0 == keyed[at].0 {
                end += 1;
            }
            if end == at + 1 {
                let (_, tet, face) = keyed[at];
                let local = FACE_NODES[face];
                let mut nodes = [0u32; 6];
                for (slot, node) in nodes.iter_mut().enumerate() {
                    *node = self.tets[tet][local[slot]];
                }
                boundary.push(BoundaryFace {
                    tet: tet as u32,
                    face: face as u8,
                    nodes,
                });
            }
            at = end;
        }
        boundary
    }

    /// The tetrahedra grouped into bodies joined across shared faces: the sets that move as one
    /// rigid piece when a part is held. Each entry is that body's tet indices, ascending, and the
    /// bodies are in order of their first tet.
    pub fn face_connected_bodies(&self) -> Vec<Vec<u32>> {
        let mut parent: Vec<usize> = (0..self.tets.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        let mut keyed: Vec<([u32; 3], usize)> = Vec::with_capacity(self.tets.len() * 4);
        for (tet, corners) in self.corner_tets.iter().enumerate() {
            for face in 0..4 {
                keyed.push((sorted_key(face_nodes_from_corners(corners, face)), tet));
            }
        }
        keyed.sort_unstable();
        for pair in keyed.windows(2) {
            if pair[0].0 == pair[1].0 {
                let a = find(&mut parent, pair[0].1);
                let b = find(&mut parent, pair[1].1);
                if a != b {
                    let (keep, drop) = if a < b { (a, b) } else { (b, a) };
                    parent[drop] = keep;
                }
            }
        }
        let mut groups: Vec<Vec<u32>> = Vec::new();
        let mut group_of_root = vec![usize::MAX; self.tets.len()];
        for tet in 0..self.tets.len() {
            let root = find(&mut parent, tet);
            if group_of_root[root] == usize::MAX {
                group_of_root[root] = groups.len();
                groups.push(Vec::new());
            }
            groups[group_of_root[root]].push(tet as u32);
        }
        groups
    }
}
