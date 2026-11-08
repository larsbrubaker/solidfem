//! The render arrays: the surface as triangles and the per-vertex values the page colours and
//! displaces it with. Each 6-node boundary face becomes four straight triangles over its corners
//! and edge midpoints (`engine.rs` says why). The peak stress skips nodes near a clamp, because a
//! clamp's corners are a stress singularity whose nodal value does not converge.

use solidfem::mesh::QuadraticTetMesh;

/// One vertex per surface node, and the triangles over them.
#[derive(Debug, Clone)]
pub struct Surface {
    /// Mesh node index of each surface vertex, ascending.
    vertex_nodes: Vec<u32>,
    /// Triangle vertex indices into `vertex_nodes`, three per triangle.
    indices: Vec<u32>,
}

/// What the page draws for the current estimate, plus the summary numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewData {
    /// Three per surface vertex: the part's undeformed position in mm.
    pub positions: Vec<f32>,
    /// Three per triangle, indexing the vertices.
    pub indices: Vec<u32>,
    /// One per vertex: the volume-weighted average von Mises stress in MPa at this estimate.
    pub von_mises: Vec<f32>,
    /// Three per vertex: displacement at this estimate in mm.
    pub displacement: Vec<f32>,
    pub min_stress: f64,
    pub max_stress: f64,
    pub max_displacement: f64,
    /// The largest von Mises stress outside the clamp zones.
    pub peak_stress_outside_clamps: f64,
    /// Material yield over that peak; `None` before there is any stress.
    pub safety_factor: Option<f64>,
}

impl Surface {
    /// Extracts the surface of a mesh.
    pub fn of(mesh: &QuadraticTetMesh) -> Self {
        let faces = mesh.boundary_faces();
        let mut used = vec![false; mesh.nodes().len()];
        for face in &faces {
            for node in &face.nodes {
                used[*node as usize] = true;
            }
        }
        let mut slot = vec![0u32; used.len()];
        let mut vertex_nodes = Vec::new();
        for (node, used) in used.iter().enumerate() {
            if *used {
                slot[node] = vertex_nodes.len() as u32;
                vertex_nodes.push(node as u32);
            }
        }
        let mut indices = Vec::with_capacity(faces.len() * 12);
        for face in &faces {
            let [c0, c1, c2, m01, m12, m20] = face.nodes.map(|n| slot[n as usize]);
            // Four triangles over the corners and midpoints, each wound like the face.
            indices.extend_from_slice(&[c0, m01, m20, m01, c1, m12, m20, m12, c2, m01, m12, m20]);
        }
        Surface {
            vertex_nodes,
            indices,
        }
    }

    /// The arrays at the current estimate.
    pub fn view(
        &self,
        mesh: &QuadraticTetMesh,
        displacements: &[[f64; 3]],
        nodal_von_mises: &[f64],
        clamps: &[f64],
        clamp_zone: f64,
        yield_strength: f64,
    ) -> ViewData {
        let count = self.vertex_nodes.len();
        let mut positions = Vec::with_capacity(count * 3);
        let mut displacement = Vec::with_capacity(count * 3);
        let mut von_mises = Vec::with_capacity(count);
        let mut min_stress = f64::INFINITY;
        let mut max_stress = 0.0f64;
        let mut max_displacement = 0.0f64;
        let mut peak = 0.0f64;
        for node in &self.vertex_nodes {
            let n = *node as usize;
            let p = mesh.nodes()[n];
            let u = displacements[n];
            let s = nodal_von_mises[n];
            positions.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32]);
            displacement.extend_from_slice(&[u[0] as f32, u[1] as f32, u[2] as f32]);
            von_mises.push(s as f32);
            let u_len = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            max_displacement = max_displacement.max(u_len);
            min_stress = min_stress.min(s);
            max_stress = max_stress.max(s);
            if clamps.iter().all(|x| (p[0] - x).abs() > clamp_zone) {
                peak = peak.max(s);
            }
        }
        let safety_factor = (peak > 0.0).then(|| yield_strength / peak);
        ViewData {
            positions,
            indices: self.indices.clone(),
            von_mises,
            displacement,
            min_stress: if min_stress.is_finite() {
                min_stress
            } else {
                0.0
            },
            max_stress,
            max_displacement,
            peak_stress_outside_clamps: peak,
            safety_factor,
        }
    }
}
