//! The triangle soup every corpus reading works on: `f64` positions and indexed triangles, built
//! from a solidmesh `Mesh` or from the remesher's output, with the small vector helpers the
//! readings share, the point-to-triangle distance among them. A soup keeps every input triangle,
//! including the non-manifold and degenerate ones the remesher's own mesh refuses, because those
//! are exactly what is being measured.

use solidmesh_math::Vector3Float;
use solidmesh_mesh::{AnalysisSurface, Mesh};

/// A point or vector.
pub type P3 = [f64; 3];

/// `a - b`.
pub fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `a * s`.
pub fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The dot product.
pub fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The cross product.
pub fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The Euclidean length.
pub fn length(a: P3) -> f64 {
    dot(a, a).sqrt()
}

/// The unit normal of a triangle, or `None` when it has no area to give a direction.
pub fn face_normal(corners: &[P3; 3]) -> Option<P3> {
    let n = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
    let len = length(n);
    (len > 0.0).then(|| scale(n, 1.0 / len))
}

/// Indexed triangles over `f64` positions.
#[derive(Clone, Debug, Default)]
pub struct TriSoup {
    pub vertices: Vec<P3>,
    pub triangles: Vec<[u32; 3]>,
}

impl TriSoup {
    /// The positions and faces of a solidmesh mesh, widened to `f64`.
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let vertices = mesh
            .vertices
            .iter()
            .map(|v| [v.x as f64, v.y as f64, v.z as f64])
            .collect();
        let triangles = mesh
            .faces
            .iter()
            .map(|f| [f.v0 as u32, f.v1 as u32, f.v2 as u32])
            .collect();
        Self {
            vertices,
            triangles,
        }
    }

    /// The surface `remesh_for_analysis` returned, in its exact `f64` coordinates.
    pub fn from_analysis_surface(surface: &AnalysisSurface) -> Self {
        let triangles = surface
            .faces
            .iter()
            // The remesher's indices point into its own vertex list, so none is negative.
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect();
        Self {
            vertices: surface.vertices.clone(),
            triangles,
        }
    }

    /// This soup as a solidmesh `Mesh`, every triangle kept in order, for the remesher. A soup
    /// read from a `Mesh` goes back to the same coordinates.
    pub fn to_mesh(&self) -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices = self
            .vertices
            .iter()
            .map(|v| Vector3Float::new(v[0], v[1], v[2]))
            .collect();
        for t in &self.triangles {
            // A corpus soup holds far fewer than `i32::MAX` vertices.
            mesh.faces
                .add(t[0] as i32, t[1] as i32, t[2] as i32, &mesh.vertices);
        }
        mesh
    }

    /// The three corner positions of triangle `t`.
    pub fn corners(&self, t: usize) -> [P3; 3] {
        let [a, b, c] = self.triangles[t];
        [
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
        ]
    }

    /// The length of the bounding-box diagonal.
    pub fn bbox_diagonal(&self) -> f64 {
        let Some(first) = self.vertices.first() else {
            return 0.0;
        };
        let (mut lo, mut hi) = (*first, *first);
        for v in &self.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        length(sub(hi, lo))
    }

    /// The signed volume, positive for an outward-wound closed surface.
    pub fn signed_volume(&self) -> f64 {
        (0..self.triangles.len())
            .map(|t| {
                let [a, b, c] = self.corners(t);
                dot(a, cross(b, c))
            })
            .sum::<f64>()
            / 6.0
    }

    /// The total triangle area.
    pub fn surface_area(&self) -> f64 {
        (0..self.triangles.len())
            .map(|t| {
                let [a, b, c] = self.corners(t);
                length(cross(sub(b, a), sub(c, a))) * 0.5
            })
            .sum()
    }
}

/// The squared distance from `p` to the segment `a..b`.
pub fn point_segment_distance_squared(p: P3, a: P3, b: P3) -> f64 {
    let ab = sub(b, a);
    let span = dot(ab, ab);
    let t = if span > 0.0 {
        (dot(sub(p, a), ab) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let off = sub(p, [0, 1, 2].map(|k| a[k] + ab[k] * t));
    dot(off, off)
}

/// The squared distance from `p` to the triangle `c`: to its plane where `p` projects inside it,
/// otherwise to the nearest of its edges. A triangle with no area is its edges.
pub fn point_triangle_distance_squared(p: P3, c: &[P3; 3]) -> f64 {
    let along_edges = (0..3)
        .map(|i| point_segment_distance_squared(p, c[i], c[(i + 1) % 3]))
        .fold(f64::INFINITY, f64::min);
    let Some(n) = face_normal(c) else {
        return along_edges;
    };
    let height = dot(sub(p, c[0]), n);
    let q = [0, 1, 2].map(|k| p[k] - n[k] * height);
    let inside = (0..3).all(|i| dot(cross(sub(c[(i + 1) % 3], c[i]), sub(q, c[i])), n) >= 0.0);
    if inside {
        height * height
    } else {
        along_edges
    }
}
