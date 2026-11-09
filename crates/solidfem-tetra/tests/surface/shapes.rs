// Closed surfaces with outward winding, built from simple geometry. `Surface` welds equal
// coordinates (by their bits) so shapes are described face by face without tracking shared
// vertices; a quad is counterclockwise seen from outside the solid and split on one diagonal.

use std::collections::HashMap;
use std::f64::consts::TAU;

#[derive(Default, Clone)]
pub struct Surface {
    pub points: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub index: HashMap<[u64; 3], u32>,
}

impl Surface {
    pub fn vertex(&mut self, point: [f64; 3]) -> u32 {
        // Adding zero maps -0.0 to 0.0, so the weld treats them as one point as the mesher does.
        let key = [
            (point[0] + 0.0).to_bits(),
            (point[1] + 0.0).to_bits(),
            (point[2] + 0.0).to_bits(),
        ];
        let points = &mut self.points;
        *self.index.entry(key).or_insert_with(|| {
            points.push(point);
            (points.len() - 1) as u32
        })
    }

    pub fn triangle(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
        let t = [self.vertex(a), self.vertex(b), self.vertex(c)];
        self.triangles.push(t);
    }

    pub fn quad(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) {
        self.triangle(a, b, c);
        self.triangle(a, c, d);
    }

    /// The other shell added with reversed winding, as a void inside this solid needs.
    pub fn add_reversed(&mut self, other: &Surface) {
        for t in &other.triangles {
            let [a, b, c] = t.map(|v| other.points[v as usize]);
            self.triangle(a, c, b);
        }
    }

    pub fn add(&mut self, other: &Surface) {
        for t in &other.triangles {
            let [a, b, c] = t.map(|v| other.points[v as usize]);
            self.triangle(a, b, c);
        }
    }
}

/// A rotation by exact (rounded) cosines: previously exactly coplanar faces become near-coplanar.
pub fn rotated(surface: &Surface) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let (sa, ca) = (0.71_f64.sin(), 0.71_f64.cos());
    let (sb, cb) = (1.13_f64.sin(), 1.13_f64.cos());
    let (sc, cc) = (0.37_f64.sin(), 0.37_f64.cos());
    let matrix = [
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ];
    let points = surface
        .points
        .iter()
        .map(|p| {
            let mut out = [0.0; 3];
            for (row, value) in matrix.iter().zip(out.iter_mut()) {
                *value = row[0] * p[0] + row[1] * p[1] + row[2] * p[2];
            }
            out
        })
        .collect();
    (points, surface.triangles.clone())
}

/// A box with outward faces, each split into `n x n` quads.
pub fn subdivided_box(low: [f64; 3], high: [f64; 3], n: usize) -> Surface {
    let mut s = Surface::default();
    // Both ends exactly, so a corner reached from any face welds: `low + (high-low)*n/n` need not
    // round back to `high`.
    let divide = |axis: usize, i: usize| {
        if i == 0 {
            low[axis]
        } else if i == n {
            high[axis]
        } else {
            low[axis] + (high[axis] - low[axis]) * i as f64 / n as f64
        }
    };
    for i in 0..n {
        for j in 0..n {
            let (u0, u1) = (divide(0, i), divide(0, i + 1));
            let (v0, v1) = (divide(1, j), divide(1, j + 1));
            s.quad(
                [u0, v0, low[2]],
                [u0, v1, low[2]],
                [u1, v1, low[2]],
                [u1, v0, low[2]],
            );
            s.quad(
                [u0, v0, high[2]],
                [u1, v0, high[2]],
                [u1, v1, high[2]],
                [u0, v1, high[2]],
            );
            let (w0, w1) = (divide(2, j), divide(2, j + 1));
            s.quad(
                [u0, low[1], w0],
                [u1, low[1], w0],
                [u1, low[1], w1],
                [u0, low[1], w1],
            );
            s.quad(
                [u0, high[1], w0],
                [u0, high[1], w1],
                [u1, high[1], w1],
                [u1, high[1], w0],
            );
            let (v0, v1) = (divide(1, i), divide(1, i + 1));
            s.quad(
                [low[0], v0, w0],
                [low[0], v0, w1],
                [low[0], v1, w1],
                [low[0], v1, w0],
            );
            s.quad(
                [high[0], v0, w0],
                [high[0], v1, w0],
                [high[0], v1, w1],
                [high[0], v0, w1],
            );
        }
    }
    s
}

/// A box with a concentric box-shaped void of wall thickness `wall`, the void wound to keep the
/// solid on the inner side of its triangles.
pub fn hollow_box(side: f64, wall: f64, outer_n: usize, inner_n: usize) -> Surface {
    let mut s = subdivided_box([0.0; 3], [side; 3], outer_n);
    let inner = subdivided_box([wall; 3], [side - wall; 3], inner_n);
    s.add_reversed(&inner);
    s
}

/// A `length x width x thick` plate built from `cols x rows` tiles of `k` boundary segments per
/// side. A tile is solid, or holds a square or round hole whose rim has `4k` points joined by a
/// strip to the outline of the tile. Every lattice coordinate comes from integer lattice indices,
/// so the boundary points of neighbouring tiles weld exactly.
pub fn plate_with_holes(
    length: f64,
    width: f64,
    thick: f64,
    cols: usize,
    rows: usize,
    k: usize,
    holes: &[(usize, usize, bool)],
) -> Surface {
    let mut s = Surface::default();
    let lattice = |ix: usize, iy: usize| {
        [
            length * ix as f64 / (cols * k) as f64,
            width * iy as f64 / (rows * k) as f64,
        ]
    };
    for col in 0..cols {
        for row in 0..rows {
            match holes.iter().find(|h| h.0 == col && h.1 == row) {
                None => {
                    for i in 0..k {
                        for j in 0..k {
                            let (ix, iy) = (col * k + i, row * k + j);
                            let at = |ax: usize, ay: usize, z: f64| {
                                let p = lattice(ax, ay);
                                [p[0], p[1], z]
                            };
                            s.quad(
                                at(ix, iy, thick),
                                at(ix + 1, iy, thick),
                                at(ix + 1, iy + 1, thick),
                                at(ix, iy + 1, thick),
                            );
                            s.quad(
                                at(ix, iy, 0.0),
                                at(ix, iy + 1, 0.0),
                                at(ix + 1, iy + 1, 0.0),
                                at(ix + 1, iy, 0.0),
                            );
                        }
                    }
                }
                Some(&(_, _, round)) => {
                    let outline: Vec<[f64; 2]> = lattice_outline(col * k, row * k, k, k)
                        .into_iter()
                        .map(|(ix, iy)| lattice(ix, iy))
                        .collect();
                    let low = lattice(col * k, row * k);
                    let high = lattice((col + 1) * k, (row + 1) * k);
                    let centre = [(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5];
                    let radius = (high[0] - low[0]).min(high[1] - low[1]) * 0.3;
                    let count = outline.len();
                    let rim: Vec<[f64; 2]> = (0..count)
                        .map(|i| {
                            if round {
                                let angle = TAU * i as f64 / count as f64 + TAU * 0.625;
                                [
                                    centre[0] + radius * angle.cos(),
                                    centre[1] + radius * angle.sin(),
                                ]
                            } else {
                                let o = outline[i];
                                [
                                    centre[0] + (o[0] - centre[0]) * 0.45,
                                    centre[1] + (o[1] - centre[1]) * 0.45,
                                ]
                            }
                        })
                        .collect();
                    for i in 0..count {
                        let n = (i + 1) % count;
                        let (a, b, c, d) = (outline[i], outline[n], rim[n], rim[i]);
                        s.quad(
                            [a[0], a[1], thick],
                            [b[0], b[1], thick],
                            [c[0], c[1], thick],
                            [d[0], d[1], thick],
                        );
                        s.quad(
                            [b[0], b[1], 0.0],
                            [a[0], a[1], 0.0],
                            [d[0], d[1], 0.0],
                            [c[0], c[1], 0.0],
                        );
                        s.quad(
                            [d[0], d[1], thick],
                            [c[0], c[1], thick],
                            [c[0], c[1], 0.0],
                            [d[0], d[1], 0.0],
                        );
                    }
                }
            }
        }
    }
    let perimeter: Vec<[f64; 2]> = lattice_outline(0, 0, cols * k, rows * k)
        .into_iter()
        .map(|(ix, iy)| lattice(ix, iy))
        .collect();
    for i in 0..perimeter.len() {
        let (a, b) = (perimeter[i], perimeter[(i + 1) % perimeter.len()]);
        s.quad(
            [a[0], a[1], 0.0],
            [b[0], b[1], 0.0],
            [b[0], b[1], thick],
            [a[0], a[1], thick],
        );
    }
    s
}

/// The lattice outline of a rectangle, counterclockwise from its low corner.
fn lattice_outline(x0: usize, y0: usize, nx: usize, ny: usize) -> Vec<(usize, usize)> {
    let mut outline = Vec::new();
    for i in 0..nx {
        outline.push((x0 + i, y0));
    }
    for j in 0..ny {
        outline.push((x0 + nx, y0 + j));
    }
    for i in 0..nx {
        outline.push((x0 + nx - i, y0 + ny));
    }
    for j in 0..ny {
        outline.push((x0, y0 + ny - j));
    }
    outline
}

/// A prism over a regular `n`-gon with both flat caps triangulated as a fan from one rim vertex
/// (long needles, nothing like the Delaunay triangulation of the plane). `zigzag` strips the caps
/// instead, between rim points in opposite pairs.
pub fn prism(n: usize, radius: f64, height: f64, zigzag: bool) -> Surface {
    let mut s = Surface::default();
    let rim: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let angle = TAU * i as f64 / n as f64;
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect();
    for i in 0..n {
        let (a, b) = (rim[i], rim[(i + 1) % n]);
        s.quad(
            [a[0], a[1], 0.0],
            [b[0], b[1], 0.0],
            [b[0], b[1], height],
            [a[0], a[1], height],
        );
    }
    let at = |i: usize, z: f64| [rim[i % n][0], rim[i % n][1], z];
    if zigzag {
        let (mut low, mut high) = (0usize, n - 1);
        while high > low + 1 {
            s.triangle(at(low, height), at(low + 1, height), at(high, height));
            s.triangle(at(low, 0.0), at(high, 0.0), at(low + 1, 0.0));
            low += 1;
            if high > low + 1 {
                s.triangle(at(low, height), at(high - 1, height), at(high, height));
                s.triangle(at(low, 0.0), at(high, 0.0), at(high - 1, 0.0));
                high -= 1;
            }
        }
    } else {
        for i in 1..n - 1 {
            s.triangle(at(0, height), at(i, height), at(i + 1, height));
            s.triangle(at(0, 0.0), at(i + 1, 0.0), at(i, 0.0));
        }
    }
    s
}

/// A `length x 20 x 20` box, the long faces split into `n` strips of quads across their full
/// width, so at `n = 200` over a length of 40 every long-face triangle is a 100:1 needle.
pub fn needle_box(length: f64, n: usize) -> Surface {
    let mut s = Surface::default();
    let x = |i: usize| length * i as f64 / n as f64;
    let (y0, y1, z0, z1) = (0.0, 20.0, 0.0, 20.0);
    for i in 0..n {
        let (a, b) = (x(i), x(i + 1));
        s.quad([a, y0, z1], [b, y0, z1], [b, y1, z1], [a, y1, z1]);
        s.quad([a, y1, z0], [b, y1, z0], [b, y0, z0], [a, y0, z0]);
        s.quad([a, y0, z0], [b, y0, z0], [b, y0, z1], [a, y0, z1]);
        s.quad([a, y1, z1], [b, y1, z1], [b, y1, z0], [a, y1, z0]);
    }
    s.quad([0.0, y0, z0], [0.0, y0, z1], [0.0, y1, z1], [0.0, y1, z0]);
    s.quad(
        [length, y0, z0],
        [length, y1, z0],
        [length, y1, z1],
        [length, y0, z1],
    );
    s
}

/// The Schoenhardt polyhedron: a triangular prism twisted so each side quad folds inward along
/// its reflex diagonal. Six vertices, eight faces, and no tetrahedralization without a new point.
pub fn schoenhardt() -> Surface {
    let mut s = Surface::default();
    let at = |angle: f64, z: f64| [angle.cos(), angle.sin(), z];
    let twist = 0.6;
    let bottom = [0.0, TAU / 3.0, 2.0 * TAU / 3.0].map(|a| at(a, 0.0));
    let top = [0.0, TAU / 3.0, 2.0 * TAU / 3.0].map(|a| at(a + twist, 1.0));
    s.triangle(top[0], top[1], top[2]);
    s.triangle(bottom[0], bottom[2], bottom[1]);
    for i in 0..3 {
        let j = (i + 1) % 3;
        // The diagonal from `bottom[i]` to `top[j]` spans the longer way round; it is the one
        // that folds inward, giving the reflex edges that make this polyhedron impossible.
        s.triangle(bottom[i], bottom[j], top[j]);
        s.triangle(bottom[i], top[j], top[i]);
    }
    s
}
