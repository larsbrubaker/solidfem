//! The layer stage of a [`Refiner`] run (`refine.rs` holds the run and the rules it keeps): each
//! surface vertex looks across the wall behind it and divides a thin wall into layers
//! (`layers.rs`), each large wall face proposes columns through its middle (`wall_fill.rs`), and
//! the stage ends by splitting every interior edge still joining opposite skins (`skins.rs`).
//!
//! The same measurements, a vertex's ray and a column, are handed to `thin_walls.rs`, which lists
//! the walls too thin for the size asked for there. That listing reads the size function at the
//! middle of the measured chord, strictly inside the solid where the wall's elements are, and only
//! records what it reads: a size that is not finite and positive there records nothing and is no
//! error, so the listing never changes whether a run succeeds, nor any point or tetrahedron.

use super::Refiner;
use crate::core::GHOST;
use crate::error::TetraError;
use crate::layers::min_layer_thickness;
use crate::thin_walls::chord_size;
use crate::wall_fill::{wall_seeds, WallColumn};

/// The through-edge splits in progress: the edges of this pass and how many it has cut.
pub(super) struct SplitRun {
    edges: Vec<(u32, u32)>,
    cursor: usize,
    /// Edges actually cut in this pass, so a pass that cuts none knows the rest are stuck.
    split_this_pass: usize,
}

impl SplitRun {
    /// Where the pass is: edges looked at, and edges in it.
    pub(super) fn progress(&self) -> (usize, usize) {
        (self.cursor, self.edges.len())
    }
}

impl Refiner {
    /// One unit of the layer stage: a surface vertex, a wall face, or an edge to split.
    pub(super) fn advance_layers(&mut self) -> Result<(), TetraError> {
        if self.splits.is_some() {
            return self.advance_split();
        }
        let Some(run) = self.layers.as_mut() else {
            return self.begin_sizing();
        };
        let Some(&vertex) = run.vertices.get(run.cursor) else {
            return self.advance_fill();
        };
        run.cursor += 1;
        let normal = run.normals[vertex as usize];
        let edge = run.edge_mean[vertex as usize];
        let origin = self.flipper.core.points[vertex as usize];
        // Inward: against the outward normal of the surface at the vertex.
        let direction = [-normal[0], -normal[1], -normal[2]];
        let Some(hit) = run.rays.nearest(origin, direction, vertex) else {
            return Ok(());
        };
        if !hit.from_inside {
            return Ok(());
        }
        let wanted = chord_size(&self.size, origin, direction, hit.distance);
        self.thin_walls
            .observe_vertex(vertex, origin, &hit, wanted, false);
        if hit.distance > self.options.thin_wall_ratio * edge {
            return Ok(());
        }
        self.report.thin_wall_vertices += 1;
        let column = WallColumn {
            origin,
            direction,
            distance: hit.distance,
            far_normal: hit.normal,
            across: 0.5 * edge,
            far: hit.face,
        };
        let rounding = self.report.walls_too_thin_for_rounding;
        let placed = self.place_layer_points(vertex, None, column)?;
        if self.report.walls_too_thin_for_rounding > rounding {
            self.thin_walls
                .observe_vertex(vertex, origin, &hit, wanted, true);
        }
        self.report.layer_points += placed;
        Ok(())
    }

    /// Places the layer points of one column of a wall, walking from a tetrahedron around
    /// `vertex`, on `face` when there is one. Returns how many were placed; a wall too thin to
    /// divide above rounding is counted and adds none.
    fn place_layer_points(
        &mut self,
        vertex: u32,
        face: Option<[u32; 3]>,
        column: WallColumn,
    ) -> Result<usize, TetraError> {
        let WallColumn {
            origin,
            direction,
            distance,
            far_normal,
            across,
            ..
        } = column;
        let count = self.options.layers_through_walls;
        let layer = distance / f64::from(count);
        let magnitude = self.layers.as_ref().map_or(0.0, |run| run.magnitude);
        if layer < min_layer_thickness(magnitude) {
            self.report.walls_too_thin_for_rounding += 1;
            return Ok(0);
        }
        let mut placed = 0;
        for k in 1..count {
            self.budget_check()?;
            let along = layer * f64::from(k);
            let candidate = [
                origin[0] + direction[0] * along,
                origin[1] + direction[1] * along,
                origin[2] + direction[2] * along,
            ];
            let Some(run) = self.layers.as_mut() else {
                return Ok(placed);
            };
            if !run
                .separation
                .accept(candidate, direction, across, 0.5 * layer)
            {
                self.report.layer_points_dropped += 1;
                continue;
            }
            if self.place_from_vertex(vertex, face, candidate)? {
                // Index fits u32: the count is checked against the point limit on insertion.
                let index = self.flipper.core.points.len() as u32 - 1;
                self.axes.insert(index, far_normal);
                placed += 1;
                self.added += 1;
            } else {
                self.report.layer_points_dropped += 1;
            }
        }
        Ok(placed)
    }

    /// Proposes the wall points of the next surface triangle (`wall_fill.rs`), ending the fill and
    /// starting the through-edge splits when none are left. Each column is also a measure of the
    /// wall for `thin_walls.rs`, the only one where a wall's vertices all stand on its rim.
    fn advance_fill(&mut self) -> Result<(), TetraError> {
        let Some(run) = self.layers.as_mut() else {
            return Ok(());
        };
        let Some(&face) = run.boundary.get(run.fill_cursor) else {
            self.thin_walls.group(&run.boundary);
            self.layers = None;
            self.begin_split_pass();
            return Ok(());
        };
        run.fill_cursor += 1;
        let seeds = wall_seeds(
            &run.rays,
            &self.flipper.core.points,
            face,
            self.options.thin_wall_ratio,
            self.options.layers_through_walls,
            run.magnitude,
        );
        for seed in seeds {
            let wanted = chord_size(&self.size, seed.origin, seed.direction, seed.distance);
            self.thin_walls.observe_column(face, &seed, wanted, false);
            let rounding = self.report.walls_too_thin_for_rounding;
            let placed = self.place_layer_points(face[0], Some(face), seed)?;
            if self.report.walls_too_thin_for_rounding > rounding {
                self.thin_walls.observe_column(face, &seed, wanted, true);
            }
            self.report.wall_fill_points += placed;
            self.report.layer_points += placed;
        }
        Ok(())
    }

    fn begin_split_pass(&mut self) {
        self.splits = Some(SplitRun {
            edges: self.skins.through_edges(&self.flipper),
            cursor: 0,
            split_this_pass: 0,
        });
    }

    /// Splits the next edge joining opposite skins. At the end of a pass, the edges still there are
    /// tried again while any split was made, the mesh around them having changed; an edge that is
    /// left when a whole pass splits nothing is the named error.
    fn advance_split(&mut self) -> Result<(), TetraError> {
        let Some(run) = self.splits.as_mut() else {
            return Ok(());
        };
        let Some(&edge) = run.edges.get(run.cursor) else {
            let split_this_pass = run.split_this_pass;
            let remaining = self.skins.through_edges(&self.flipper);
            let Some(&first) = remaining.first() else {
                self.splits = None;
                return self.begin_sizing();
            };
            if split_this_pass == 0 {
                self.report.through_edges_refused = remaining.len();
                return Err(TetraError::ThroughEdgeNotSplit {
                    edge: [first.0, first.1],
                    remaining: remaining.len(),
                });
            }
            self.splits = Some(SplitRun {
                edges: remaining,
                cursor: 0,
                split_this_pass: 0,
            });
            return Ok(());
        };
        run.cursor += 1;
        self.budget_check()?;
        let layers = self.options.layers_through_walls;
        let outcome = self
            .skins
            .split_through_edge(&mut self.flipper, edge, layers)?;
        for &point in &outcome.added {
            self.axes.insert(point, outcome.axis);
            self.added += 1;
        }
        if outcome.cut {
            self.report.through_edges_split += 1;
            if let Some(run) = self.splits.as_mut() {
                run.split_this_pass += 1;
            }
        }
        Ok(())
    }

    /// Inserts `candidate` starting the walk at a real tetrahedron around `vertex`, one holding the
    /// rest of `face` when one is given, so the walk does not leave a curved wall to get there.
    fn place_from_vertex(
        &mut self,
        vertex: u32,
        face: Option<[u32; 3]>,
        candidate: [f64; 3],
    ) -> Result<bool, TetraError> {
        let around = self.flipper.tets_around(vertex);
        let real = |t: &u32| !self.flipper.core.tets[*t as usize].contains(&GHOST);
        let on_face = |t: &u32| {
            face.is_some_and(|f| {
                let tet = self.flipper.core.tets[*t as usize];
                f.iter().all(|v| tet.contains(v))
            })
        };
        let Some(start) = around
            .iter()
            .copied()
            .find(|t| real(t) && on_face(t))
            .or_else(|| around.iter().copied().find(real))
        else {
            return Ok(false);
        };
        let Some(seeds) = self.flipper.locate_interior(start, candidate) else {
            return Ok(false);
        };
        self.flipper.insert_interior(&seeds, candidate, 0.0)
    }
}
