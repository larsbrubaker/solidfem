//! The public face of the crate: [`Triangulator`] takes a point set, inserts it a slice at a time
//! under a host's control, accepts further points once done, and hands out the finished
//! [`Tetrahedralization`]. Input checks and the starting tetrahedron are here; `order.rs` chooses
//! the insertion order and `core.rs` does the insertion.
//!
//! Slicing changes nothing about the output: all state lives in the [`Triangulator`], and a slice
//! boundary falls between two points, so any step size reproduces the same bits.

use crate::core::{Core, DEAD, GHOST};
use crate::error::TetraError;
use crate::order::insertion_order;
use crate::predicates::orient;

/// Beyond this magnitude products inside the exact predicates could overflow. In-sphere is degree
/// 5 in coordinate differences; `(2 * 1e55)^5` is about `3e276`, leaving room for the determinant's
/// sums and the exact-arithmetic error terms below `f64::MAX`.
const MAX_MAGNITUDE: f64 = 1e55;

/// Below this magnitude (and not zero) products inside the exact predicates could underflow. Every
/// non-zero coordinate is a multiple of `MIN_MAGNITUDE * 2^-52`, and the degree-5 products of
/// those are still multiples of the smallest subnormal, so no bit of an exact product is lost.
const MIN_MAGNITUDE: f64 = 1e-48;

/// `u32` vertex indices leave [`GHOST`] and `DEAD` unused, and the search stamps need twice the
/// insertion count to fit `u32`.
const MAX_POINTS: usize = (u32::MAX / 2 - 1) as usize;

/// A neighbour index meaning this face is on the convex hull.
pub const HULL: u32 = u32::MAX;

/// The Delaunay tetrahedralization of a point set, covering its convex hull exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct Tetrahedralization {
    /// Vertices, exactly the input points with coordinates untouched, in input order.
    pub points: Vec<[f64; 3]>,
    /// Four point indices per tetrahedron, with `(b-a) x (c-a) * (d-a) > 0` exactly.
    pub tets: Vec<[u32; 4]>,
    /// The tetrahedron across the face opposite each vertex, or [`HULL`].
    pub neighbors: Vec<[u32; 4]>,
}

/// How a call to [`Triangulator::step`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    /// More points remain; call `step` again.
    Running,
    /// Every point is inserted; [`Triangulator::result`] is available.
    Done,
    /// The cancel check returned true. Nothing is lost: another `step` continues.
    Cancelled,
}

/// The resumable triangulator of one point set.
pub struct Triangulator {
    core: Core,
    order: Vec<u32>,
    next: usize,
}

impl Triangulator {
    /// Validates `points`, finds the first tetrahedron and prepares the insertion order.
    ///
    /// This is the one part that does not slice: it sorts all points, about 4 percent of a whole
    /// run, 0.25 s for a million points in a native release build.
    pub fn new(points: Vec<[f64; 3]>) -> Result<Self, TetraError> {
        if points.len() < 4 {
            return Err(TetraError::TooFewPoints {
                count: points.len(),
            });
        }
        if points.len() > MAX_POINTS {
            return Err(TetraError::TooManyPoints {
                count: points.len(),
            });
        }
        for (index, point) in points.iter().enumerate() {
            check_point(point, index)?;
        }
        let mut order = insertion_order(&points);
        let simplex = first_simplex(&points, &order)?;
        order.retain(|index| !simplex.contains(index));
        Ok(Self {
            core: Core::new(points, simplex)?,
            order,
            next: 0,
        })
    }

    /// Inserts up to `max_points` points, checking `cancel` before each.
    ///
    /// # Errors
    /// [`TetraError::DuplicatePoint`] when a point repeats an earlier one (the same error comes
    /// back on any later call); the invariant errors of [`TetraError`] on a defect.
    pub fn step(
        &mut self,
        max_points: usize,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<StepStatus, TetraError> {
        let mut done_this_call = 0;
        while self.next < self.order.len() {
            if done_this_call == max_points {
                return Ok(StepStatus::Running);
            }
            if cancel() {
                return Ok(StepStatus::Cancelled);
            }
            let point = self.order[self.next];
            self.core.insert(point)?;
            self.next += 1;
            done_this_call += 1;
        }
        Ok(StepStatus::Done)
    }

    /// Whether every point has been inserted.
    pub fn is_done(&self) -> bool {
        self.next == self.order.len()
    }

    /// The number of points inserted so far.
    pub fn inserted(&self) -> usize {
        self.next + 4
    }

    /// The total number of points, including any added by [`Self::insert_point`].
    pub fn total(&self) -> usize {
        self.order.len() + 4
    }

    /// Adds a point to the finished triangulation and returns its index in the result. The point
    /// can be inside or outside the current hull. The tie rule ranks points by index, so the
    /// tetrahedra are the same as if the point had been in the initial set at this index.
    ///
    /// # Errors
    /// [`TetraError::StillInserting`] until [`Self::step`] reports `Done`; a coordinate error for
    /// an invalid point; [`TetraError::DuplicatePoint`] when it repeats a point. Nothing changes
    /// on an error.
    pub fn insert_point(&mut self, point: [f64; 3]) -> Result<u32, TetraError> {
        if !self.is_done() {
            return Err(TetraError::StillInserting {
                inserted: self.inserted(),
                total: self.total(),
            });
        }
        if self.core.points.len() >= MAX_POINTS {
            return Err(TetraError::TooManyPoints {
                count: self.core.points.len() + 1,
            });
        }
        check_point(&point, self.core.points.len())?;
        let index = self.core.push_point(point);
        if let Err(error) = self.core.insert(index) {
            self.core.pop_point();
            return Err(error);
        }
        self.order.push(index);
        self.next += 1;
        Ok(index)
    }

    /// The finished tetrahedralization: real tetrahedra only, in a deterministic order, with
    /// faces on the hull marked [`HULL`].
    pub fn result(&self) -> Result<Tetrahedralization, TetraError> {
        if !self.is_done() {
            return Err(TetraError::NotFinished {
                inserted: self.inserted(),
                total: self.total(),
            });
        }
        let core = &self.core;
        let mut output_index = vec![HULL; core.tets.len()];
        let mut count = 0u32;
        for (slot, tet) in core.tets.iter().enumerate() {
            if tet.iter().all(|&v| v != GHOST && v != DEAD) {
                output_index[slot] = count;
                count += 1;
            }
        }
        let mut tets = Vec::with_capacity(count as usize);
        let mut neighbors = Vec::with_capacity(count as usize);
        for (slot, tet) in core.tets.iter().enumerate() {
            if output_index[slot] == HULL {
                continue;
            }
            let adjacent = core.neighbors[slot];
            // Internal tetrahedra have positive `orient` (`d` below `a b c`); callers get the
            // right-handed order, so swap the first two vertices and their faces.
            tets.push([tet[1], tet[0], tet[2], tet[3]]);
            neighbors.push([
                output_index[adjacent[1] as usize],
                output_index[adjacent[0] as usize],
                output_index[adjacent[2] as usize],
                output_index[adjacent[3] as usize],
            ]);
        }
        Ok(Tetrahedralization {
            points: core.points.clone(),
            tets,
            neighbors,
        })
    }

    /// The bytes of memory the triangulator holds per point right now, counting the capacity of
    /// the tetrahedron and neighbour arrays and the point array.
    pub fn memory_bytes(&self) -> usize {
        let core = &self.core;
        core.tets.capacity() * std::mem::size_of::<[u32; 4]>()
            + core.neighbors.capacity() * std::mem::size_of::<[u32; 4]>()
            + core.tets.capacity() * std::mem::size_of::<u32>()
            + core.points.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.order.capacity() * std::mem::size_of::<u32>()
    }
}

/// Refuses a point the exact predicates cannot decide correctly.
fn check_point(point: &[f64; 3], index: usize) -> Result<(), TetraError> {
    for &value in point {
        if !value.is_finite() {
            return Err(TetraError::NonFiniteCoordinate { index, value });
        }
        let magnitude = value.abs();
        if magnitude > MAX_MAGNITUDE || (magnitude != 0.0 && magnitude < MIN_MAGNITUDE) {
            return Err(TetraError::CoordinateOutOfRange { index, value });
        }
    }
    Ok(())
}

/// The first four points in insertion order that span a tetrahedron. Exact tests only, so a
/// nearly flat input is either found out by its fourth point or correctly reported as flat.
fn first_simplex(points: &[[f64; 3]], order: &[u32]) -> Result<[u32; 4], TetraError> {
    let at = |index: u32| &points[index as usize];
    let first = order[0];
    let second = *order
        .iter()
        .find(|&&candidate| at(candidate) != at(first))
        .ok_or(TetraError::AllPointsCoincident)?;
    let third = *order
        .iter()
        .find(|&&candidate| !collinear(at(first), at(second), at(candidate)))
        .ok_or(TetraError::AllPointsCollinear)?;
    let fourth = *order
        .iter()
        .find(|&&candidate| orient(at(first), at(second), at(third), at(candidate)) != 0.0)
        .ok_or(TetraError::AllPointsCoplanar)?;
    Ok([first, second, third, fourth])
}

/// Whether three points are on one line: every 2D projection of their triangle has zero area.
fn collinear(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    use robust::{orient2d, Coord};
    let area = |i: usize, j: usize| {
        let project = |p: &[f64; 3]| Coord { x: p[i], y: p[j] };
        orient2d(project(a), project(b), project(c))
    };
    area(0, 1) == 0.0 && area(1, 2) == 0.0 && area(0, 2) == 0.0
}
