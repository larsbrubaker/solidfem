//! Where a part's walls are too thin for the elements asked for there, as places a caller can point
//! at. The layer stage (`refine/layer_stage.rs`) measures the wall at every surface vertex, by a ray
//! inward along the vertex normal, and through the middle of every large wall face, by a column
//! straight across it (`wall_fill.rs`); this file keeps the measures under
//! `size / `[`crate::RefineOptions::thin_wall_warning_ratio`] and groups them into
//! [`ThinWallSpot`]s once the stage ends. It reads the measurements and never changes the mesh: a
//! run gives the same points and tetrahedra, and succeeds or fails the same way, whatever the ratio.
//!
//! **What is recorded.** A surface vertex whose inward ray reaches the far side of the wall from
//! inside the solid, and a wall-fill column, whose thickness is under `size / ratio`, the size read
//! at the middle of the measured chord ([`chord_size`]: strictly inside the solid, where the wall's
//! elements are). Every vertex and column [`crate::RefineReport::walls_too_thin_for_rounding`]
//! counts is recorded too, the extreme case of a wall that got no layers, whatever the ratio. A
//! size that is not finite and positive there records nothing.
//!
//! **What a measure reads.** A column stands in the middle of a face and crosses the wall square to
//! it, so it reads a flat wall's thickness; that is the measure of a wall whose vertices are all on
//! its rim, like a web between two blocks or a coarsely cut plate. A vertex ray runs along the
//! vertex normal: a vertex in the middle of a flat or evenly curved wall reads the thickness, one
//! where the faces lean against each other by an angle, or on a rim where the normal is the mean of
//! two faces' (`sqrt 2` at a square rim, `sqrt 3` at a corner), reads more. A spot can therefore
//! miss a measure near its edge, never report a wall thinner than it is.
//!
//! **Grouping.** Two records are linked when they are within reach of each other and either share a
//! surface edge, or one's measure ended on a triangle the other is a corner of or stands on (they
//! face each other across the wall), or they are columns of one face or of two faces sharing an
//! edge, or a column and a corner of its face. Within reach means closer than
//! [`WALL_WIDTH_RATIO`] times the thinner of the two thicknesses, plus one grid step when either is
//! a column (neighbouring columns of a face are at most one step apart). That is the width past
//! which wall fill lays columns across a face: a longer edge or a larger far triangle does not link
//! its ends, which may be different walls (a thin plate either side of a thick boss), and the
//! columns laid across it link the wall instead. Only one step is allowed, never two, because a
//! step grows with the triangle (a sixteenth of its longest edge): on a long coarse face two steps
//! reach over the unrecorded columns under a thick boss. So one unrecorded column inside an
//! otherwise thin face can split its spot in two; two warnings for one wall were chosen over one
//! warning spanning a thick region. A spot never reaches past its records: where a fin meets a thick body the ray of a vertex
//! on the junction runs into the body, so the spot stops at the last recorded measure before it.
//!
//! **Order.** Spots come thinnest first, ties by the earlier record (vertices first, by index, then
//! columns in the order the faces are visited), and links join the lower record as the root, so a
//! run lists the same spots in the same order every time.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::layers::Hit;
use crate::refine::SizeFn;
use crate::wall_fill::{WallColumn, WALL_WIDTH_RATIO};

/// One place where a wall is too thin for the elements asked for there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThinWallSpot {
    /// The surface vertex where the spot is thinnest, an index into the refined mesh's points
    /// (surface vertices keep their indices and bits through refinement); `None` when the thinnest
    /// measure is a column through the middle of a face, which stands on no vertex.
    pub vertex: Option<u32>,
    /// Where the thinnest measure starts on the surface.
    pub position: [f64; 3],
    /// The wall's thickness there.
    pub thickness: f64,
    /// The element size the size function asked for there, read at the middle of the wall.
    pub size: f64,
    /// How many surface vertices the spot holds; zero when only columns measured it.
    pub vertices: usize,
    /// The low corner of the box around the spot's measures, vertices and columns.
    pub low: [f64; 3],
    /// The high corner of that box.
    pub high: [f64; 3],
}

/// The size asked for in the middle of the chord from `origin` along unit `direction` for
/// `distance`, or `None` when the size function gives no finite positive value there.
pub(crate) fn chord_size(
    size: &SizeFn,
    origin: [f64; 3],
    direction: [f64; 3],
    distance: f64,
) -> Option<f64> {
    let half = 0.5 * distance;
    let middle = [0, 1, 2].map(|k| origin[k] + direction[k] * half);
    let value = size(middle);
    (value.is_finite() && value > 0.0).then_some(value)
}

/// One recorded measure: a vertex's ray or a column of a face.
#[derive(Clone, Copy)]
struct Record {
    vertex: Option<u32>,
    /// The face a column stands on; `None` for a vertex.
    face: Option<[u32; 3]>,
    position: [f64; 3],
    thickness: f64,
    size: f64,
    /// The triangle the measure ended on, across the wall.
    far: [u32; 3],
    /// How far the measure reaches to a neighbour along the surface: a column's grid step, zero for
    /// a vertex.
    reach: f64,
}

/// The recorded measures while the layer stage runs, and the spots once it ends.
pub(crate) struct ThinWalls {
    ratio: f64,
    records: Vec<Record>,
    spots: Option<Vec<ThinWallSpot>>,
}

impl ThinWalls {
    pub(crate) fn new(ratio: f64) -> Self {
        Self {
            ratio,
            records: Vec::new(),
            spots: None,
        }
    }

    /// Looks at `vertex`, at `position`, whose inward ray reached the far side of the wall from
    /// inside at `hit`, with `size` asked for in the wall there.
    pub(crate) fn observe_vertex(
        &mut self,
        vertex: u32,
        position: [f64; 3],
        hit: &Hit,
        size: Option<f64>,
        too_thin_for_rounding: bool,
    ) {
        let record = Record {
            vertex: Some(vertex),
            face: None,
            position,
            thickness: hit.distance,
            size: 0.0,
            far: hit.face,
            reach: 0.0,
        };
        self.observe(record, size, too_thin_for_rounding);
    }

    /// Looks at `column`, of surface triangle `face`, with `size` asked for in the wall there.
    pub(crate) fn observe_column(
        &mut self,
        face: [u32; 3],
        column: &WallColumn,
        size: Option<f64>,
        too_thin_for_rounding: bool,
    ) {
        let record = Record {
            vertex: None,
            face: Some(face),
            position: column.origin,
            thickness: column.distance,
            size: 0.0,
            far: column.far,
            // `across` is half the face's grid step.
            reach: 2.0 * column.across,
        };
        self.observe(record, size, too_thin_for_rounding);
    }

    /// Records `record` when the wall is thinner than `size / ratio` or `too_thin_for_rounding`,
    /// at most once: a second look at the same measure comes straight after its first.
    fn observe(&mut self, mut record: Record, size: Option<f64>, too_thin_for_rounding: bool) {
        let Some(size) = size else {
            return;
        };
        if !(too_thin_for_rounding || record.thickness < size / self.ratio) {
            return;
        }
        if self.records.last().is_some_and(|last| {
            last.vertex == record.vertex
                && last.face == record.face
                && last.position == record.position
        }) {
            return;
        }
        record.size = size;
        self.records.push(record);
    }

    /// Groups the recorded measures into spots over the surface triangles `boundary`.
    pub(crate) fn group(&mut self, boundary: &[[u32; 3]]) {
        let records = &self.records;
        let mut vertex_record: BTreeMap<u32, usize> = BTreeMap::new();
        let mut face_columns: BTreeMap<[u32; 3], Vec<usize>> = BTreeMap::new();
        for (index, record) in records.iter().enumerate() {
            if let Some(v) = record.vertex {
                vertex_record.entry(v).or_insert(index);
            }
            if let Some(face) = record.face {
                face_columns.entry(sorted(face)).or_default().push(index);
            }
        }
        let mut edge_faces: BTreeMap<(u32, u32), Vec<[u32; 3]>> = BTreeMap::new();
        if !face_columns.is_empty() {
            for face in boundary {
                for key in edges(*face) {
                    edge_faces.entry(key).or_default().push(*face);
                }
            }
        }
        let mut links = Links::new(records);
        for face in boundary {
            for (u, v) in edges(*face) {
                if let (Some(&a), Some(&b)) = (vertex_record.get(&u), vertex_record.get(&v)) {
                    links.join_near(a, b);
                }
            }
        }
        let no_columns: &[usize] = &[];
        let columns_on = |face: [u32; 3]| {
            face_columns
                .get(&sorted(face))
                .map_or(no_columns, Vec::as_slice)
        };
        for (a, record) in records.iter().enumerate() {
            let mut corners = record.far.to_vec();
            let mut faces = vec![sorted(record.far)];
            if let Some(face) = record.face {
                corners.extend(face);
                for key in edges(face) {
                    // Holds `face` itself, once per edge; the sort and dedup below keep one.
                    faces.extend(
                        edge_faces
                            .get(&key)
                            .into_iter()
                            .flatten()
                            .map(|f| sorted(*f)),
                    );
                }
            }
            faces.sort_unstable();
            faces.dedup();
            for v in corners {
                if let Some(&b) = vertex_record.get(&v) {
                    links.join_near(a, b);
                }
            }
            for face in faces {
                for &b in columns_on(face) {
                    links.join_near(a, b);
                }
            }
        }
        self.spots = Some(links.spots());
    }

    /// The spots, thinnest first, once the layer stage has ended; `None` before.
    pub(crate) fn spots(&self) -> Option<&[ThinWallSpot]> {
        self.spots.as_deref()
    }
}

/// The records and which of them are linked, as a union-find whose roots are the lowest record.
struct Links<'a> {
    records: &'a [Record],
    parent: Vec<usize>,
}

impl<'a> Links<'a> {
    fn new(records: &'a [Record]) -> Self {
        Self {
            records,
            parent: (0..records.len()).collect(),
        }
    }

    /// Joins `a` and `b` when they are within reach of each other (this file's header).
    fn join_near(&mut self, a: usize, b: usize) {
        let (first, second) = (&self.records[a], &self.records[b]);
        let d = [0, 1, 2].map(|k| first.position[k] - second.position[k]);
        let apart = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let thinner = first.thickness.min(second.thickness);
        if apart >= WALL_WIDTH_RATIO * thinner + first.reach.max(second.reach) {
            return;
        }
        let (ra, rb) = (self.root_of(a), self.root_of(b));
        if ra != rb {
            self.parent[ra.max(rb)] = ra.min(rb);
        }
    }

    fn root_of(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    /// One spot per group, thinnest first.
    fn spots(mut self) -> Vec<ThinWallSpot> {
        let records = self.records;
        let mut slot_of_root: Vec<Option<usize>> = vec![None; records.len()];
        // Each spot with the record of its thinnest measure, the tie-break between spots.
        let mut spots: Vec<(ThinWallSpot, usize)> = Vec::new();
        for (index, record) in records.iter().enumerate() {
            let counted = usize::from(record.vertex.is_some());
            let root = self.root_of(index);
            let Some(slot) = slot_of_root[root] else {
                slot_of_root[root] = Some(spots.len());
                let spot = ThinWallSpot {
                    vertex: record.vertex,
                    position: record.position,
                    thickness: record.thickness,
                    size: record.size,
                    vertices: counted,
                    low: record.position,
                    high: record.position,
                };
                spots.push((spot, index));
                continue;
            };
            let (spot, thinnest) = &mut spots[slot];
            spot.vertices += counted;
            for k in 0..3 {
                spot.low[k] = spot.low[k].min(record.position[k]);
                spot.high[k] = spot.high[k].max(record.position[k]);
            }
            // Records come in order, so a tie keeps the earlier one.
            if record.thickness < spot.thickness {
                spot.vertex = record.vertex;
                spot.position = record.position;
                spot.thickness = record.thickness;
                spot.size = record.size;
                *thinnest = index;
            }
        }
        spots.sort_by(thinnest_first);
        spots.into_iter().map(|(spot, _)| spot).collect()
    }
}

fn thinnest_first(a: &(ThinWallSpot, usize), b: &(ThinWallSpot, usize)) -> Ordering {
    a.0.thickness
        .total_cmp(&b.0.thickness)
        .then_with(|| a.1.cmp(&b.1))
}

fn sorted(mut face: [u32; 3]) -> [u32; 3] {
    face.sort_unstable();
    face
}

/// A triangle's three edges, each with its lower vertex first.
fn edges(face: [u32; 3]) -> [(u32, u32); 3] {
    [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])].map(|(u, v)| (u.min(v), u.max(v)))
}
