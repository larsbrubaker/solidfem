//! Reproducible surfaces for the real-part benchmarks: each part is built through solidmesh's CSG,
//! extrude and revolve operations, then remeshed with `remesh_for_analysis` at each mesh density
//! and written in the text format `solidfem-gate file` and `solidfem-benchmarks` read
//! (`benchmarks::pipeline::read_surface`).
//!
//! This file holds the table of parts and densities, the remesh of one density, the text format
//! and the errors. `shapes.rs` holds the shape of each part. `src/bin/surfaces.rs` is the
//! `solidfem-surfaces` command line over them. Each `SurfaceDensity` states the same `Sizing` as
//! the benchmark case that reads its file, so the surface and the interior of the volume mesh
//! grade alike; the `surfaces` tests hold the table and `benchmarks::cases` to that.
//!
//! Everything here depends on solidmesh and is native only, as the corpus measurement is.

pub mod shapes;

use std::fmt;
use std::path::Path;

use solidmesh_mesh::{
    remesh_for_analysis, AnalysisRemesh, AnalysisRemeshError, AnalysisRemeshOptions,
    AnalysisSurface, CancelToken, Mesh,
};

use crate::benchmarks::cases::Sizing;

/// Why a surface was not produced.
#[derive(Debug)]
pub enum SurfaceError {
    /// A CSG, extrude or revolve operation failed while building the part.
    Shape { part: &'static str, reason: String },
    /// The remesher refused the part at this density.
    Refused {
        stem: String,
        reason: AnalysisRemeshError,
    },
    /// A surface file could not be written.
    Write {
        path: String,
        source: std::io::Error,
    },
    /// A name on the command line is neither a part nor a density.
    UnknownName { name: String, known: Vec<String> },
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SurfaceError::Shape { part, reason } => {
                write!(f, "could not build part {part}: {reason}")
            }
            SurfaceError::Refused { stem, reason } => {
                write!(f, "the remesher refused {stem}: {reason}")
            }
            SurfaceError::Write { path, source } => write!(f, "cannot write {path}: {source}"),
            SurfaceError::UnknownName { name, known } => {
                write!(f, "no part or density is named {name}; the names are: ")?;
                write!(f, "{}", known.join(" "))
            }
        }
    }
}

impl std::error::Error for SurfaceError {}

/// One mesh density of a part.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceDensity {
    /// The file stem: the surface is written to `<stem>.txt`.
    pub stem: String,
    /// The edge length asked for at each position.
    pub size: Sizing,
}

/// One benchmark part and the densities it is remeshed at, coarse to fine.
pub struct SurfacePart {
    /// The name it is selected by, also the start of its files' names.
    pub name: &'static str,
    /// How far the remeshed surface may stray from the built part, in the part's units.
    pub deviation: f64,
    pub densities: Vec<SurfaceDensity>,
    build: fn(&CancelToken) -> Result<Mesh, SurfaceError>,
}

impl SurfacePart {
    /// Builds the part's closed triangle mesh.
    pub fn mesh(&self, cancel: &CancelToken) -> Result<Mesh, SurfaceError> {
        (self.build)(cancel)
    }
}

fn part(
    name: &'static str,
    deviation: f64,
    build: fn(&CancelToken) -> Result<Mesh, SurfaceError>,
    labels: &[(&str, f64)],
    size: impl Fn(f64) -> Sizing,
) -> SurfacePart {
    SurfacePart {
        name,
        deviation,
        densities: labels
            .iter()
            .map(|&(label, value)| SurfaceDensity {
                stem: format!("{name}_{label}"),
                size: size(value),
            })
            .collect(),
        build,
    }
}

/// A size field around `centre`: `near` within `r0`, growing at slope `grade` up to `far`.
fn around(centre: [f64; 2], r0: f64, grade: f64, far: f64) -> impl Fn(f64) -> Sizing {
    move |near| Sizing::AroundPoint {
        centre,
        r0,
        near,
        grade,
        far,
    }
}

/// Every benchmark part with its densities, in the order the benchmarks run them.
pub fn all_parts() -> Vec<SurfacePart> {
    let uniform = Sizing::Uniform;
    vec![
        part(
            "strip",
            0.05,
            shapes::strip,
            &[("h2.0", 2.0), ("h1.4", 1.4), ("h1.0", 1.0)],
            uniform,
        ),
        part(
            "plate",
            0.02,
            shapes::plate_with_hole,
            &[("n1.2", 1.2), ("n0.8", 0.8), ("n0.55", 0.55)],
            around([150.0, 50.0], 5.0, 0.3, 6.0),
        ),
        part(
            "ring",
            0.03,
            shapes::ring,
            &[("h1.4", 1.4), ("h1.0", 1.0), ("h0.75", 0.75)],
            uniform,
        ),
        part(
            "lsharp",
            0.03,
            shapes::l_bracket_sharp,
            &[
                ("c1.0", 1.0),
                ("c0.5", 0.5),
                ("c0.25", 0.25),
                ("c0.125", 0.125),
            ],
            around([10.0, 10.0], 0.0, 0.25, 4.0),
        ),
        part(
            "lfillet",
            0.02,
            shapes::l_bracket_filleted,
            &[("f1.2", 1.2), ("f0.8", 0.8), ("f0.5", 0.5)],
            around([15.0, 15.0], 6.0, 0.25, 4.0),
        ),
        part(
            "le1",
            0.5,
            shapes::nafems_le1,
            &[("n60", 60.0), ("n40", 40.0), ("n28", 28.0)],
            around([2000.0, 0.0], 0.0, 0.2, 120.0),
        ),
        part(
            "le10",
            0.5,
            shapes::nafems_le10,
            &[("n120", 120.0), ("n85", 85.0), ("n60", 60.0)],
            around([2000.0, 0.0], 0.0, 0.25, 150.0),
        ),
        part(
            "hook",
            0.03,
            shapes::hook,
            &[("h3.0", 3.0), ("h2.2", 2.2), ("h1.6", 1.6)],
            uniform,
        ),
        part(
            "snap",
            0.02,
            shapes::snap_fit,
            &[("h1.0", 1.0), ("h0.7", 0.7), ("h0.5", 0.5)],
            uniform,
        ),
        part(
            "hollow",
            0.05,
            shapes::hollow_box,
            &[("h2.0", 2.0), ("h1.5", 1.5), ("h1.0", 1.0)],
            uniform,
        ),
        part(
            "gear",
            0.02,
            shapes::gear_tooth,
            &[("n0.9", 0.9), ("n0.6", 0.6), ("n0.4", 0.4)],
            around([0.0, 38.0], 10.0, 0.3, 3.0),
        ),
    ]
}

/// The parts `names` select: a part's name selects all its densities, a stem such as `gear_n0.6`
/// selects that one. No names selects everything.
pub fn select(names: &[String]) -> Result<Vec<SurfacePart>, SurfaceError> {
    let mut parts = all_parts();
    if names.is_empty() {
        return Ok(parts);
    }
    let known: Vec<String> = parts
        .iter()
        .flat_map(|part| {
            std::iter::once(part.name.to_string())
                .chain(part.densities.iter().map(|density| density.stem.clone()))
        })
        .collect();
    if let Some(name) = names.iter().find(|name| !known.contains(name)) {
        return Err(SurfaceError::UnknownName {
            name: name.clone(),
            known,
        });
    }
    for part in &mut parts {
        if !names.iter().any(|name| name == part.name) {
            part.densities
                .retain(|density| names.contains(&density.stem));
        }
    }
    parts.retain(|part| !part.densities.is_empty());
    Ok(parts)
}

/// Remeshes `mesh`, which `part` built, at one of its densities.
pub fn remesh_density(
    mesh: &Mesh,
    part: &SurfacePart,
    density: &SurfaceDensity,
    cancel: &CancelToken,
) -> Result<AnalysisRemesh, SurfaceError> {
    let size = density.size;
    let size_at = move |p: [f64; 3]| size.at(p);
    let mut options = AnalysisRemeshOptions::new(&size_at, part.deviation);
    options.cancel = Some(cancel);
    remesh_for_analysis(mesh, &options).map_err(|reason| SurfaceError::Refused {
        stem: density.stem.clone(),
        reason,
    })
}

/// `surface` as text: a `vertices faces` line, then one line per vertex and per face. Each
/// coordinate is written as the shortest text that parses back to the same `f64`, so the file
/// holds the exact coordinates the remesher returned.
pub fn surface_text(surface: &AnalysisSurface) -> String {
    use std::fmt::Write as _;
    let mut text = String::new();
    // Writing to a String cannot fail.
    let _ = writeln!(text, "{} {}", surface.vertices.len(), surface.faces.len());
    for p in &surface.vertices {
        let _ = writeln!(text, "{} {} {}", p[0], p[1], p[2]);
    }
    for t in &surface.faces {
        let _ = writeln!(text, "{} {} {}", t[0], t[1], t[2]);
    }
    text
}

/// Writes `surface` to `path` in the text format, replacing any file there.
pub fn write_surface(path: &Path, surface: &AnalysisSurface) -> Result<(), SurfaceError> {
    std::fs::write(path, surface_text(surface)).map_err(|source| SurfaceError::Write {
        path: path.display().to_string(),
        source,
    })
}
