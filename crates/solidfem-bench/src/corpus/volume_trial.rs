//! One volume mesh of a remeshed part and what came of it: the remeshed surface through the same
//! volume stage the benchmarks run (`benchmarks/volume.rs`: surface recovery, refinement with
//! layers through thin walls, the 10-node upgrade), at the remesh's own edge length, and the
//! outcome as a stable short name so a corpus can be counted by failure class. A finished mesh is
//! kept on the result for the solve stage (`solve_trial.rs`).
//!
//! The work is capped three ways so one part costs at most about a minute and stays inside the
//! memory the budget promises: the edge length, chosen once for the remesh and the volume alike
//! (`remesh_trial::choose_target`), is sized for the corpus's unknowns budget; the mesher is held
//! to a hard element cap a little above that budget ([`tet_cap`]), whose size stage stops at the
//! cap and still finishes (recorded as `size_stopped`), refusing only a part whose surface and
//! wall layers alone exceed it (`tetra:TetBudgetExceeded`); and the run is cancelled at
//! [`TIME_CAP_SECONDS`], which is its own named outcome.

use std::time::Instant;

use solidfem::elastic::Quadrature;
use solidfem::mesh::QuadraticTetMesh;
use solidfem::tetra::{RefineOptions, SizeFn};

use super::remesh_trial::UNKNOWNS_BUDGET;
use super::soup::TriSoup;
use crate::benchmarks::cases::min_dihedral_degrees;
use crate::benchmarks::sizing::{TET_CAP_HEADROOM, UNKNOWNS_PER_TET};
use crate::benchmarks::volume::{volume_mesh, VolumeFailure, VolumeStage};

/// The volume stage is cancelled after this long, so a part costs its own row, not the batch's
/// whole timeout.
pub const TIME_CAP_SECONDS: f64 = 60.0;

/// Elements requested through each thin wall, as every benchmark case requests.
const LAYERS_THROUGH_WALLS: u32 = 2;

/// The mesher's hard element cap: the corpus's unknowns budget in elements, with
/// [`TET_CAP_HEADROOM`] above it. A part whose estimate ran low then gets a coarser interior and
/// stays inside the memory the budget promises, rather than being refused.
pub fn tet_cap() -> usize {
    // The value is finite and about 2.6e5, far below `usize::MAX` on any target, so the cast
    // cannot saturate.
    (UNKNOWNS_BUDGET / UNKNOWNS_PER_TET * TET_CAP_HEADROOM).ceil() as usize
}

/// What the volume stage did to one remeshed part.
#[derive(Clone, Debug)]
pub struct VolumeTrial {
    pub edge_length: f64,
    pub seconds: f64,
    pub outcome: Result<VolumePart, VolumeFailure>,
}

/// A volume mesh that reached 10-node elements, measured.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumePart {
    pub tets: usize,
    /// Nodes of the 10-node mesh.
    pub nodes: usize,
    /// The smallest dihedral angle of any element, degrees.
    pub min_dihedral_degrees: f64,
    /// Elements whose smallest dihedral angle is under 1 degree.
    pub under_one_degree: usize,
    /// Elements whose smallest dihedral angle is under 5 degrees.
    pub under_five_degrees: usize,
    /// The size stage stopped at [`tet_cap`], so the interior is coarser than the edge asks.
    pub size_stopped: bool,
    /// Size points, quality flips and quality points the cap refused.
    pub refused_moves: usize,
    /// The 10-node mesh itself, which the solve stage (`solve_trial.rs`) solves.
    pub mesh: QuadraticTetMesh,
}

/// Volume-meshes the closed `surface`, remeshed at `edge`, at that same edge length, and
/// measures the result.
pub fn volume_and_measure(surface: &TriSoup, edge: f64) -> VolumeTrial {
    let start = Instant::now();
    let size: SizeFn = Box::new(move |_| edge);
    let options = RefineOptions {
        layers_through_walls: LAYERS_THROUGH_WALLS,
        // The mesher keeps the cap as a promise, refusing a surface that alone exceeds it.
        max_tets: tet_cap(),
        ..RefineOptions::default()
    };
    let mut past_cap = || start.elapsed().as_secs_f64() >= TIME_CAP_SECONDS;
    let outcome = volume_mesh(
        (surface.vertices.clone(), surface.triangles.clone()),
        size,
        options,
        Quadrature::FourPoint,
        &mut past_cap,
    )
    .map(|volume| {
        let solid = &volume.refined;
        let angles: Vec<f64> = solid
            .tets
            .iter()
            .map(|tet| min_dihedral_degrees(tet.map(|n| solid.points[n as usize])))
            .collect();
        VolumePart {
            tets: volume.mesh.tets().len(),
            nodes: volume.mesh.nodes().len(),
            min_dihedral_degrees: angles.iter().copied().fold(f64::INFINITY, f64::min),
            under_one_degree: angles.iter().filter(|a| **a < 1.0).count(),
            under_five_degrees: angles.iter().filter(|a| **a < 5.0).count(),
            size_stopped: volume.report.size_stopped_at_budget,
            refused_moves: volume.report.moves_refused_for_budget,
            mesh: volume.mesh,
        }
    });
    VolumeTrial {
        edge_length: edge,
        seconds: start.elapsed().as_secs_f64(),
        outcome,
    }
}

/// The stable short name of a volume outcome: `ok`, `tetra:<variant>` for the tetrahedral
/// mesher's errors (surface recovery and refinement), `quadratic:<variant>` for the 10-node
/// upgrade's, and `time-cap:<stage>` for a run cancelled at [`TIME_CAP_SECONDS`].
pub fn outcome_name(outcome: &Result<VolumePart, VolumeFailure>) -> String {
    match outcome {
        Ok(_) => "ok".to_string(),
        Err(
            VolumeFailure::Surface(error)
            | VolumeFailure::Recovery(error)
            | VolumeFailure::Refiner(error)
            | VolumeFailure::Refinement(error),
        ) => format!("tetra:{}", variant_path(&format!("{error:?}"))),
        Err(VolumeFailure::Upgrade(error)) => {
            format!("quadratic:{}", variant_path(&format!("{error:?}")))
        }
        Err(VolumeFailure::Stopped { stage, .. }) => match stage {
            VolumeStage::Recovery => "time-cap:recovery".to_string(),
            VolumeStage::Refinement => "time-cap:refinement".to_string(),
        },
    }
}

/// The variant names at the head of a derived `Debug` text, a wrapped variant joined to its
/// wrapper by a dot: `Mesh(Empty)` is `Mesh.Empty`, `SurfaceNotRecovered { .. }` is
/// `SurfaceNotRecovered`. The field values, which differ per part, are dropped.
pub(crate) fn variant_path(debug: &str) -> String {
    let mut names = Vec::new();
    let mut rest = debug;
    loop {
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        names.push(&rest[..end]);
        match rest[end..].strip_prefix('(') {
            Some(inner) if inner.starts_with(|c: char| c.is_ascii_uppercase()) => rest = inner,
            _ => break,
        }
    }
    names.join(".")
}
