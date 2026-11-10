//! The volume stage every measurement here shares: a closed surface through this workspace's
//! surface mesher, the refiner with a size function and through-wall layers, and the upgrade to
//! 10-node elements. `pipeline.rs` runs it before each benchmark solve and `corpus/volume_trial.rs`
//! runs it on every remeshed corpus part, so a mesher behaviour the corpus counts is the one the
//! benchmarks meet. A failure keeps the stage it happened in and the library's own error, so a
//! caller can word it or name it without parsing text.

use solidfem::elastic::Quadrature;
use solidfem::mesh::QuadraticTetMesh;
use solidfem::meshing::{quadratic_from_solid, MeshingError};
use solidfem::tetra::{
    RefineOptions, RefineReport, Refiner, SizeFn, SolidMesh, StepStatus, SurfaceMesher, TetraError,
};

use super::pipeline::Surface;

/// The stage a run of the volume stage was in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeStage {
    /// Recovering the surface inside the Delaunay tetrahedralization of its vertices.
    Recovery,
    /// Adding points inside the solid: layers, sizes and quality.
    Refinement,
}

/// Why a surface did not become a 10-node mesh, by the stage that refused it.
#[derive(Debug, Clone, PartialEq)]
pub enum VolumeFailure {
    /// The surface mesher refused the surface before starting.
    Surface(TetraError),
    /// Surface recovery failed.
    Recovery(TetraError),
    /// The refiner refused the recovered solid before starting.
    Refiner(TetraError),
    /// Refinement failed.
    Refinement(TetraError),
    /// A stage returned before finishing: the cancel check stopped it.
    Stopped {
        stage: VolumeStage,
        status: StepStatus,
    },
    /// The 10-node upgrade refused the refined solid.
    Upgrade(MeshingError),
}

/// A finished volume mesh.
#[derive(Debug, Clone)]
pub struct VolumeMesh {
    /// The refined 4-node solid the 10-node mesh was built from.
    pub refined: SolidMesh,
    /// The 10-node mesh.
    pub mesh: QuadraticTetMesh,
    /// What refinement did, with the dihedral angles before and after its quality pass.
    pub report: RefineReport,
}

/// Meshes the closed `surface` into 10-node elements: surface recovery, refinement under `size`
/// and `options`, then the upgrade checked against `quadrature`. `cancel` is asked between units
/// of work; a stage it stops is [`VolumeFailure::Stopped`]. `options.max_tets` is kept: the mesh
/// has at most that many tetrahedra, or the refiner's [`TetraError::TetBudgetExceeded`] comes back
/// as [`VolumeFailure::Refiner`] (the surface alone is over) or [`VolumeFailure::Refinement`]
/// (the layers took it over).
///
/// # Errors
/// The [`VolumeFailure`] naming the stage that refused and its error.
pub fn volume_mesh(
    (points, triangles): Surface,
    size: SizeFn,
    options: RefineOptions,
    quadrature: Quadrature,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<VolumeMesh, VolumeFailure> {
    let mut mesher = SurfaceMesher::new(points, triangles).map_err(VolumeFailure::Surface)?;
    match mesher.step(usize::MAX, cancel) {
        Ok(StepStatus::Done) => {}
        Ok(status) => {
            return Err(VolumeFailure::Stopped {
                stage: VolumeStage::Recovery,
                status,
            })
        }
        Err(error) => return Err(VolumeFailure::Recovery(error)),
    }
    let solid = mesher.result().map_err(VolumeFailure::Recovery)?;
    let mut refiner = Refiner::new(solid, size, options).map_err(VolumeFailure::Refiner)?;
    match refiner.step(usize::MAX, cancel) {
        Ok(StepStatus::Done) => {}
        Ok(status) => {
            return Err(VolumeFailure::Stopped {
                stage: VolumeStage::Refinement,
                status,
            })
        }
        Err(error) => return Err(VolumeFailure::Refinement(error)),
    }
    let refined = refiner.result().map_err(VolumeFailure::Refinement)?;
    let mesh = quadratic_from_solid(&refined, quadrature).map_err(VolumeFailure::Upgrade)?;
    Ok(VolumeMesh {
        refined,
        mesh,
        report: refiner.report(),
    })
}
