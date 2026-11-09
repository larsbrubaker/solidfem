//! The benchmark surface generator: `solidfem-surfaces <output-dir> [name ...]` builds each
//! benchmark part through solidmesh and remeshes it with `remesh_for_analysis` at each density,
//! writing one `<stem>.txt` per surface. The parts and densities live in the library
//! (`solidfem_bench::surfaces`); this file is only the command line. Run in release.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::Path;
    use std::process::ExitCode;
    use std::time::Instant;

    use solidfem_bench::surfaces::{remesh_density, select, write_surface, SurfaceError};
    use solidmesh_mesh::CancelToken;

    const USAGE: &str = "\
usage: solidfem-surfaces <output-dir> [name ...]

Builds each real-part benchmark surface and writes it to <output-dir>/<stem>.txt, for example
gear_n0.6.txt. A name selects a whole part (gear) or one density (gear_n0.6); with no names every
benchmark surface is generated. A coarser surface used only by the demo page, such as
hollow_demo_h4.0, is generated only when named. A surface the remesher refuses is reported and the rest still generate; the
exit status is non-zero if any was refused. Run in release: the finest densities take minutes.

To run the benchmarks on the result, give the same directory to solidfem-benchmarks:

    cargo run --release --bin solidfem-surfaces -- surfaces
    cargo run --release --bin solidfem-benchmarks -- surfaces [case-id ...]";

    fn run(directory: &str, names: &[String]) -> Result<usize, SurfaceError> {
        let parts = select(names)?;
        std::fs::create_dir_all(directory).map_err(|source| SurfaceError::Write {
            path: directory.to_string(),
            source,
        })?;
        let cancel = CancelToken::new();
        let mut refused = 0;
        for part in &parts {
            let mesh = part.mesh(&cancel)?;
            for density in &part.densities {
                let started = Instant::now();
                match remesh_density(&mesh, part, density, &cancel) {
                    Ok(remeshed) => {
                        let path = Path::new(directory).join(format!("{}.txt", density.stem));
                        write_surface(&path, &remeshed.surface)?;
                        println!(
                            "{}: {} vertices, {} faces, {:.1} s",
                            density.stem,
                            remeshed.surface.vertices.len(),
                            remeshed.surface.faces.len(),
                            started.elapsed().as_secs_f64()
                        );
                    }
                    Err(error @ SurfaceError::Refused { .. }) => {
                        println!("{error}");
                        refused += 1;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(refused)
    }

    /// Runs the command and reports what happened as an exit status.
    pub fn main() -> ExitCode {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let Some((directory, names)) = args.split_first() else {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        };
        match run(directory, names) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(refused) => {
                eprintln!(
                    "{refused} surface(s) refused; the rest are in {directory}. \
                     See each refusal above for what to change."
                );
                ExitCode::FAILURE
            }
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
