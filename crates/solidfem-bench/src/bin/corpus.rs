//! The corpus measurement binary. `solidfem-corpus synthetic` measures the built-in parts;
//! `solidfem-corpus sample <thingi-root> <out-dir> <count>` extracts a fixed every-Nth sample of
//! Thingi10K's `.stl.zip` files with the system `unzip`; `solidfem-corpus batch <jobs>
//! <timeout-seconds> <path>...` measures every `.stl` under the given files and directories, each
//! in its own child process (`solidfem-corpus stl <file>`), so a crash or a runaway remesh on one
//! untrusted file costs only that file's row. Each input gets a record for the mesh; a part whose
//! own reading says it is a closed, consistently wound solid (`corpus::measure::sample_exclusion`)
//! also gets one for the remesh and, when the remesh produced a surface, one for the volume mesh
//! of that surface (`corpus/volume_trial.rs`), and when that reached 10-node elements, one for a
//! linear static bending solve of the mesh (`corpus/solve_trial.rs`). A part left out is counted as `input:<reason>`,
//! never as a remesher or mesher failure. `batch` prints the records as tables and summaries, the
//! remesh, volume and solve outcomes counted by name. A child writes and flushes each record as
//! soon as it is known, so a child that times out or crashes in the volume or solve stage still
//! has its earlier ones, and the parent writes that stage's record as `child:timeout` or
//! `child:crashed`; the timeout must cover remesh, volume (60 s cap) and solve (120 s cap). `solidfem-corpus diagnose
//! <file-or-synthetic-name> <scale,...> <cap-seconds>` remeshes one part at each multiple of its
//! corpus target edge length and prints, a key=value line per multiple, what surface recovery
//! made of it and where anything it left missing sits (`corpus/diagnose.rs`).

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::VecDeque;
    use std::error::Error;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    use solidfem_bench::corpus::diagnose::diagnose;
    use solidfem_bench::corpus::measure::{measure_input, sample_exclusion};
    use solidfem_bench::corpus::parts::all_parts;
    use solidfem_bench::corpus::record::{
        by_dihedral_bucket, counts_by, distribution, distribution_sci, field, input_fields,
        parse_line, remesh_fields, share_nonzero, share_true, solve_fields, spread,
        stage_lost_fields, table, to_line, volume_fields, Record,
    };
    use solidfem_bench::corpus::remesh_trial::remesh_and_measure;
    use solidfem_bench::corpus::solve_trial::solve_and_check;
    use solidfem_bench::corpus::soup::TriSoup;
    use solidfem_bench::corpus::volume_trial::volume_and_measure;
    use solidmesh_mesh::processors::stl_processing::parse_file_contents;
    use solidmesh_mesh::{CancelToken, Mesh};

    const INPUT_COLUMNS: &[(&str, &str)] = &[
        ("name", "part"),
        ("tris", "tris"),
        ("comps", "cc"),
        ("closed", "closed"),
        ("oriented", "oriented"),
        ("bedges", "open"),
        ("nmedges", "nonman"),
        ("misedges", "flip"),
        ("xpairs", "x-pairs"),
        ("folds", "folds"),
        ("degen", "degen"),
        ("armed", "AR med"),
        ("ar95", "AR p95"),
        ("ar99", "AR p99"),
        ("arworst", "AR max"),
        ("minang", "min deg"),
        ("emin", "edge min"),
        ("emed", "edge med"),
        ("emax", "edge max"),
        ("sharp", "sharp"),
        ("wallp5", "wall p5"),
        ("wallmin", "wall min"),
        ("diag_wallp5", "diag/wall"),
    ];

    const REMESH_COLUMNS: &[(&str, &str)] = &[
        ("name", "part"),
        ("status", "status"),
        ("hsrc", "h from"),
        ("h", "h"),
        ("pred_tets", "pred tets"),
        ("walls", "walls"),
        ("wall_ratio", "h/(wall/2)"),
        ("tris_after", "tris"),
        ("closed_after", "closed"),
        ("oriented_after", "oriented"),
        ("vol_pct", "vol %"),
        ("dev_out", "out>in"),
        ("dev_in", "in>out"),
        ("sharp_in", "sharp in"),
        ("sharp_after", "sharp out"),
        ("sharp_pct", "kept %"),
        ("ar_med_after", "AR med"),
        ("ar_worst_after", "AR max"),
        ("min_ang_after", "min deg"),
        ("degen_after", "degen"),
        ("xpairs_after", "x-pairs"),
        ("folds_after", "folds"),
        ("sec", "sec"),
    ];

    const VOLUME_COLUMNS: &[(&str, &str)] = &[
        ("name", "part"),
        ("status", "status"),
        ("vh", "h"),
        ("tets", "tets"),
        ("nodes", "nodes"),
        ("mindih", "min dih"),
        ("lt1", "<1 deg"),
        ("lt5", "<5 deg"),
        ("size_stopped", "capped"),
        ("refused_moves", "refused"),
        ("sec", "sec"),
    ];

    const SOLVE_COLUMNS: &[(&str, &str)] = &[
        ("name", "part"),
        ("status", "status"),
        ("unknowns", "unknowns"),
        ("mindih", "min dih"),
        ("iters", "iters"),
        ("resid", "residual"),
        ("imbalance", "imbalance"),
        ("umax_diag", "umax/diag"),
        ("setup_sec", "setup s"),
        ("solve_sec", "solve s"),
        ("mg_bytes", "mg bytes"),
        ("sec", "sec"),
    ];

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.first().map(String::as_str) {
            Some("synthetic") => synthetic(),
            Some("sample") if args.len() == 4 => {
                sample(Path::new(&args[1]), Path::new(&args[2]), args[3].parse()?)
            }
            Some("batch") if args.len() >= 4 => batch(
                args[1].parse()?,
                Duration::from_secs(args[2].parse()?),
                &args[3..],
            ),
            Some("stl") if args.len() == 2 => measure_stl(Path::new(&args[1])),
            Some("diagnose") if args.len() == 4 => {
                diagnose_part(&args[1], &args[2], args[3].parse()?)
            }
            _ => {
                Err("usage: synthetic | sample <root> <out> <count> | batch <jobs> <timeout> <path>... | stl <file> | diagnose <file-or-synthetic-name> <scale,scale,...> <cap-seconds>".into())
            }
        }
    }

    /// Diagnoses surface recovery on one STL file, or on the synthetic part of that name, at each
    /// comma-separated multiple of the corpus target edge length, a line per multiple.
    fn diagnose_part(part: &str, scales: &str, cap_seconds: f64) -> Result<(), Box<dyn Error>> {
        let scales = scales
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<f64>, _>>()?;
        let (name, soup) = match all_parts()?.into_iter().find(|p| p.name == part) {
            Some(found) => (found.name.to_string(), TriSoup::from_mesh(&found.mesh)),
            None => {
                let path = Path::new(part);
                let bytes = std::fs::read(path)?;
                let mesh = parse_file_contents(&bytes, &CancelToken::new(), None)?;
                let name = path
                    .file_name()
                    .ok_or("no file name")?
                    .to_string_lossy()
                    .replace(' ', "_");
                (name, TriSoup::from_mesh(&mesh))
            }
        };
        for fields in diagnose(&name, &soup, &scales, cap_seconds) {
            println!("{}", to_line(&fields));
        }
        Ok(())
    }

    /// Measures one mesh, handing each record line to `emit` as soon as it is known.
    fn measure_mesh(
        name: &str,
        mesh: &Mesh,
        emit: &mut dyn FnMut(String) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let soup = TriSoup::from_mesh(mesh);
        let metrics = measure_input(&soup);
        emit(to_line(&input_fields(name, &metrics)))?;
        if sample_exclusion(&metrics).is_some() {
            return Ok(());
        }
        let trial = remesh_and_measure(&soup, &metrics);
        emit(to_line(&remesh_fields(name, &trial)))?;
        if let Ok(part) = &trial.outcome {
            let volume = volume_and_measure(&part.surface, trial.target_edge_length);
            emit(to_line(&volume_fields(name, &volume)))?;
            if let Ok(meshed) = &volume.outcome {
                let solved = solve_and_check(&meshed.mesh);
                emit(to_line(&solve_fields(
                    name,
                    &solved,
                    meshed.min_dihedral_degrees,
                )))?;
            }
        }
        Ok(())
    }

    fn measure_stl(path: &Path) -> Result<(), Box<dyn Error>> {
        let name = path
            .file_name()
            .ok_or("no file name")?
            .to_string_lossy()
            .replace(' ', "_");
        let bytes = std::fs::read(path)?;
        let mesh = parse_file_contents(&bytes, &CancelToken::new(), None)?;
        let mut stdout = std::io::stdout();
        measure_mesh(&name, &mesh, &mut |line| {
            writeln!(stdout, "{line}")?;
            stdout.flush()?;
            Ok(())
        })
    }

    fn synthetic() -> Result<(), Box<dyn Error>> {
        let mut records = Vec::new();
        for part in all_parts()? {
            let started = Instant::now();
            measure_mesh(part.name, &part.mesh, &mut |line| {
                records.push(parse_line(&line));
                Ok(())
            })?;
            eprintln!("{} took {:.1}s", part.name, started.elapsed().as_secs_f64());
        }
        print_tables(&records);
        Ok(())
    }

    /// Sorted by numeric id so the same dataset always gives the same sample.
    fn sample(root: &Path, out: &Path, count: usize) -> Result<(), Box<dyn Error>> {
        let mut zips = Vec::new();
        for shard in std::fs::read_dir(root)? {
            let meshes = shard?.path().join("meshes");
            if meshes.is_dir() {
                for entry in std::fs::read_dir(meshes)? {
                    let path = entry?.path();
                    if path.to_string_lossy().ends_with(".stl.zip") {
                        zips.push(path);
                    }
                }
            }
        }
        zips.sort_by_key(|p| (model_id(p), p.clone()));
        std::fs::create_dir_all(out)?;
        let stride = (zips.len() / count.max(1)).max(1);
        let picked: Vec<&PathBuf> = zips.iter().step_by(stride).take(count).collect();
        for zip in &picked {
            let status = Command::new("unzip")
                .args(["-o", "-q"])
                .arg(zip)
                .arg("-d")
                .arg(out)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()?;
            if !status.success() {
                eprintln!("could not unpack {}", zip.display());
            }
        }
        println!(
            "unpacked {} of {} zip files (every {}th) into {}",
            picked.len(),
            zips.len(),
            stride,
            out.display()
        );
        Ok(())
    }

    /// The leading number of a Thingi10K file name, the dataset's stable identifier.
    fn model_id(path: &Path) -> u64 {
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .and_then(|n| n.split('.').next().and_then(|id| id.parse().ok()))
            .unwrap_or(u64::MAX)
    }

    fn stl_files(paths: &[String]) -> Result<Vec<PathBuf>, Box<dyn Error>> {
        let mut files = Vec::new();
        for path in paths {
            let path = PathBuf::from(path);
            if path.is_dir() {
                for entry in std::fs::read_dir(&path)? {
                    let entry = entry?.path();
                    if is_stl(&entry) {
                        files.push(entry);
                    }
                }
            } else if is_stl(&path) {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    fn is_stl(path: &Path) -> bool {
        path.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("stl"))
    }

    /// One running child and how it is doing.
    struct Running {
        path: PathBuf,
        child: Child,
        started: Instant,
    }

    fn batch(jobs: usize, timeout: Duration, paths: &[String]) -> Result<(), Box<dyn Error>> {
        let exe = std::env::current_exe()?;
        let mut queue: VecDeque<PathBuf> = stl_files(paths)?.into();
        let total = queue.len();
        let mut running: Vec<Running> = Vec::new();
        let mut records: Vec<Record> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        while !queue.is_empty() || !running.is_empty() {
            while running.len() < jobs.max(1) && !queue.is_empty() {
                if let Some(path) = queue.pop_front() {
                    let child = Command::new(&exe)
                        .arg("stl")
                        .arg(&path)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null())
                        .spawn()?;
                    running.push(Running {
                        path,
                        child,
                        started: Instant::now(),
                    });
                }
            }
            std::thread::sleep(Duration::from_millis(100));
            let mut i = 0;
            while i < running.len() {
                match poll(&mut running[i], timeout)? {
                    Poll::Running => i += 1,
                    Poll::Done(lines) => {
                        records.extend(lines.iter().map(|l| parse_line(l)));
                        running.swap_remove(i);
                    }
                    Poll::Failed { reason, how, lines } => {
                        let partial: Vec<Record> = lines.iter().map(|l| parse_line(l)).collect();
                        match lost_in_stage(&partial) {
                            Some((name, kind)) => {
                                records.extend(partial);
                                let lost = stage_lost_fields(&name, kind, how);
                                records.push(parse_line(&to_line(&lost)));
                            }
                            None => {
                                failures.push(format!("{}: {reason}", running[i].path.display()));
                            }
                        }
                        running.swap_remove(i);
                    }
                }
            }
        }
        println!(
            "measured {} of {} files (a child lost in the volume or solve stage still counts)",
            total - failures.len(),
            total
        );
        for failure in &failures {
            println!("  no row: {failure}");
        }
        records.sort_by(|a, b| field(a, "name").cmp(field(b, "name")));
        print_tables(&records);
        Ok(())
    }

    enum Poll {
        Running,
        Done(Vec<String>),
        /// The child crashed or timed out, with the record lines it had written by then.
        Failed {
            reason: String,
            how: &'static str,
            lines: Vec<String>,
        },
    }

    /// The part's name and the stage the child was lost in: `volume` when its records hold a
    /// remesh that produced a surface and no volume record, `solve` when they hold a finished
    /// volume mesh and no solve record.
    fn lost_in_stage(partial: &[Record]) -> Option<(String, &'static str)> {
        let has = |kind: &str| partial.iter().any(|r| field(r, "kind") == kind);
        let finished = |kind: &str| {
            partial
                .iter()
                .find(|r| field(r, "kind") == kind && field(r, "status") == "ok")
                .map(|r| field(r, "name").to_string())
        };
        if let Some(name) = finished("volume") {
            return (!has("solve")).then_some((name, "solve"));
        }
        let name = finished("remesh")?;
        (!has("volume")).then_some((name, "volume"))
    }

    /// The child's whole standard output, once it has exited.
    fn written_lines(run: &mut Running) -> Result<Vec<String>, Box<dyn Error>> {
        let mut text = String::new();
        if let Some(mut out) = run.child.stdout.take() {
            std::io::Read::read_to_string(&mut out, &mut text)?;
        }
        Ok(text.lines().map(str::to_string).collect())
    }

    /// Checks one child: finished, crashed, or past its timeout (killed by its own handle).
    fn poll(run: &mut Running, timeout: Duration) -> Result<Poll, Box<dyn Error>> {
        if let Some(status) = run.child.try_wait()? {
            let lines = written_lines(run)?;
            if !status.success() {
                return Ok(Poll::Failed {
                    reason: format!("exited with {status}"),
                    how: "crashed",
                    lines,
                });
            }
            return Ok(Poll::Done(lines));
        }
        if run.started.elapsed() > timeout {
            run.child.kill()?;
            run.child.wait()?;
            return Ok(Poll::Failed {
                reason: format!("timed out after {timeout:?}"),
                how: "timeout",
                lines: written_lines(run)?,
            });
        }
        Ok(Poll::Running)
    }

    fn print_tables(records: &[Record]) {
        let inputs: Vec<Record> = records
            .iter()
            .filter(|r| field(r, "kind") == "input")
            .cloned()
            .collect();
        let remeshes: Vec<Record> = records
            .iter()
            .filter(|r| field(r, "kind") == "remesh")
            .cloned()
            .collect();
        println!("== INPUT ({} parts)", inputs.len());
        print!("{}", table(&inputs, INPUT_COLUMNS));
        println!("\n== SAMPLE");
        for (sample, count) in counts_by(&inputs, "sample") {
            println!("  {count:>5}  {sample}");
        }
        println!("\n== REMESH ({} attempted)", remeshes.len());
        print!("{}", table(&remeshes, REMESH_COLUMNS));
        println!("remesh outcomes, most frequent first:");
        for (status, count) in counts_by(&remeshes, "status") {
            println!("  {count:>5}  {status}");
        }
        println!("\n== SUMMARY");
        println!(
            "closed (every edge exactly two faces): {}",
            share_true(&inputs, "closed")
        );
        println!(
            "closed, manifold and consistently wound: {}",
            share_true(&inputs, "oriented")
        );
        println!(
            "has open boundary edges: {}",
            share_nonzero(&inputs, "bedges")
        );
        println!(
            "has non-manifold edges: {}",
            share_nonzero(&inputs, "nmedges")
        );
        println!(
            "has misoriented edges: {}",
            share_nonzero(&inputs, "misedges")
        );
        println!(
            "has penetrating triangle pairs: {}",
            share_nonzero(&inputs, "xpairs")
        );
        println!(
            "has folded triangle pairs: {}",
            share_nonzero(&inputs, "folds")
        );
        println!(
            "has degenerate triangles: {}",
            share_nonzero(&inputs, "degen")
        );
        println!(
            "more than one shell: {}",
            share_nonzero_above_one(&inputs, "comps")
        );
        println!("median triangle aspect ratio: {}", spread(&inputs, "armed"));
        println!(
            "worst triangle aspect ratio: {}",
            spread(&inputs, "arworst")
        );
        println!("sharp edges: {}", spread(&inputs, "sharp"));
        println!("diagonal / p5 wall: {}", spread(&inputs, "diag_wallp5"));
        let ok: Vec<Record> = remeshes
            .iter()
            .filter(|r| field(r, "status") == "ok")
            .cloned()
            .collect();
        println!("remesh produced a surface: {}/{}", ok.len(), remeshes.len());
        println!(
            "remesh output crosses itself (corpus's own count): {}",
            share_nonzero(&ok, "xpairs_after")
        );
        println!(
            "remesh output folds onto itself (corpus's own count): {}",
            share_nonzero(&ok, "folds_after")
        );
        println!("remesh stayed closed: {}", share_true(&ok, "closed_after"));
        println!(
            "remesh stayed oriented: {}",
            share_true(&ok, "oriented_after")
        );
        println!("remesh |volume change %|: {}", spread_abs(&ok, "vol_pct"));
        println!(
            "remesh output-to-input deviation / diag: {}",
            spread(&ok, "dev_out")
        );
        println!(
            "remesh input-to-output deviation / diag: {}",
            spread(&ok, "dev_in")
        );
        println!("remesh sharp edges kept %: {}", spread(&ok, "sharp_pct"));
        println!(
            "remesh min angle after (deg): {}",
            spread(&ok, "min_ang_after")
        );
        println!("remesh seconds: {}", distribution(&remeshes, "sec"));
        print_volume(records);
    }

    fn print_volume(records: &[Record]) {
        let volumes: Vec<Record> = records
            .iter()
            .filter(|r| field(r, "kind") == "volume")
            .cloned()
            .collect();
        println!("\n== VOLUME ({} remeshed parts attempted)", volumes.len());
        print!("{}", table(&volumes, VOLUME_COLUMNS));
        println!("\n== VOLUME SUMMARY");
        println!("outcomes, most frequent first:");
        for (status, count) in counts_by(&volumes, "status") {
            println!("  {count:>5}  {status}");
        }
        let ok: Vec<Record> = volumes
            .iter()
            .filter(|r| field(r, "status") == "ok")
            .cloned()
            .collect();
        println!(
            "volume meshed to 10-node elements: {}/{}",
            ok.len(),
            volumes.len()
        );
        println!("min dihedral (deg): {}", distribution(&ok, "mindih"));
        println!("tets under 1 deg: {}", distribution(&ok, "lt1"));
        println!("tets under 5 deg: {}", distribution(&ok, "lt5"));
        println!("tets: {}", distribution(&ok, "tets"));
        let capped = ok
            .iter()
            .filter(|r| field(r, "size_stopped") == "yes")
            .count();
        println!(
            "size stage stopped at the element cap: {capped}/{}",
            ok.len()
        );
        println!(
            "with moves refused for the cap: {}",
            share_nonzero(&ok, "refused_moves")
        );
        println!("seconds: {}", distribution(&ok, "sec"));
        print_solve(records);
    }

    fn print_solve(records: &[Record]) {
        let solves: Vec<Record> = records
            .iter()
            .filter(|r| field(r, "kind") == "solve")
            .cloned()
            .collect();
        println!(
            "\n== SOLVE ({} volume-meshed parts attempted)",
            solves.len()
        );
        print!("{}", table(&solves, SOLVE_COLUMNS));
        println!("\n== SOLVE SUMMARY");
        println!("outcomes, most frequent first:");
        for (status, count) in counts_by(&solves, "status") {
            println!("  {count:>5}  {status}");
        }
        let ok: Vec<Record> = solves
            .iter()
            .filter(|r| field(r, "status") == "ok")
            .cloned()
            .collect();
        println!(
            "solved with balanced reactions and a physical displacement: {}/{}",
            ok.len(),
            solves.len()
        );
        println!("iterations (ok): {}", distribution(&ok, "iters"));
        println!("iterations (any): {}", distribution(&solves, "iters"));
        println!("unknowns (any): {}", distribution(&solves, "unknowns"));
        println!(
            "setup seconds (any): {}",
            distribution(&solves, "setup_sec")
        );
        println!(
            "solve seconds (any): {}",
            distribution(&solves, "solve_sec")
        );
        println!("stage seconds (any): {}", distribution(&solves, "sec"));
        println!(
            "multigrid bytes (any): {}",
            distribution(&solves, "mg_bytes")
        );
        println!(
            "final relative residual (ok): {}",
            distribution_sci(&ok, "resid")
        );
        println!(
            "reaction imbalance (solved): {}",
            distribution_sci(&solves, "imbalance")
        );
        println!(
            "max displacement / diagonal (solved): {}",
            distribution_sci(&solves, "umax_diag")
        );
        println!("iterations by the mesh's worst dihedral (every record with iterations):");
        for (bucket, inside) in by_dihedral_bucket(&solves) {
            let ok_count = inside.iter().filter(|r| field(r, "status") == "ok").count();
            println!(
                "  {bucket:<13} parts {:>4}  ok {:>4}  iterations {}",
                inside.len(),
                ok_count,
                distribution(&inside, "iters")
            );
        }
    }

    fn share_nonzero_above_one(records: &[Record], key: &str) -> String {
        let hits = records
            .iter()
            .filter(|r| solidfem_bench::corpus::record::field_f64(r, key).is_some_and(|v| v > 1.0))
            .count();
        format!("{hits}/{}", records.len())
    }

    fn spread_abs(records: &[Record], key: &str) -> String {
        let absolute: Vec<Record> = records
            .iter()
            .map(|r| {
                let mut copy = r.clone();
                if let Some(v) = solidfem_bench::corpus::record::field_f64(r, key) {
                    copy.insert(key.to_string(), format!("{}", v.abs()));
                }
                copy
            })
            .collect();
        spread(&absolute, key)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = native::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
