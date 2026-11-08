//! The native benchmark binary: `solidfem-bench known`, `solidfem-bench parts [name-filter]`, and
//! `solidfem-bench page <wasm-path> <out-path>` which writes the self-contained browser page with
//! the wasm embedded as base64. Each case is one line of output, so several processes can run
//! cases side by side.

use std::time::Instant;

use solidfem_bench::cases::part_cases;
use solidfem_bench::known_answer::{
    cantilever_variant, evaluate, measure_geometry, mid_span_bending_stress, timoshenko_deflection,
    EndSelection,
};
use solidfem_bench::pipeline::{
    advance, finish, prepare, Case, Ending, Measured, Prepared, Refusal,
};
use solidfem_bench::table::{parts_header, parts_row, refusal_row};
use solidfem_elastic::Clock;

const CAP_SECONDS: f64 = 120.0;
const BYTE_LIMIT: u64 = 16 * (1 << 30);
const SLICE: usize = 200;

fn native_clock() -> Clock {
    let start = Instant::now();
    Box::new(move || start.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("known") => known_answer(),
        Some("parts") => parts(args.get(1).map(String::as_str)),
        Some("page") => write_page(&args[1], &args[2]),
        _ => eprintln!("usage: solidfem-bench known | parts [filter] | page <wasm> <out>"),
    }
}

fn run(case: &Case) -> Result<(Prepared, Measured), Refusal> {
    let mut prepared = prepare(case, &mut native_clock, BYTE_LIMIT)?;
    let start = Instant::now();
    let ending = loop {
        if !advance(&mut prepared, SLICE)? {
            break Ending::Converged;
        }
        if start.elapsed().as_secs_f64() > CAP_SECONDS {
            break Ending::Capped;
        }
    };
    let measured = finish(&mut prepared, ending)?;
    Ok((prepared, measured))
}

fn known_answer() {
    println!("{}", parts_header());
    println!("  deflection error against Timoshenko, stress error against Mc/I");
    let variants = [
        (0.0, EndSelection::Plane),
        (0.0, EndSelection::Band),
        (0.37, EndSelection::Band),
    ];
    for (shift, selection) in variants {
        println!("-- grid shift {shift} cells, end selection {selection:?}");
        for cells in [3u32, 4, 6, 8, 12, 16, 24] {
            known_row(cells, shift, selection);
        }
    }
}

fn known_row(cells: u32, shift: f64, selection: EndSelection) {
    let case = cantilever_variant(cells, shift, selection);
    let (prepared, measured) = match run(&case) {
        Ok(run) => run,
        Err(refusal) => {
            println!("{}", refusal_row(&case.name, &format!("{refusal:?}")));
            return;
        }
    };
    println!("{}", parts_row(&case.name, &prepared, &measured));
    let Some(solution) = &measured.solution else {
        println!("  capped");
        return;
    };
    let (tip, stress) = evaluate(&prepared.mesh, solution, prepared.dx, selection);
    let geometry = measure_geometry(&prepared.mesh, prepared.dx, selection);
    println!(
        "  tip {:+.3}%  peak stress {:+.3}%  worst dihedral {:.1}  volume {:.4} I {:.4} root area {:.3} tip area {:.3}",
        100.0 * (tip / timoshenko_deflection() - 1.0),
        100.0 * (stress / mid_span_bending_stress() - 1.0),
        prepared.max_dihedral,
        geometry.volume_ratio,
        geometry.second_moment_ratio,
        geometry.root_area_ratio,
        geometry.tip_area_ratio,
    );
}

fn parts(filter: Option<&str>) {
    println!("{}", parts_header());
    for case in part_cases() {
        if filter.is_some_and(|f| !case.name.contains(f)) {
            continue;
        }
        match run(&case) {
            Ok((prepared, measured)) => println!("{}", parts_row(&case.name, &prepared, &measured)),
            Err(refusal) => println!("{}", refusal_row(&case.name, &format!("{refusal:?}"))),
        }
    }
}

fn write_page(wasm_path: &str, out_path: &str) {
    let wasm = match std::fs::read(wasm_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read {wasm_path}: {error}");
            std::process::exit(1);
        }
    };
    let page = PAGE.replace("WASM_BASE64", &base64(&wasm));
    if let Err(error) = std::fs::write(out_path, page) {
        eprintln!("cannot write {out_path}: {error}");
        std::process::exit(1);
    }
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len() * 4 / 3 + 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let value = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[(value >> 18) as usize & 63] as char);
        out.push(TABLE[(value >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(value >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[value as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// The page: decodes the wasm, runs it in slices on `setTimeout`, prints to the page and console.
const PAGE: &str = r#"<!doctype html>
<meta charset="utf-8"><title>solidfem browser benchmark</title>
<pre id="out">starting...
</pre>
<script>
const out = document.getElementById("out");
function line(text) { out.textContent += text + "\n"; console.log(text); }
const b64 = "WASM_BASE64";
const raw = atob(b64);
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
let instance;
function read(pointer, length) {
  return new TextDecoder().decode(new Uint8Array(instance.exports.memory.buffer, pointer, length));
}
const imports = { env: {
  now: () => performance.now() / 1000,
  log: (pointer, length) => line(read(pointer, length)),
}};
WebAssembly.instantiate(bytes, imports).then(result => {
  instance = result.instance;
  function tick() {
    let more;
    try { more = instance.exports.bench_advance(); }
    catch (error) { line("TRAPPED: " + error); line("BENCH DONE"); return; }
    if (more) { setTimeout(tick, 0); } else { line("BENCH DONE"); }
  }
  tick();
});
</script>
"#;
