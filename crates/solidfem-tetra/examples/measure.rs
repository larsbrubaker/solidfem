//! Measures the triangulator in release: time and tetrahedra per point on random and thin-plate
//! inputs, and the memory it holds, against the near-linear behaviour a Delaunay insertion
//! expects. Run with `cargo run --release -p solidfem-tetra --example measure`.

use std::time::Instant;

use solidfem_tetra::Triangulator;

/// A small deterministic generator (SplitMix64) so each run measures the same points.
struct Rng(u64);

impl Rng {
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn random(count: usize) -> Vec<[f64; 3]> {
    let mut rng = Rng(1);
    (0..count)
        .map(|_| [rng.next_unit(), rng.next_unit(), rng.next_unit()])
        .collect()
}

/// A plate 100 x 100 x 0.5: flat in `z`, points everywhere through its thickness.
fn thin_plate(count: usize) -> Vec<[f64; 3]> {
    let mut rng = Rng(2);
    (0..count)
        .map(|_| {
            [
                rng.next_unit() * 100.0,
                rng.next_unit() * 100.0,
                rng.next_unit() * 0.5,
            ]
        })
        .collect()
}

fn measure(name: &str, points: Vec<[f64; 3]>) {
    let n = points.len();
    let start = Instant::now();
    let mut triangulator = Triangulator::new(points).expect("the input is valid");
    let setup = start.elapsed().as_secs_f64();
    triangulator
        .step(usize::MAX, &mut || false)
        .expect("every point inserts");
    let total = start.elapsed().as_secs_f64();
    let memory = triangulator.memory_bytes();
    let result = triangulator.result().expect("the run finished");
    println!(
        "{name:>16}: n={n:>8} tets={tets:>9} tets/pt={per:.3} setup {setup:.3}s total {total:.3}s \
         ({us:.3} us/pt) memory {mb:.1} MiB ({bytes:.0} B/pt)",
        tets = result.tets.len(),
        per = result.tets.len() as f64 / n as f64,
        us = total * 1e6 / n as f64,
        mb = memory as f64 / (1024.0 * 1024.0),
        bytes = memory as f64 / n as f64,
    );
}

fn main() {
    measure("random 10k", random(10_000));
    measure("random 100k", random(100_000));
    measure("random 1M", random(1_000_000));
    measure("thin plate 100k", thin_plate(100_000));
}
