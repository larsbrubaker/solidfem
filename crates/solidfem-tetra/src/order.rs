//! The deterministic spatially coherent order points are inserted in: rounds of doubling size
//! (a biased randomized insertion order, with the "random" drawn from a fixed hash of the point's
//! index so two runs agree), each round sorted along a 3D Hilbert curve. Randomized rounds keep
//! cavities small on structured input such as grids and sorted files, where plain curve order
//! produces long thin intermediate triangulations; the curve keeps successive points close, so
//! the walk in `core.rs` is short. Used by `triangulator.rs` when building. The order decides
//! performance and the numbering of the output's tetrahedra only: which tetrahedra exist is set
//! by the points and their indices through the tie rule in `predicates.rs`.

/// Bits of Hilbert resolution per axis; 3 x 21 = 63 fits a `u64` key.
const HILBERT_BITS: u32 = 21;

/// A fixed odd seed so a round assignment is the same on every run and platform.
const ROUND_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bit mixer (the SplitMix64 finalizer) turning an index into a well-spread value.
fn mix(index: u64) -> u64 {
    let mut z = index.wrapping_add(ROUND_SEED);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The round a point falls in: 0 for half of the points, 1 for a quarter, and so on. Larger
/// rounds are inserted first, so the set grows by roughly doubling.
fn round_of(index: u32) -> u32 {
    mix(u64::from(index)).trailing_zeros()
}

/// The position along a Hilbert curve of cell `axes` in a `2^bits` grid.
fn hilbert_key(mut axes: [u32; 3], bits: u32) -> u64 {
    let top = 1u32 << (bits - 1);
    // Undo the rotations the curve would have applied at each level, from the top bit down, so
    // the remaining bits read as a Gray code along the curve.
    let mut q = top;
    while q > 1 {
        let p = q - 1;
        for i in 0..3 {
            if axes[i] & q != 0 {
                axes[0] ^= p;
            } else {
                let t = (axes[0] ^ axes[i]) & p;
                axes[0] ^= t;
                axes[i] ^= t;
            }
        }
        q >>= 1;
    }
    // Gray code to binary.
    axes[1] ^= axes[0];
    axes[2] ^= axes[1];
    let mut t = 0;
    let mut q = top;
    while q > 1 {
        if axes[2] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for axis in &mut axes {
        *axis ^= t;
    }
    // Interleave the transposed axes, most significant bit first.
    let mut key = 0u64;
    for bit in (0..bits).rev() {
        for axis in axes {
            key = (key << 1) | u64::from((axis >> bit) & 1);
        }
    }
    key
}

/// The insertion order of `points`: every index exactly once, rarest round first, along the
/// Hilbert curve within a round, ties broken by index so the sort is total.
pub(crate) fn insertion_order(points: &[[f64; 3]]) -> Vec<u32> {
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    }
    // One scale for all axes keeps Hilbert cells cubic, so a thin plate keeps its in-plane
    // resolution.
    let extent = (0..3)
        .map(|axis| high[axis] - low[axis])
        .fold(0.0, f64::max);
    let last_cell = (1u32 << HILBERT_BITS) - 1;
    let scale = if extent > 0.0 {
        f64::from(last_cell) / extent
    } else {
        0.0
    };
    let mut keyed: Vec<(u32, u64, u32)> = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let mut axes = [0u32; 3];
            for axis in 0..3 {
                // The product is at most `last_cell` plus rounding; `as` saturates.
                axes[axis] = (((point[axis] - low[axis]) * scale) as u32).min(last_cell);
            }
            // `index` fits u32: the caller refuses larger inputs before ordering.
            let index = index as u32;
            (
                u32::MAX - round_of(index),
                hilbert_key(axes, HILBERT_BITS),
                index,
            )
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, _, index)| index).collect()
}
