//! A small persistent pool of worker threads for the solver's row-by-row products, built from the
//! standard library alone. A [`WorkerPool`] runs one closure per chunk of rows and waits for all
//! of them, so a product borrows its matrix and vectors like an ordinary call.
//!
//! **Why the answer cannot depend on the thread count.** Every value written here is a pure
//! function of its own row: the row's stored blocks and the input vectors, summed in column
//! order. Which thread computes a row, and which rows share a chunk, changes no operation and no
//! order of operations on any value, so the bits are those of a serial run. Nothing here sums
//! across rows; the sums that do (the conjugate gradient dot products, the transpose product) stay
//! serial in their callers, in index order.
//!
//! **Why a pool and not a scoped spawn per product.** Measured on a Windows machine with 20 logical
//! processors, under load: an empty pool region costs 76 us, against 255 us to start and join
//! seven scoped threads. A V-cycle runs about fourteen regions on each large level, so the pool
//! spends about 1 ms a cycle there where scoped threads would spend about 3.5 ms. A hierarchy that
//! never reaches [`MIN_PARALLEL_VALUES`] starts no thread at all.
//!
//! A target without threads (`wasm32`) gets no pool and runs every region on the calling thread,
//! through the same closure. [`for_rows`] with no pool or one chunk is a plain call.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};

/// A level or system with fewer stored values than this runs on one thread: below it, waking the
/// workers costs a noticeable share of a product.
pub(crate) const MIN_PARALLEL_VALUES: usize = 100_000;

/// The most threads the default asks for.
const MAX_DEFAULT_THREADS: usize = 8;

/// The most threads a pool ever starts, whatever was asked for: `usize::MAX` is a natural way to
/// say "no limit", and neither it nor a mistyped count may exhaust memory or flood the machine. It
/// is a fixed number and not the machine's processor count, so a count below it means the same
/// thing on every machine.
const MAX_THREADS: usize = 64;

/// The thread count the multigrid options start with: half the logical processors, which on a
/// machine with hyperthreads is about one per core and leaves the rest of the machine its share,
/// and one where threads do not exist.
pub(crate) fn default_threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    let logical = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    (logical / 2).clamp(1, MAX_DEFAULT_THREADS)
}

/// One region's closure with its lifetime erased; see [`WorkerPool::run`].
#[derive(Clone, Copy)]
struct Task(*const (dyn Fn(usize) + Sync + 'static));

// SAFETY: the pointer is only a way to hand a `Sync` closure to the workers: `WorkerPool::run`
// keeps the closure alive and blocks until every worker has finished with it.
unsafe impl Send for Task {}

struct State {
    /// Counts regions, so a worker can tell a new one from the one it just finished.
    region: u64,
    task: Option<Task>,
    /// Workers yet to finish the current region.
    running: usize,
    panicked: Option<Box<dyn std::any::Any + Send>>,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    finished: Condvar,
}

impl Shared {
    /// No code here panics while holding the lock, so a poisoned lock's state is still valid.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Worker threads parked between parallel regions.
pub(crate) struct WorkerPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    /// A pool that runs regions on `threads` threads including the caller, at most
    /// [`MAX_THREADS`], or `None` when that is one thread or the target has none. A worker the
    /// system refuses to start is simply not there: the pool runs on the threads it got, and the
    /// answer does not depend on how many.
    pub(crate) fn new(threads: usize) -> Option<Self> {
        let threads = threads.min(MAX_THREADS);
        if threads <= 1 || cfg!(target_arch = "wasm32") {
            return None;
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                region: 0,
                task: None,
                running: 0,
                panicked: None,
                shutdown: false,
            }),
            start: Condvar::new(),
            finished: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(threads - 1);
        for index in 1..threads {
            let worker_shared = Arc::clone(&shared);
            let spawned = thread::Builder::new()
                .name(format!("solidfem-solve-{index}"))
                .spawn(move || worker(&worker_shared, index));
            match spawned {
                Ok(handle) => workers.push(handle),
                Err(_) => break,
            }
        }
        if workers.is_empty() {
            return None;
        }
        Some(WorkerPool { shared, workers })
    }

    /// Threads a region runs on, the caller included.
    pub(crate) fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Runs `task(0)` on the calling thread and `task(i)` on worker `i`, and returns when all
    /// have finished. A panic in any of them is raised here, after they have all stopped.
    fn run(&self, task: &(dyn Fn(usize) + Sync)) {
        // SAFETY: only the closure's lifetime is erased, and only for the duration of this call.
        // One region runs at a time, which the assertion below checks under the lock before
        // anything is published, so no other call can replace or clear this pointer while workers
        // use it. The wait below does not return until `running` reaches zero, which every worker
        // sets after its last use of the pointer, and the caller's own panic is caught so the wait
        // still happens.
        let erased: *const (dyn Fn(usize) + Sync + 'static) =
            unsafe { std::mem::transmute(task as *const (dyn Fn(usize) + Sync + '_)) };
        {
            let mut state = self.shared.lock();
            assert!(state.task.is_none(), "one parallel region at a time");
            state.region = state.region.wrapping_add(1);
            state.task = Some(Task(erased));
            state.running = self.workers.len();
        }
        self.shared.start.notify_all();
        let own = catch_unwind(AssertUnwindSafe(|| task(0)));
        let mut state = self.shared.lock();
        while state.running > 0 {
            state = self
                .shared
                .finished
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.task = None;
        let panicked = state.panicked.take();
        drop(state);
        if let Err(payload) = own {
            resume_unwind(payload);
        }
        if let Some(payload) = panicked {
            resume_unwind(payload);
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.start.notify_all();
        for handle in self.workers.drain(..) {
            // A worker catches its tasks' panics, so joining has nothing further to report.
            let _ = handle.join();
        }
    }
}

fn worker(shared: &Shared, index: usize) {
    let mut seen = 0;
    loop {
        let task = {
            let mut state = shared.lock();
            while !state.shutdown && state.region == seen {
                state = shared
                    .start
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if state.shutdown {
                return;
            }
            seen = state.region;
            state.task
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            if let Some(Task(pointer)) = task {
                // SAFETY: `run` set this pointer for the current region and does not return, so
                // the closure stays alive, until this worker has counted itself finished below.
                unsafe { (*pointer)(index) }
            }
        }));
        let mut state = shared.lock();
        if let Err(payload) = outcome {
            state.panicked = Some(payload);
        }
        state.running -= 1;
        if state.running == 0 {
            shared.finished.notify_one();
        }
    }
}

/// Splits rows into at most `chunks` contiguous ranges of about equal weight, from the rows'
/// cumulative weights (`cumulative[row]` is the weight before `row`, the last entry the total).
/// Returns the chunk boundaries, from zero to the row count; a chunk never comes out empty.
pub(crate) fn balanced_bounds(cumulative: &[usize], chunks: usize) -> Vec<usize> {
    let rows = cumulative.len().saturating_sub(1);
    let total = cumulative.last().copied().unwrap_or(0);
    let mut bounds = Vec::with_capacity(chunks + 1);
    bounds.push(0);
    for chunk in 1..chunks {
        // Cannot overflow in practice: `total` is a stored-value count, far below 2^64 / chunks.
        let target = total * chunk / chunks;
        let row = cumulative
            .partition_point(|&weight| weight < target)
            .min(rows);
        if row > bounds[bounds.len() - 1] && row < rows {
            bounds.push(row);
        }
    }
    if rows > 0 {
        bounds.push(rows);
    }
    bounds
}

/// The mutable arrays of one region as raw parts, handed to one chunk at a time.
struct Arrays<V, const N: usize>([*mut V; N]);

// SAFETY: a chunk reads and writes only its own rows of each array, and no two chunks share a
// row, so sharing the pointers between threads amounts to sending disjoint `&mut [V]`, which
// needs `V: Send`.
unsafe impl<V: Send, const N: usize> Sync for Arrays<V, N> {}

/// Runs `work` once per chunk of `bounds`, passing the chunk's rows and those rows of each array
/// (`stride` values per row). Every array must hold `stride` values for every row. `work` must
/// compute each row from that row and from data it only reads.
pub(crate) fn for_rows<V, const N: usize, F>(
    pool: Option<&WorkerPool>,
    bounds: &[usize],
    stride: usize,
    arrays: [&mut [V]; N],
    work: &F,
) where
    V: Send,
    F: Fn(Range<usize>, [&mut [V]; N]) + Sync,
{
    let rows = bounds.last().copied().unwrap_or(0);
    let pool = match pool {
        Some(pool) if bounds.len() > 2 => pool,
        _ => return work(0..rows, arrays),
    };
    let lengths_match = arrays.iter().all(|array| array.len() == rows * stride);
    assert!(lengths_match, "every array holds stride values per row");
    assert!(bounds.len() - 1 <= pool.threads(), "one chunk per thread");
    // The chunks' disjointness below rests on this: reversed bounds would build overlapping
    // mutable slices.
    let increasing = bounds[0] == 0 && bounds.windows(2).all(|pair| pair[0] < pair[1]);
    assert!(increasing, "chunk bounds increase from zero");
    let parts = Arrays(arrays.map(|array| array.as_mut_ptr()));
    let parts = &parts;
    pool.run(&|chunk| {
        let Some(window) = bounds.get(chunk..chunk + 2) else {
            return;
        };
        let (start, end) = (window[0], window[1]);
        let slices = parts.0.map(|pointer| {
            // SAFETY: `bounds` increases from zero, as checked above, so the rows `start..end` of
            // this chunk are disjoint from every other chunk's and inside the array, whose length
            // was also checked; each chunk index runs exactly once per region, and the arrays stay
            // exclusively borrowed by this function until `run` returns.
            unsafe {
                std::slice::from_raw_parts_mut(pointer.add(start * stride), (end - start) * stride)
            }
        });
        work(start..end, slices);
    });
}
