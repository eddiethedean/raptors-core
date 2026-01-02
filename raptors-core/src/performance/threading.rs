//! Threading utilities and thread pool management
//!
//! This module provides thread pool configuration and utilities for
//! NumPy-compatible parallel operations.

use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::ThreadPoolBuilder;

/// Minimum array size threshold for parallelization (default: 10,000 elements)
/// Arrays smaller than this will use sequential operations
pub const PARALLEL_THRESHOLD: usize = 10_000;

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Initialize the thread pool with the given number of threads
///
/// If not called, Rayon's default thread pool is used (number of CPU cores).
/// This function allows customization to match NumPy's threading behavior.
pub fn init_thread_pool(num_threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
    ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;
    NUM_THREADS.store(num_threads, Ordering::Relaxed);
    Ok(())
}

/// Get the number of threads to use for parallel operations
pub fn num_threads() -> usize {
    let stored = NUM_THREADS.load(Ordering::Relaxed);
    if stored > 0 {
        stored
    } else {
        // Use Rayon's default (number of CPU cores)
        rayon::current_num_threads()
    }
}

/// Determine if an array size should use parallel operations
///
/// Arrays smaller than PARALLEL_THRESHOLD use sequential operations
/// to avoid thread overhead for small arrays.
pub fn should_parallelize(size: usize) -> bool {
    size >= PARALLEL_THRESHOLD
}

/// Initialize thread pool from environment variable RAPTORS_NUM_THREADS
///
/// This is called automatically if the environment variable is set.
pub fn init_from_env() -> Result<(), rayon::ThreadPoolBuildError> {
    if let Ok(num_str) = std::env::var("RAPTORS_NUM_THREADS") {
        if let Ok(num) = num_str.parse::<usize>() {
            if num > 0 {
                return init_thread_pool(num);
            }
        }
    }
    Ok(())
}

