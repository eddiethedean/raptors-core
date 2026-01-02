//! Optimized ufunc loop implementations
//!
//! This module provides specialized optimized loop variants for common cases,
//! including loop unrolling for small fixed sizes and architecture-specific hints.

/// Unroll factor for small loops (process 4 elements at a time when beneficial)
#[allow(dead_code)] // Reserved for future use
const UNROLL_FACTOR: usize = 4;

/// Add loop for double precision with loop unrolling for small arrays
///
/// This version processes multiple elements per iteration to reduce loop overhead.
/// Best for small to medium arrays where loop overhead matters.
///
/// # Safety
/// Caller must ensure all pointers are valid, aligned, and contiguous.
#[allow(dead_code)] // Reserved for future use
pub unsafe fn add_loop_double_unrolled(
    in1: *const f64,
    in2: *const f64,
    out: *mut f64,
    count: usize,
) {
    let mut i = 0;
    
    // Process UNROLL_FACTOR elements at a time
    let unrolled_end = count - (count % UNROLL_FACTOR);
    while i < unrolled_end {
        out.add(i).write(*in1.add(i) + *in2.add(i));
        out.add(i + 1).write(*in1.add(i + 1) + *in2.add(i + 1));
        out.add(i + 2).write(*in1.add(i + 2) + *in2.add(i + 2));
        out.add(i + 3).write(*in1.add(i + 3) + *in2.add(i + 3));
        i += UNROLL_FACTOR;
    }
    
    // Handle remaining elements
    while i < count {
        out.add(i).write(*in1.add(i) + *in2.add(i));
        i += 1;
    }
}

/// Add loop for float precision with loop unrolling
///
/// # Safety
/// Caller must ensure all pointers are valid, aligned, and contiguous.
#[allow(dead_code)] // Reserved for future use
pub unsafe fn add_loop_float_unrolled(
    in1: *const f32,
    in2: *const f32,
    out: *mut f32,
    count: usize,
) {
    let mut i = 0;
    
    let unrolled_end = count - (count % UNROLL_FACTOR);
    while i < unrolled_end {
        out.add(i).write(*in1.add(i) + *in2.add(i));
        out.add(i + 1).write(*in1.add(i + 1) + *in2.add(i + 1));
        out.add(i + 2).write(*in1.add(i + 2) + *in2.add(i + 2));
        out.add(i + 3).write(*in1.add(i + 3) + *in2.add(i + 3));
        i += UNROLL_FACTOR;
    }
    
    while i < count {
        out.add(i).write(*in1.add(i) + *in2.add(i));
        i += 1;
    }
}

/// Multiply loop for double precision with loop unrolling
///
/// # Safety
/// Caller must ensure all pointers are valid, aligned, and contiguous.
#[allow(dead_code)] // Reserved for future use
pub unsafe fn multiply_loop_double_unrolled(
    in1: *const f64,
    in2: *const f64,
    out: *mut f64,
    count: usize,
) {
    let mut i = 0;
    
    let unrolled_end = count - (count % UNROLL_FACTOR);
    while i < unrolled_end {
        out.add(i).write(*in1.add(i) * *in2.add(i));
        out.add(i + 1).write(*in1.add(i + 1) * *in2.add(i + 1));
        out.add(i + 2).write(*in1.add(i + 2) * *in2.add(i + 2));
        out.add(i + 3).write(*in1.add(i + 3) * *in2.add(i + 3));
        i += UNROLL_FACTOR;
    }
    
    while i < count {
        out.add(i).write(*in1.add(i) * *in2.add(i));
        i += 1;
    }
}

/// Threshold for using unrolled loops (array size must be >= this)
/// For very small arrays, the unrolled version may not be beneficial due to overhead
#[allow(dead_code)] // Reserved for future use
const UNROLL_THRESHOLD: usize = 16;

/// Check if unrolling should be used for a given array size
#[inline(always)]
#[allow(dead_code)] // Reserved for future use
pub fn should_unroll(count: usize) -> bool {
    count >= UNROLL_THRESHOLD && count.is_multiple_of(UNROLL_FACTOR)
}

