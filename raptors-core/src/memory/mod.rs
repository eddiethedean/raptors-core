//! Memory management module
//!
//! This module provides memory allocation and management functionality,
//! equivalent to NumPy's alloc.c and memory.c

mod alloc;

pub use alloc::{allocate_aligned, deallocate_aligned, reallocate_aligned, 
                simd_alignment, allocate_simd_aligned, verify_alignment, verify_simd_alignment};

