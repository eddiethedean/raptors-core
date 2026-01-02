//! Cache-friendly algorithm utilities
//!
//! This module provides utilities for optimizing memory access patterns
//! to improve cache locality.

/// Cache line size in bytes (typically 64 bytes on modern CPUs)
pub const CACHE_LINE_SIZE: usize = 64;

/// L1 cache size (typical value, may vary by CPU)
pub const L1_CACHE_SIZE: usize = 32 * 1024; // 32 KB

/// L2 cache size (typical value, may vary by CPU)
pub const L2_CACHE_SIZE: usize = 256 * 1024; // 256 KB

/// Calculate optimal block size for cache-friendly operations
///
/// Returns a block size that fits in cache while maintaining good performance.
pub fn optimal_block_size(element_size: usize) -> usize {
    // Use a fraction of L2 cache to allow for multiple blocks
    let cache_budget = L2_CACHE_SIZE / 4; // Use 1/4 of L2 cache
    
    // Calculate how many elements fit in the cache budget
    let elements_per_block = cache_budget / element_size;
    
    // Round down to a power of 2 for better performance
    elements_per_block.next_power_of_two() / 2
}

/// Align address to cache line boundary
pub fn align_to_cache_line(addr: usize) -> usize {
    (addr + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// Check if two addresses are on the same cache line
pub fn same_cache_line(addr1: usize, addr2: usize) -> bool {
    (addr1 >> 6) == (addr2 >> 6) // Assuming 64-byte cache lines (2^6)
}

/// Prefetch hint for reading data
///
/// # Safety
/// The caller must ensure that `addr` is valid and points to memory that can be safely prefetched.
/// Prefetching invalid addresses may cause undefined behavior.
#[inline(always)]
pub unsafe fn prefetch_read(addr: *const u8) {
    // Use compiler intrinsic if available, otherwise no-op
    #[cfg(target_arch = "x86_64")]
    {
        core::arch::x86_64::_mm_prefetch(addr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    // Note: ARM prefetch intrinsics are unstable, so we skip them for now
    #[cfg(target_arch = "aarch64")]
    {
        // No-op for ARM - can be enabled when stable
        let _ = addr;
    }
}

/// Prefetch hint for writing data
///
/// # Safety
/// The caller must ensure that `addr` is valid and points to memory that can be safely prefetched.
/// Prefetching invalid addresses may cause undefined behavior.
#[inline(always)]
pub unsafe fn prefetch_write(addr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        core::arch::x86_64::_mm_prefetch(addr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // No-op for ARM - can be enabled when stable
        let _ = addr;
    }
}

