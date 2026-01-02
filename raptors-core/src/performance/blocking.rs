//! Blocked algorithm implementations
//!
//! This module provides blocked (tiled) algorithms that process arrays
//! in cache-sized blocks to improve memory access patterns.

use crate::performance::cache::optimal_block_size;

/// Block iterator for cache-friendly array processing
///
/// This iterator splits a range into blocks of optimal size for cache efficiency.
pub struct BlockIterator {
    #[allow(dead_code)] // Reserved for future use
    start: usize,
    end: usize,
    block_size: usize,
    current: usize,
}

impl BlockIterator {
    /// Create a new block iterator
    pub fn new(start: usize, end: usize, element_size: usize) -> Self {
        let block_size = optimal_block_size(element_size).max(1);
        BlockIterator {
            start,
            end,
            block_size,
            current: start,
        }
    }
    
    /// Create with explicit block size
    pub fn with_block_size(start: usize, end: usize, block_size: usize) -> Self {
        BlockIterator {
            start,
            end,
            block_size: block_size.max(1),
            current: start,
        }
    }
}

impl Iterator for BlockIterator {
    type Item = (usize, usize); // (block_start, block_end)
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }
        
        let block_start = self.current;
        let block_end = (self.current + self.block_size).min(self.end);
        self.current = block_end;
        
        Some((block_start, block_end))
    }
}

/// Process a 1D array in blocks for cache efficiency
pub fn process_blocks_1d<F>(
    size: usize,
    element_size: usize,
    mut f: F,
) where
    F: FnMut(usize, usize), // f(block_start, block_end)
{
    for (block_start, block_end) in BlockIterator::new(0, size, element_size) {
        f(block_start, block_end);
    }
}

/// Process a 2D array in blocks for cache efficiency
pub fn process_blocks_2d<F>(
    rows: usize,
    cols: usize,
    element_size: usize,
    mut f: F,
) where
    F: FnMut(usize, usize, usize, usize), // f(row_start, row_end, col_start, col_end)
{
    let row_block_size = optimal_block_size(element_size * cols).max(1);
    let col_block_size = optimal_block_size(element_size).max(1);
    
    for row_start in (0..rows).step_by(row_block_size) {
        let row_end = (row_start + row_block_size).min(rows);
        for col_start in (0..cols).step_by(col_block_size) {
            let col_end = (col_start + col_block_size).min(cols);
            f(row_start, row_end, col_start, col_end);
        }
    }
}

