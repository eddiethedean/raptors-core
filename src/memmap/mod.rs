//! Memory-mapped arrays
//!
//! This module provides memory-mapped array support,
//! allowing arrays backed by memory-mapped files

mod mmap_array;
mod creation;
mod io;

pub use mmap_array::*;
pub use creation::*;
pub use io::*;

