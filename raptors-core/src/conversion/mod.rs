//! Type conversion module
//!
//! This module provides type conversion functionality,
//! equivalent to NumPy's convert_datatype.c and convert.c

mod promotion;
mod casting;

pub use promotion::*;
pub use casting::*;

