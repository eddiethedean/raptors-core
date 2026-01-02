//! String operations
//!
//! This module provides string array operations,
//! equivalent to NumPy's strfuncs.c

mod string_array;
mod concatenation;
mod comparison;
mod formatting;
mod encoding;

pub use string_array::*;
pub use concatenation::*;
pub use comparison::*;
pub use formatting::*;
pub use encoding::*;

