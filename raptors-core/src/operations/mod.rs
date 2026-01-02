//! Array operations module
//!
//! This module provides high-level array operations,
//! built on top of ufuncs

mod arithmetic;
mod comparison;

pub use arithmetic::*;
pub use comparison::*;

