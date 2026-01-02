//! Core array implementation
//!
//! This module provides the core array structure and operations,
//! equivalent to NumPy's `arrayobject.c` and related files.

mod arrayobject;
mod creation;
mod flags;

pub use arrayobject::*;
pub use creation::*;
pub use flags::*;

