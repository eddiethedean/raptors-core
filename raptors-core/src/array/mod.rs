//! Core array implementation
//!
//! This module provides the core array structure and operations,
//! equivalent to NumPy's `arrayobject.c` and related files.

mod arrayobject;
mod creation;
mod flags;
mod builder;
mod iter_ops;
mod subclassing;

pub use arrayobject::*;
pub use creation::*;
pub use flags::*;
pub use builder::{ArrayBuilder, MemoryOrder};
pub use iter_ops::ArrayIterOps;
pub use subclassing::*;

