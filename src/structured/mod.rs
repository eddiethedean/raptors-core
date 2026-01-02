//! Structured arrays
//!
//! This module provides structured array support,
//! allowing arrays with composite/structured dtypes

mod dtype;
mod array;
mod fields;
mod creation;

pub use dtype::*;
pub use array::*;
pub use fields::*;
pub use creation::*;

