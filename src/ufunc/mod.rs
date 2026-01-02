//! Universal functions (ufuncs) module
//!
//! This module provides universal function functionality,
//! equivalent to NumPy's umath module

mod advanced;
mod arithmetic;
mod comparison;
mod loop_exec;
mod loops;
mod optimized;
mod parallel;
mod reduction;
#[allow(clippy::module_inception)]
mod ufunc;

pub use advanced::*;
pub use arithmetic::*;
pub use comparison::*;
pub use loop_exec::{create_unary_ufunc_loop, LoopExecutionError};
pub use loops::*;
pub use parallel::{add_parallel, multiply_parallel, should_use_parallel_ufunc};
pub use reduction::{sum_along_axis, min_along_axis, max_along_axis, mean_along_axis, ReductionError};
pub use ufunc::*;

