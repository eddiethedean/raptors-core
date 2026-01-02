//! Comparison ufunc implementations
//!
//! This module provides comparison ufunc implementations

use crate::ufunc::{Ufunc, LoopFunction};
use crate::ufunc::loops::{equal_loop_double, less_loop_double};
use crate::types::NpyType;

/// Create equal ufunc
pub fn create_equal_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("equal".to_string(), 2, 1);
    
    ufunc.register_loop(vec![NpyType::Double, NpyType::Double], equal_loop_double);
    
    ufunc
}

/// Create less ufunc
pub fn create_less_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("less".to_string(), 2, 1);
    
    ufunc.register_loop(vec![NpyType::Double, NpyType::Double], less_loop_double);
    
    ufunc
}

/// Create not_equal ufunc (uses equal and negates)
pub fn create_not_equal_ufunc() -> Ufunc {
    // For now, simplified - would need proper implementation
    Ufunc::new("not_equal".to_string(), 2, 1)
}

