//! Arithmetic ufunc implementations
//!
//! This module provides arithmetic ufunc implementations

use crate::ufunc::{Ufunc, LoopFunction};
use crate::ufunc::loops::{
    add_loop_double, subtract_loop_double, multiply_loop_double, divide_loop_double,
    add_loop_float, add_loop_int,
};
use crate::types::NpyType;

/// Create add ufunc
pub fn create_add_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("add".to_string(), 2, 1);
    
    // Register loops for common types
    ufunc.register_loop(vec![NpyType::Double, NpyType::Double], add_loop_double);
    ufunc.register_loop(vec![NpyType::Float, NpyType::Float], add_loop_float);
    ufunc.register_loop(vec![NpyType::Int, NpyType::Int], add_loop_int);
    
    ufunc
}

/// Create subtract ufunc
pub fn create_subtract_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("subtract".to_string(), 2, 1);
    
    ufunc.register_loop(vec![NpyType::Double, NpyType::Double], subtract_loop_double);
    
    ufunc
}

/// Create multiply ufunc
pub fn create_multiply_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("multiply".to_string(), 2, 1);
    
    ufunc.register_loop(vec![NpyType::Double, NpyType::Double], multiply_loop_double);
    
    ufunc
}

/// Create divide ufunc
pub fn create_divide_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("divide".to_string(), 2, 1);
    
    ufunc.register_loop(vec![NpyType::Double, NpyType::Double], divide_loop_double);
    
    ufunc
}

