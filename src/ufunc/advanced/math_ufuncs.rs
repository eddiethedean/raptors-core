//! Advanced mathematical ufunc constructors
//!
//! This module provides constructors for advanced mathematical ufuncs

use crate::ufunc::{Ufunc, UnaryLoopFunction};
use crate::ufunc::advanced::math_loops::*;
use crate::types::NpyType;

/// Create sin ufunc
pub fn create_sin_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("sin".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], sin_loop_double);
    ufunc.register_unary_loop(vec![NpyType::Float], sin_loop_float);
    ufunc
}

/// Create cos ufunc
pub fn create_cos_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("cos".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], cos_loop_double);
    ufunc
}

/// Create tan ufunc
pub fn create_tan_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("tan".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], tan_loop_double);
    ufunc
}

/// Create asin ufunc
pub fn create_asin_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("asin".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], asin_loop_double);
    ufunc
}

/// Create acos ufunc
pub fn create_acos_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("acos".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], acos_loop_double);
    ufunc
}

/// Create atan ufunc
pub fn create_atan_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("atan".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], atan_loop_double);
    ufunc
}

/// Create sinh ufunc
pub fn create_sinh_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("sinh".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], sinh_loop_double);
    ufunc
}

/// Create cosh ufunc
pub fn create_cosh_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("cosh".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], cosh_loop_double);
    ufunc
}

/// Create tanh ufunc
pub fn create_tanh_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("tanh".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], tanh_loop_double);
    ufunc
}

/// Create exp ufunc
pub fn create_exp_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("exp".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], exp_loop_double);
    ufunc.register_unary_loop(vec![NpyType::Float], exp_loop_float);
    ufunc
}

/// Create log (natural logarithm) ufunc
pub fn create_log_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("log".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], log_loop_double);
    ufunc.register_unary_loop(vec![NpyType::Float], log_loop_float);
    ufunc
}

/// Create log10 ufunc
pub fn create_log10_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("log10".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], log10_loop_double);
    ufunc
}

/// Create log2 ufunc
pub fn create_log2_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("log2".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], log2_loop_double);
    ufunc
}

/// Create sqrt ufunc
pub fn create_sqrt_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("sqrt".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], sqrt_loop_double);
    ufunc.register_unary_loop(vec![NpyType::Float], sqrt_loop_float);
    ufunc
}

/// Create abs (absolute value) ufunc
pub fn create_abs_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("abs".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], abs_loop_double);
    ufunc.register_unary_loop(vec![NpyType::Float], abs_loop_float);
    ufunc
}

/// Create sign ufunc
pub fn create_sign_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("sign".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], sign_loop_double);
    ufunc
}

/// Create floor ufunc
pub fn create_floor_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("floor".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], floor_loop_double);
    ufunc
}

/// Create ceil ufunc
pub fn create_ceil_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("ceil".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], ceil_loop_double);
    ufunc
}

/// Create round ufunc
pub fn create_round_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("round".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], round_loop_double);
    ufunc
}

/// Create trunc ufunc
pub fn create_trunc_ufunc() -> Ufunc {
    let mut ufunc = Ufunc::new("trunc".to_string(), 1, 1);
    ufunc.register_unary_loop(vec![NpyType::Double], trunc_loop_double);
    ufunc
}

