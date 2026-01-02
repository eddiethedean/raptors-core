//! Rust unit tests for Python ufunc bindings

use pyo3::prelude::*;

#[test]
fn test_arithmetic_ufuncs() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let float64 = raptors.getattr("float64").unwrap();
        
        let a = raptors.call_method1("ones", (vec![3, 3], float64)).unwrap();
        let b = raptors.call_method1("ones", (vec![3, 3], float64)).unwrap();
        
        // Test add
        let add_func = raptors.getattr("add").unwrap();
        let result = add_func.call1((a, b)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 3]);
        
        // Test subtract
        let sub_func = raptors.getattr("subtract").unwrap();
        let result = sub_func.call1((a, b)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 3]);
    });
}

#[test]
fn test_math_ufuncs() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let float64 = raptors.getattr("float64").unwrap();
        let arr = raptors.call_method1("ones", (vec![3, 3], float64)).unwrap();
        
        // Test sin
        let sin_func = raptors.getattr("sin").unwrap();
        let result = sin_func.call1((arr,)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 3]);
        
        // Test cos
        let cos_func = raptors.getattr("cos").unwrap();
        let result = cos_func.call1((arr,)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 3]);
    });
}

#[test]
fn test_reduction_ufuncs() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let float64 = raptors.getattr("float64").unwrap();
        let arr = raptors.call_method1("ones", (vec![3, 4], float64)).unwrap();
        
        // Test sum
        let sum_func = raptors.getattr("sum").unwrap();
        let result = sum_func.call1((arr,)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        // Sum with no axis should return scalar (empty shape or shape [1])
        assert!(shape.is_empty() || shape == vec![1]);
    });
}

