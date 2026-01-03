//! Rust unit tests for Python Array bindings

use pyo3::prelude::*;
use pyo3::types::PyList;

#[test]
fn test_array_creation() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        
        // Test zeros
        let zeros = raptors.call_method1("zeros", (vec![3, 4], raptors.getattr("float64").unwrap())).unwrap();
        let shape: Vec<i64> = zeros.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 4]);
        
        // Test ones
        let ones = raptors.call_method1("ones", (vec![2, 3], raptors.getattr("float64").unwrap())).unwrap();
        let shape: Vec<i64> = ones.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![2, 3]);
    });
}

#[test]
fn test_array_properties() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let arr = raptors.call_method1("zeros", (vec![3, 4, 5], raptors.getattr("float64").unwrap())).unwrap();
        
        let shape: Vec<i64> = arr.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 4, 5]);
        
        let size: i64 = arr.getattr("size").unwrap().extract().unwrap();
        assert_eq!(size, 60);
        
        let ndim: usize = arr.getattr("ndim").unwrap().extract().unwrap();
        assert_eq!(ndim, 3);
        
        let dtype = arr.getattr("dtype").unwrap();
        let dtype_name: String = dtype.getattr("name").unwrap().extract().unwrap();
        assert_eq!(dtype_name, "float64");
    });
}

#[test]
fn test_array_arithmetic() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let float64 = raptors.getattr("float64").unwrap();
        
        let a = raptors.call_method1("ones", (vec![2, 2], float64.clone())).unwrap();
        let b = raptors.call_method1("ones", (vec![2, 2], float64)).unwrap();
        
        // Test addition
        let result = a.call_method1("__add__", (b.clone(),)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![2, 2]);
        
        // Test multiplication
        let result = a.call_method1("__mul__", (b,)).unwrap();
        let shape: Vec<i64> = result.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![2, 2]);
    });
}

#[test]
fn test_array_reshape() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let arr = raptors.call_method1("zeros", (vec![3, 4], raptors.getattr("float64").unwrap())).unwrap();
        
        let reshaped = arr.call_method1("reshape", (vec![12],)).unwrap();
        let shape: Vec<i64> = reshaped.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![12]);
    });
}

#[test]
fn test_array_transpose() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let arr = raptors.call_method1("zeros", (vec![3, 4], raptors.getattr("float64").unwrap())).unwrap();
        
        let transposed = arr.call_method0("transpose").unwrap();
        let shape: Vec<i64> = transposed.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![4, 3]);
    });
}

#[test]
fn test_array_copy() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let arr = raptors.call_method1("zeros", (vec![3, 4], raptors.getattr("float64").unwrap())).unwrap();
        
        let copied = arr.call_method0("copy").unwrap();
        let shape: Vec<i64> = copied.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![3, 4]);
    });
}

#[test]
fn test_array_indexing() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let arr = raptors.call_method1("ones", (vec![5], raptors.getattr("float64").unwrap())).unwrap();
        
        // Test getitem
        let value: f64 = arr.call_method1("__getitem__", (2,)).unwrap().extract().unwrap();
        assert_eq!(value, 1.0);
    });
}

