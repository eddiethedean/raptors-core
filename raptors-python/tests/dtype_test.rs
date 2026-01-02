//! Rust unit tests for Python DType bindings

use pyo3::prelude::*;

#[test]
fn test_dtype_creation() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        
        // Test creating dtype from string
        let dtype_class = raptors.getattr("DType").unwrap();
        let dtype = dtype_class.call1(("float64",)).unwrap();
        
        let name: String = dtype.getattr("name").unwrap().extract().unwrap();
        assert_eq!(name, "float64");
        
        let itemsize: usize = dtype.getattr("itemsize").unwrap().extract().unwrap();
        assert_eq!(itemsize, 8);
    });
}

#[test]
fn test_dtype_constants() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        
        // Test float64 constant
        let float64 = raptors.getattr("float64").unwrap();
        let name: String = float64.getattr("name").unwrap().extract().unwrap();
        assert_eq!(name, "float64");
        
        // Test int64 constant
        let int64 = raptors.getattr("int64").unwrap();
        let name: String = int64.getattr("name").unwrap().extract().unwrap();
        assert_eq!(name, "int64");
    });
}

#[test]
fn test_dtype_properties() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let float32 = raptors.getattr("float32").unwrap();
        
        let name: String = float32.getattr("name").unwrap().extract().unwrap();
        assert_eq!(name, "float32");
        
        let itemsize: usize = float32.getattr("itemsize").unwrap().extract().unwrap();
        assert_eq!(itemsize, 4);
        
        let kind: String = float32.getattr("kind").unwrap().extract().unwrap();
        assert_eq!(kind, "f");
    });
}

