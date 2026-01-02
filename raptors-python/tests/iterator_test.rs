//! Rust unit tests for Python iterator bindings

use pyo3::prelude::*;

#[test]
fn test_array_iterator() {
    Python::with_gil(|py| {
        let raptors = PyModule::import(py, "raptors").unwrap();
        let arr = raptors.call_method1("ones", (vec![5], raptors.getattr("float64").unwrap())).unwrap();
        
        // Create iterator
        let iter = arr.call_method0("__iter__").unwrap();
        
        // Iterate and collect values
        let mut count = 0;
        loop {
            let next_result = iter.call_method0("__next__");
            match next_result {
                Ok(value) => {
                    let val: f64 = value.extract().unwrap();
                    assert_eq!(val, 1.0);
                    count += 1;
                }
                Err(_) => break, // StopIteration
            }
        }
        
        assert_eq!(count, 5);
    });
}

