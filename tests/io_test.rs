//! Tests for I/O operations

#[cfg(test)]
mod tests {
    use raptors_core::zeros;
    use raptors_core::io::{save_npy, load_npy};
    use raptors_core::types::{DType, NpyType};
    use std::fs;

    #[test]
    fn test_save_load_roundtrip() {
        let dtype = DType::new(NpyType::Double);
        let shape = vec![3, 2];
        let mut array = zeros(shape, dtype).unwrap();
        
        // Set some values
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
            *ptr.add(4) = 5.0;
            *ptr.add(5) = 6.0;
        }
        
        let test_path = "/tmp/test_npy_roundtrip.npy";
        
        // Save
        save_npy(test_path, &array).unwrap();
        
        // Load
        let loaded = load_npy(test_path).unwrap();
        
        // Verify shape and dtype
        assert_eq!(loaded.shape(), array.shape());
        assert_eq!(loaded.dtype().type_(), array.dtype().type_());
        assert_eq!(loaded.size(), array.size());
        
        // Verify data
        unsafe {
            let original_ptr = array.data_ptr() as *const f64;
            let loaded_ptr = loaded.data_ptr() as *const f64;
            for i in 0..array.size() {
                assert_eq!(*original_ptr.add(i), *loaded_ptr.add(i));
            }
        }
        
        // Cleanup
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_save_load_1d() {
        let dtype = DType::new(NpyType::Double);
        let shape = vec![5];
        let mut array = zeros(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..5 {
                *ptr.add(i) = i as f64;
            }
        }
        
        let test_path = "/tmp/test_npy_1d.npy";
        save_npy(test_path, &array).unwrap();
        let loaded = load_npy(test_path).unwrap();
        
        assert_eq!(loaded.shape(), &[5]);
        unsafe {
            let ptr = loaded.data_ptr() as *const f64;
            for i in 0..5 {
                assert_eq!(*ptr.add(i), i as f64);
            }
        }
        
        let _ = fs::remove_file(test_path);
    }
}

