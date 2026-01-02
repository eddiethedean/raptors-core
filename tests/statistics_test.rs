//! Tests for statistical operations

#[cfg(test)]
mod tests {
    use raptors_core::{Array, empty};
    use raptors_core::statistics::{percentile, median, mode, std, var, histogram};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_percentile() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
            *ptr.add(4) = 5.0;
        }
        
        let p50 = percentile(&array, 50.0, None).unwrap();
        
        unsafe {
            let p50_ptr = p50.data_ptr() as *const f64;
            // 50th percentile should be 3.0 (median)
            assert_eq!(*p50_ptr, 3.0);
        }
    }

    #[test]
    fn test_median() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
            *ptr.add(4) = 5.0;
        }
        
        let med = median(&array, None).unwrap();
        
        unsafe {
            let med_ptr = med.data_ptr() as *const f64;
            assert_eq!(*med_ptr, 3.0);
        }
    }

    #[test]
    fn test_mode() {
        let shape = vec![7];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 1;
            *ptr.add(1) = 2;
            *ptr.add(2) = 2;
            *ptr.add(3) = 3;
            *ptr.add(4) = 2;
            *ptr.add(5) = 4;
            *ptr.add(6) = 2;
        }
        
        let mode_arr = mode(&array, None).unwrap();
        
        unsafe {
            let mode_ptr = mode_arr.data_ptr() as *const i32;
            assert_eq!(*mode_ptr, 2); // 2 appears most frequently
        }
    }

    #[test]
    fn test_std() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..5 {
                *ptr.add(i) = (i + 1) as f64; // [1, 2, 3, 4, 5]
            }
        }
        
        let std_val = std(&array, None, 0).unwrap();
        
        unsafe {
            let std_ptr = std_val.data_ptr() as *const f64;
            // Standard deviation should be positive
            assert!(*std_ptr > 0.0);
        }
    }

    #[test]
    fn test_var() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..5 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }
        
        let var_val = var(&array, None, 0).unwrap();
        
        unsafe {
            let var_ptr = var_val.data_ptr() as *const f64;
            // Variance should be positive
            assert!(*var_ptr > 0.0);
        }
    }

    #[test]
    fn test_histogram() {
        let shape = vec![10];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..10 {
                *ptr.add(i) = i as f64;
            }
        }
        
        let (counts, edges) = histogram(&array, 5).unwrap();
        
        assert_eq!(counts.size(), 5);
        assert_eq!(edges.size(), 6);
        
        unsafe {
            let counts_ptr = counts.data_ptr() as *const i64;
            // Each bin should have 2 elements
            assert_eq!(*counts_ptr.add(0), 2);
        }
    }

    #[test]
    fn test_percentile_empty() {
        let shape = vec![0];
        let dtype = DType::new(NpyType::Double);
        let array = empty(shape, dtype).unwrap();
        
        let result = percentile(&array, 50.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_percentile_invalid() {
        use raptors_core::zeros;
        
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        // Invalid percentile (> 100)
        let result = percentile(&array, 150.0, None);
        assert!(result.is_err());
        
        // Invalid percentile (< 0)
        let result = percentile(&array, -10.0, None);
        assert!(result.is_err());
    }
}

