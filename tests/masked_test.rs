//! Tests for masked arrays

#[cfg(test)]
mod tests {
    use raptors_core::{Array, DType, zeros};
    use raptors_core::types::NpyType;
    use raptors_core::masked::*;

    #[test]
    fn test_masked_array_creation() {
        let dtype = DType::new(NpyType::Double);
        let data = zeros(vec![3], dtype).unwrap();
        
        let mask_dtype = DType::new(NpyType::Bool);
        let mask = zeros(vec![3], mask_dtype).unwrap();
        
        let masked = MaskedArray::new(data, mask).unwrap();
        assert_eq!(masked.size(), 3);
        assert_eq!(masked.count_masked(), 0);
        assert_eq!(masked.count_valid(), 3);
    }

    #[test]
    fn test_masked_array_with_indices() {
        let dtype = DType::new(NpyType::Double);
        let data = zeros(vec![5], dtype).unwrap();
        
        let masked = masked_array_with_indices(data, &[1, 3]).unwrap();
        assert_eq!(masked.count_masked(), 2);
        assert_eq!(masked.count_valid(), 3);
        
        assert!(masked.is_masked(1).unwrap());
        assert!(masked.is_masked(3).unwrap());
        assert!(!masked.is_masked(0).unwrap());
        assert!(!masked.is_masked(2).unwrap());
    }

    #[test]
    fn test_masked_sum() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![4], dtype).unwrap();
        
        // Set some values
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
        }
        
        // Mask element at index 1
        let masked = masked_array_with_indices(data, &[1]).unwrap();
        
        let result = masked_sum(&masked, None).unwrap();
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            assert_eq!(*ptr, 8.0); // 1.0 + 3.0 + 4.0 (element 1 is masked)
        }
    }

    #[test]
    fn test_masked_mean() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![4], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 2.0;
            *ptr.add(1) = 4.0;
            *ptr.add(2) = 6.0;
            *ptr.add(3) = 8.0;
        }
        
        // Mask element at index 1
        let masked = masked_array_with_indices(data, &[1]).unwrap();
        
        let result = masked_mean(&masked, None).unwrap();
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            assert_eq!(*ptr, (2.0 + 6.0 + 8.0) / 3.0); // Mean of 3 unmasked values
        }
    }

    #[test]
    fn test_fill_masked() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![3], dtype).unwrap();
        
        let mut masked = masked_array_with_indices(data, &[1]).unwrap();
        
        fill_masked(&mut masked, 99.0).unwrap();
        
        let data_ref = masked.data();
        unsafe {
            let ptr = data_ref.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 0.0);
            assert_eq!(*ptr.add(1), 99.0); // Filled
            assert_eq!(*ptr.add(2), 0.0);
        }
    }

    #[test]
    fn test_get_valid_values() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![4], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
        }
        
        let masked = masked_array_with_indices(data, &[1, 3]).unwrap();
        let valid = get_valid_values(&masked).unwrap();
        
        assert_eq!(valid.size(), 2);
        unsafe {
            let ptr = valid.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 1.0);
            assert_eq!(*ptr.add(1), 3.0);
        }
    }

    #[test]
    fn test_masked_multiply() {
        let dtype = DType::new(NpyType::Double);
        let mut data1 = zeros(vec![3], dtype.clone()).unwrap();
        let mut data2 = zeros(vec![3], dtype).unwrap();
        
        unsafe {
            let ptr1 = data1.data_ptr_mut() as *mut f64;
            *ptr1.add(0) = 2.0;
            *ptr1.add(1) = 3.0;
            *ptr1.add(2) = 4.0;
            
            let ptr2 = data2.data_ptr_mut() as *mut f64;
            *ptr2.add(0) = 5.0;
            *ptr2.add(1) = 6.0;
            *ptr2.add(2) = 7.0;
        }
        
        let masked1 = masked_array_with_indices(data1, &[1]).unwrap();
        let masked2 = masked_array_with_indices(data2, &[2]).unwrap();
        
        let result = masked_multiply(&masked1, &masked2).unwrap();
        let data = result.data();
        unsafe {
            let ptr = data.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 10.0); // 2.0 * 5.0
            // Elements 1 and 2 should be masked in result
        }
    }

    #[test]
    fn test_masked_subtract() {
        let dtype = DType::new(NpyType::Double);
        let mut data1 = zeros(vec![3], dtype.clone()).unwrap();
        let mut data2 = zeros(vec![3], dtype).unwrap();
        
        unsafe {
            let ptr1 = data1.data_ptr_mut() as *mut f64;
            *ptr1.add(0) = 10.0;
            *ptr1.add(1) = 20.0;
            *ptr1.add(2) = 30.0;
            
            let ptr2 = data2.data_ptr_mut() as *mut f64;
            *ptr2.add(0) = 3.0;
            *ptr2.add(1) = 5.0;
            *ptr2.add(2) = 7.0;
        }
        
        let masked1 = masked_array_with_indices(data1, &[1]).unwrap();
        let masked2 = masked_array_with_indices(data2, &[2]).unwrap();
        
        let result = masked_subtract(&masked1, &masked2).unwrap();
        let data = result.data();
        unsafe {
            let ptr = data.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 7.0); // 10.0 - 3.0
        }
    }

    #[test]
    fn test_masked_min() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![5], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 5.0;
            *ptr.add(1) = 1.0;  // Will be masked
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 2.0;
            *ptr.add(4) = 4.0;
        }
        
        let masked = masked_array_with_indices(data, &[1]).unwrap();
        let result = masked_min(&masked, None).unwrap();
        
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            assert_eq!(*ptr, 2.0); // Min of unmasked values (5, 3, 2, 4)
        }
    }

    #[test]
    fn test_masked_max() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![5], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 2.0;
            *ptr.add(1) = 10.0; // Will be masked
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
            *ptr.add(4) = 5.0;
        }
        
        let masked = masked_array_with_indices(data, &[1]).unwrap();
        let result = masked_max(&masked, None).unwrap();
        
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            assert_eq!(*ptr, 5.0); // Max of unmasked values (2, 3, 4, 5)
        }
    }

    #[test]
    fn test_masked_array_invalid() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![4], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = f64::NAN;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = f64::INFINITY;
        }
        
        let masked = masked_array_invalid(data).unwrap();
        assert_eq!(masked.count_masked(), 2); // NaN and Infinity
        assert!(masked.is_masked(1).unwrap());
        assert!(!masked.is_masked(0).unwrap());
        assert!(!masked.is_masked(2).unwrap());
        assert!(masked.is_masked(3).unwrap());
    }

    #[test]
    fn test_get_masked_values() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![4], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
        }
        
        let masked = masked_array_with_indices(data, &[1, 3]).unwrap();
        let masked_vals = get_masked_values(&masked).unwrap();
        
        assert_eq!(masked_vals.size(), 2);
        unsafe {
            let ptr = masked_vals.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 2.0);
            assert_eq!(*ptr.add(1), 4.0);
        }
    }

    #[test]
    fn test_masked_all_masked() {
        let dtype = DType::new(NpyType::Double);
        let data = zeros(vec![3], dtype).unwrap();
        
        let masked = masked_array_with_indices(data, &[0, 1, 2]).unwrap();
        assert_eq!(masked.count_masked(), 3);
        assert_eq!(masked.count_valid(), 0);
        
        // Sum of all masked should handle gracefully
        let result = masked_sum(&masked, None);
        // Should return 0.0 for sum of no valid values
        assert!(result.is_ok());
    }

    #[test]
    fn test_masked_none_masked() {
        let dtype = DType::new(NpyType::Double);
        let mut data = zeros(vec![3], dtype).unwrap();
        
        unsafe {
            let ptr = data.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
        }
        
        let masked = masked_array_with_indices(data, &[]).unwrap();
        assert_eq!(masked.count_masked(), 0);
        assert_eq!(masked.count_valid(), 3);
        
        let result = masked_sum(&masked, None).unwrap();
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            assert_eq!(*ptr, 6.0); // 1.0 + 2.0 + 3.0
        }
    }

    #[test]
    fn test_masked_invalid_mask_shape() {
        let dtype = DType::new(NpyType::Double);
        let data = zeros(vec![3], dtype).unwrap();
        
        let mask_dtype = DType::new(NpyType::Bool);
        let mask = zeros(vec![2], mask_dtype).unwrap(); // Wrong shape
        
        let result = MaskedArray::new(data, mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_masked_invalid_mask_type() {
        let dtype = DType::new(NpyType::Double);
        let data = zeros(vec![3], dtype.clone()).unwrap();
        
        let mask = zeros(vec![3], dtype).unwrap(); // Not boolean
        
        let result = MaskedArray::new(data, mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_masked_mean_all_masked() {
        let dtype = DType::new(NpyType::Double);
        let data = zeros(vec![3], dtype).unwrap();
        
        let masked = masked_array_with_indices(data, &[0, 1, 2]).unwrap();
        let result = masked_mean(&masked, None);
        assert!(result.is_err()); // Division by zero
    }
}

