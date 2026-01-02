//! Tests for array operations (arithmetic and comparison)

#[cfg(test)]
mod tests {
    use raptors_core::array::Array;
    use raptors_core::operations::*;
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_add_arrays() {
        // Add two arrays
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill a1 with [1.0, 2.0, 3.0]
        unsafe {
            let ptr = a1.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }

        // Fill a2 with [4.0, 5.0, 6.0]
        unsafe {
            let ptr = a2.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 4) as f64;
            }
        }

        let result = add(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);

        // Check result: [5.0, 7.0, 9.0]
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert!((*result_ptr.add(0) - 5.0).abs() < 1e-10);
            assert!((*result_ptr.add(1) - 7.0).abs() < 1e-10);
            assert!((*result_ptr.add(2) - 9.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_subtract_arrays() {
        // Subtract two arrays
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill a1 with [5.0, 7.0, 9.0]
        unsafe {
            let ptr = a1.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 5.0;
            *ptr.add(1) = 7.0;
            *ptr.add(2) = 9.0;
        }

        // Fill a2 with [1.0, 2.0, 3.0]
        unsafe {
            let ptr = a2.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }

        let result = subtract(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);

        // Check result: [4.0, 5.0, 6.0]
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert!((*result_ptr.add(0) - 4.0).abs() < 1e-10);
            assert!((*result_ptr.add(1) - 5.0).abs() < 1e-10);
            assert!((*result_ptr.add(2) - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multiply_arrays() {
        // Multiply two arrays
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill a1 with [2.0, 3.0, 4.0]
        unsafe {
            let ptr = a1.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 2) as f64;
            }
        }

        // Fill a2 with [5.0, 6.0, 7.0]
        unsafe {
            let ptr = a2.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 5) as f64;
            }
        }

        let result = multiply(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);

        // Check result: [10.0, 18.0, 28.0]
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert!((*result_ptr.add(0) - 10.0).abs() < 1e-10);
            assert!((*result_ptr.add(1) - 18.0).abs() < 1e-10);
            assert!((*result_ptr.add(2) - 28.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_divide_arrays() {
        // Divide two arrays
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill a1 with [10.0, 18.0, 28.0]
        unsafe {
            let ptr = a1.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 18.0;
            *ptr.add(2) = 28.0;
        }

        // Fill a2 with [2.0, 3.0, 4.0]
        unsafe {
            let ptr = a2.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 2) as f64;
            }
        }

        let result = divide(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);

        // Check result: [5.0, 6.0, 7.0]
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert!((*result_ptr.add(0) - 5.0).abs() < 1e-10);
            assert!((*result_ptr.add(1) - 6.0).abs() < 1e-10);
            assert!((*result_ptr.add(2) - 7.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_equal_arrays() {
        // Equal comparison
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill both with [1.0, 2.0, 3.0]
        unsafe {
            let ptr1 = a1.data_ptr_mut() as *mut f64;
            let ptr2 = a2.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr1.add(i) = (i + 1) as f64;
                *ptr2.add(i) = (i + 1) as f64;
            }
        }

        let result = equal(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);
        assert_eq!(result.dtype().type_(), NpyType::Bool);

        // Check result: all true
        unsafe {
            let result_ptr = result.data_ptr() as *const bool;
            assert!(*result_ptr.add(0));
            assert!(*result_ptr.add(1));
            assert!(*result_ptr.add(2));
        }
    }

    #[test]
    fn test_not_equal_arrays() {
        // Not equal comparison
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill a1 with [1.0, 2.0, 3.0]
        unsafe {
            let ptr = a1.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }

        // Fill a2 with [1.0, 5.0, 3.0] (different middle element)
        unsafe {
            let ptr = a2.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 5.0;
            *ptr.add(2) = 3.0;
        }

        let result = not_equal(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);
        assert_eq!(result.dtype().type_(), NpyType::Bool);

        // Check result: [false, true, false]
        unsafe {
            let result_ptr = result.data_ptr() as *const bool;
            assert!(!*result_ptr.add(0));
            assert!(*result_ptr.add(1));
            assert!(!*result_ptr.add(2));
        }
    }

    #[test]
    fn test_less_arrays() {
        // Less than comparison
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut a1 = Array::new(shape.clone(), dtype.clone()).unwrap();
        let mut a2 = Array::new(shape.clone(), dtype.clone()).unwrap();

        // Fill a1 with [1.0, 2.0, 3.0]
        unsafe {
            let ptr = a1.data_ptr_mut() as *mut f64;
            for i in 0..3 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }

        // Fill a2 with [2.0, 1.0, 4.0]
        unsafe {
            let ptr = a2.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 2.0;
            *ptr.add(1) = 1.0;
            *ptr.add(2) = 4.0;
        }

        let result = less(&a1, &a2).unwrap();
        assert_eq!(result.shape(), &shape);
        assert_eq!(result.dtype().type_(), NpyType::Bool);

        // Check result: [true, false, true]
        unsafe {
            let result_ptr = result.data_ptr() as *const bool;
            assert!(*result_ptr.add(0));
            assert!(!*result_ptr.add(1));
            assert!(*result_ptr.add(2));
        }
    }
}

