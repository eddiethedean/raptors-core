//! Tests for concatenation operations

#[cfg(test)]
mod tests {
    use raptors_core::{Array, zeros};
    use raptors_core::concatenation::{concatenate, stack, split, SplitSpec};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_concatenate_1d() {
        let dtype = DType::new(NpyType::Double);
        
        // Create arrays [1.0, 2.0] and [3.0, 4.0]
        let shape1 = vec![2];
        let mut arr1 = zeros(shape1, dtype.clone()).unwrap();
        unsafe {
            let ptr = arr1.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
        }
        
        let shape2 = vec![2];
        let mut arr2 = zeros(shape2, dtype.clone()).unwrap();
        unsafe {
            let ptr = arr2.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 3.0;
            *ptr.add(1) = 4.0;
        }
        
        let arrays = vec![&arr1, &arr2];
        let result = concatenate(&arrays, Some(0)).unwrap();
        
        assert_eq!(result.size(), 4);
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 1.0);
            assert_eq!(*ptr.add(1), 2.0);
            assert_eq!(*ptr.add(2), 3.0);
            assert_eq!(*ptr.add(3), 4.0);
        }
    }

    #[test]
    fn test_concatenate_axis_none() {
        let dtype = DType::new(NpyType::Double);
        
        let shape1 = vec![2];
        let mut arr1 = zeros(shape1, dtype.clone()).unwrap();
        unsafe {
            let ptr = arr1.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
        }
        
        let shape2 = vec![2];
        let mut arr2 = zeros(shape2, dtype.clone()).unwrap();
        unsafe {
            let ptr = arr2.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 3.0;
            *ptr.add(1) = 4.0;
        }
        
        let arrays = vec![&arr1, &arr2];
        let result = concatenate(&arrays, None).unwrap();
        
        assert_eq!(result.size(), 4);
    }

    #[test]
    fn test_stack() {
        let dtype = DType::new(NpyType::Double);
        
        let shape = vec![2];
        let mut arr1 = zeros(shape.clone(), dtype.clone()).unwrap();
        unsafe {
            let ptr = arr1.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
        }
        
        let mut arr2 = zeros(shape.clone(), dtype.clone()).unwrap();
        unsafe {
            let ptr = arr2.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 3.0;
            *ptr.add(1) = 4.0;
        }
        
        let arrays = vec![&arr1, &arr2];
        let result = stack(&arrays, 0).unwrap();
        
        assert_eq!(result.ndim(), 2);
        assert_eq!(result.shape()[0], 2);
        assert_eq!(result.shape()[1], 2);
    }

    #[test]
    fn test_split_sections() {
        let dtype = DType::new(NpyType::Double);
        
        let shape = vec![6];
        let mut array = zeros(shape, dtype).unwrap();
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..6 {
                *ptr.add(i) = i as f64;
            }
        }
        
        let result = split(&array, SplitSpec::Sections(3), 0).unwrap();
        
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].size(), 2);
        assert_eq!(result[1].size(), 2);
        assert_eq!(result[2].size(), 2);
        
        unsafe {
            let ptr0 = result[0].data_ptr() as *const f64;
            assert_eq!(*ptr0.add(0), 0.0);
            assert_eq!(*ptr0.add(1), 1.0);
            
            let ptr1 = result[1].data_ptr() as *const f64;
            assert_eq!(*ptr1.add(0), 2.0);
            assert_eq!(*ptr1.add(1), 3.0);
        }
    }
}

