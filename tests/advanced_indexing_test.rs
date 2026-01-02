//! Tests for advanced indexing functionality

#[cfg(test)]
mod tests {
    use raptors_core::zeros;
    use raptors_core::indexing::{fancy_index_array, boolean_index_array};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_fancy_indexing_basic() {
        // Create array [10.0, 20.0, 30.0, 40.0]
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(shape, dtype).unwrap();
        
        unsafe {
            let data_ptr = array.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 10.0;
            *data_ptr.add(1) = 20.0;
            *data_ptr.add(2) = 30.0;
            *data_ptr.add(3) = 40.0;
        }
        
        // Create index array [0, 2, 3]
        let index_shape = vec![3];
        let index_dtype = DType::new(NpyType::Int);
        let mut indices = zeros(index_shape, index_dtype).unwrap();
        
        unsafe {
            let index_ptr = indices.data_ptr_mut() as *mut i32;
            *index_ptr.add(0) = 0;
            *index_ptr.add(1) = 2;
            *index_ptr.add(2) = 3;
        }
        
        let result = fancy_index_array(&array, &indices).unwrap();
        
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert_eq!(*result_ptr.add(0), 10.0);
            assert_eq!(*result_ptr.add(1), 30.0);
            assert_eq!(*result_ptr.add(2), 40.0);
        }
    }

    #[test]
    fn test_fancy_indexing_negative() {
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(shape, dtype).unwrap();
        
        unsafe {
            let data_ptr = array.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 10.0;
            *data_ptr.add(1) = 20.0;
            *data_ptr.add(2) = 30.0;
            *data_ptr.add(3) = 40.0;
        }
        
        // Create index array with negative indices [-1, -2]
        let index_shape = vec![2];
        let index_dtype = DType::new(NpyType::Int);
        let mut indices = zeros(index_shape, index_dtype).unwrap();
        
        unsafe {
            let index_ptr = indices.data_ptr_mut() as *mut i32;
            *index_ptr.add(0) = -1; // Should be 3
            *index_ptr.add(1) = -2; // Should be 2
        }
        
        let result = fancy_index_array(&array, &indices).unwrap();
        
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert_eq!(*result_ptr.add(0), 40.0); // array[3]
            assert_eq!(*result_ptr.add(1), 30.0); // array[2]
        }
    }

    #[test]
    fn test_boolean_indexing_1d() {
        // Create array [10.0, 20.0, 30.0, 40.0]
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(shape, dtype).unwrap();
        
        unsafe {
            let data_ptr = array.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 10.0;
            *data_ptr.add(1) = 20.0;
            *data_ptr.add(2) = 30.0;
            *data_ptr.add(3) = 40.0;
        }
        
        // Create boolean mask [True, False, True, False]
        let mask_shape = vec![4];
        let mask_dtype = DType::new(NpyType::Bool);
        let mut mask = zeros(mask_shape, mask_dtype).unwrap();
        
        unsafe {
            let mask_ptr = mask.data_ptr_mut() as *mut bool;
            *mask_ptr.add(0) = true;
            *mask_ptr.add(1) = false;
            *mask_ptr.add(2) = true;
            *mask_ptr.add(3) = false;
        }
        
        let result = boolean_index_array(&array, &mask).unwrap();
        
        assert_eq!(result.size(), 2);
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert_eq!(*result_ptr.add(0), 10.0);
            assert_eq!(*result_ptr.add(1), 30.0);
        }
    }

    #[test]
    fn test_boolean_indexing_all_true() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(shape, dtype).unwrap();
        
        unsafe {
            let data_ptr = array.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 1.0;
            *data_ptr.add(1) = 2.0;
            *data_ptr.add(2) = 3.0;
        }
        
        let mask_shape = vec![3];
        let mask_dtype = DType::new(NpyType::Bool);
        let mut mask = zeros(mask_shape, mask_dtype).unwrap();
        
        unsafe {
            let mask_ptr = mask.data_ptr_mut() as *mut bool;
            *mask_ptr.add(0) = true;
            *mask_ptr.add(1) = true;
            *mask_ptr.add(2) = true;
        }
        
        let result = boolean_index_array(&array, &mask).unwrap();
        
        assert_eq!(result.size(), 3);
        unsafe {
            let result_ptr = result.data_ptr() as *const f64;
            assert_eq!(*result_ptr.add(0), 1.0);
            assert_eq!(*result_ptr.add(1), 2.0);
            assert_eq!(*result_ptr.add(2), 3.0);
        }
    }

    #[test]
    fn test_boolean_indexing_all_false() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(shape, dtype).unwrap();
        
        unsafe {
            let data_ptr = array.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 1.0;
            *data_ptr.add(1) = 2.0;
            *data_ptr.add(2) = 3.0;
        }
        
        let mask_shape = vec![3];
        let mask_dtype = DType::new(NpyType::Bool);
        let mask = zeros(mask_shape, mask_dtype).unwrap(); // All false by default
        
        let result = boolean_index_array(&array, &mask).unwrap();
        
        assert_eq!(result.size(), 0);
    }
}

