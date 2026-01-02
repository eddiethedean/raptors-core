//! Tests for sorting functionality

#[cfg(test)]
mod tests {
    use raptors_core::{Array, empty};
    use raptors_core::sorting::{sort, argsort, searchsorted, partition, SortKind};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_sort_double() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        // Set values: [3.0, 1.0, 4.0, 1.0, 5.0]
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 3.0;
            *ptr.add(1) = 1.0;
            *ptr.add(2) = 4.0;
            *ptr.add(3) = 1.0;
            *ptr.add(4) = 5.0;
        }
        
        sort(&mut array, SortKind::Quick).unwrap();
        
        unsafe {
            let ptr = array.data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 1.0);
            assert_eq!(*ptr.add(1), 1.0);
            assert_eq!(*ptr.add(2), 3.0);
            assert_eq!(*ptr.add(3), 4.0);
            assert_eq!(*ptr.add(4), 5.0);
        }
    }

    #[test]
    fn test_argsort() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        // Set values: [3.0, 1.0, 4.0, 1.0, 5.0]
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 3.0;
            *ptr.add(1) = 1.0;
            *ptr.add(2) = 4.0;
            *ptr.add(3) = 1.0;
            *ptr.add(4) = 5.0;
        }
        
        let indices = argsort(&array, SortKind::Quick).unwrap();
        
        unsafe {
            let idx_ptr = indices.data_ptr() as *const i64;
            // Indices that would sort: [1, 3, 0, 2, 4]
            assert_eq!(*idx_ptr.add(0), 1);
            assert_eq!(*idx_ptr.add(1), 3);
            assert_eq!(*idx_ptr.add(2), 0);
            assert_eq!(*idx_ptr.add(3), 2);
            assert_eq!(*idx_ptr.add(4), 4);
        }
    }

    #[test]
    fn test_searchsorted() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype.clone()).unwrap();
        
        // Set sorted values: [1.0, 2.0, 3.0, 4.0, 5.0]
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..5 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }
        
        let mut values = empty(vec![3], dtype).unwrap();
        unsafe {
            let val_ptr = values.data_ptr_mut() as *mut f64;
            *val_ptr.add(0) = 0.5; // Before first
            *val_ptr.add(1) = 2.5; // Between 2 and 3
            *val_ptr.add(2) = 6.0; // After last
        }
        
        let indices = searchsorted(&array, &values).unwrap();
        
        unsafe {
            let idx_ptr = indices.data_ptr() as *const i64;
            assert_eq!(*idx_ptr.add(0), 0); // Insert at start
            assert_eq!(*idx_ptr.add(1), 2); // Insert between 2 and 3
            assert_eq!(*idx_ptr.add(2), 5); // Insert at end
        }
    }

    #[test]
    fn test_partition() {
        let shape = vec![7];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        // Set values: [3, 1, 4, 1, 5, 9, 2]
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 3;
            *ptr.add(1) = 1;
            *ptr.add(2) = 4;
            *ptr.add(3) = 1;
            *ptr.add(4) = 5;
            *ptr.add(5) = 9;
            *ptr.add(6) = 2;
        }
        
        // Partition around index 3 (4th element, 0-indexed)
        partition(&mut array, 3).unwrap();
        
        unsafe {
            let ptr = array.data_ptr() as *const i32;
            let pivot = *ptr.add(3);
            // All elements before index 3 should be <= pivot
            for i in 0..3 {
                assert!(*ptr.add(i) <= pivot);
            }
            // All elements after index 3 should be >= pivot
            for i in 4..7 {
                assert!(*ptr.add(i) >= pivot);
            }
        }
    }

    #[test]
    fn test_sort_empty() {
        let shape = vec![0];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        // Should not panic
        sort(&mut array, SortKind::Quick).unwrap();
    }

    #[test]
    fn test_sort_single() {
        let shape = vec![1];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr = 5.0;
        }
        
        sort(&mut array, SortKind::Quick).unwrap();
        
        unsafe {
            let ptr = array.data_ptr() as *const f64;
            assert_eq!(*ptr, 5.0);
        }
    }
}

