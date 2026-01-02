//! Tests for array manipulation functionality

#[cfg(test)]
mod tests {
    use raptors_core::empty;
    use raptors_core::manipulation::{flip, flipud, fliplr, rotate90, roll, repeat, tile, unique};
    use raptors_core::manipulation::{union1d, intersect1d};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_flip_2d() {
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        // Set values: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..6 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }
        
        // Flip along axis 0 (vertical flip)
        let flipped = flip(&array, 0).unwrap();
        
        unsafe {
            let flipped_ptr = flipped.data_ptr() as *const f64;
            // Should be [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
            assert_eq!(*flipped_ptr.add(0), 4.0);
            assert_eq!(*flipped_ptr.add(3), 1.0);
        }
    }

    #[test]
    fn test_flipud() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 1;
            *ptr.add(1) = 2;
            *ptr.add(2) = 3;
            *ptr.add(3) = 4;
        }
        
        let flipped = flipud(&array).unwrap();
        
        unsafe {
            let flipped_ptr = flipped.data_ptr() as *const i32;
            assert_eq!(*flipped_ptr.add(0), 3);
            assert_eq!(*flipped_ptr.add(2), 1);
        }
    }

    #[test]
    fn test_fliplr() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 1;
            *ptr.add(1) = 2;
            *ptr.add(2) = 3;
            *ptr.add(3) = 4;
        }
        
        let flipped = fliplr(&array).unwrap();
        
        unsafe {
            let flipped_ptr = flipped.data_ptr() as *const i32;
            assert_eq!(*flipped_ptr.add(0), 2);
            assert_eq!(*flipped_ptr.add(1), 1);
        }
    }

    #[test]
    fn test_rotate90() {
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Double);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..6 {
                *ptr.add(i) = (i + 1) as f64;
            }
        }
        
        // Rotate 90 degrees clockwise
        let rotated = rotate90(&array, 1).unwrap();
        
        assert_eq!(rotated.shape(), &[3, 2]);
    }

    #[test]
    fn test_roll() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            for i in 0..5 {
                *ptr.add(i) = i as i32;
            }
        }
        
        // Roll by 2 positions
        let rolled = roll(&array, 2, None).unwrap();
        
        unsafe {
            let rolled_ptr = rolled.data_ptr() as *const i32;
            // [0, 1, 2, 3, 4] -> [3, 4, 0, 1, 2]
            assert_eq!(*rolled_ptr.add(0), 3);
            assert_eq!(*rolled_ptr.add(1), 4);
            assert_eq!(*rolled_ptr.add(2), 0);
        }
    }

    #[test]
    fn test_repeat() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 1;
            *ptr.add(1) = 2;
            *ptr.add(2) = 3;
        }
        
        // Repeat each element 2 times
        let repeated = repeat(&array, 2, None).unwrap();
        
        assert_eq!(repeated.size(), 6);
        unsafe {
            let repeated_ptr = repeated.data_ptr() as *const i32;
            assert_eq!(*repeated_ptr.add(0), 1);
            assert_eq!(*repeated_ptr.add(1), 1);
            assert_eq!(*repeated_ptr.add(2), 2);
        }
    }

    #[test]
    fn test_tile() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 1;
            *ptr.add(1) = 2;
            *ptr.add(2) = 3;
            *ptr.add(3) = 4;
        }
        
        // Tile 2x2
        let tiled = tile(&array, &[2, 2]).unwrap();
        
        assert_eq!(tiled.shape(), &[4, 4]);
    }

    #[test]
    fn test_unique() {
        let shape = vec![6];
        let dtype = DType::new(NpyType::Int);
        let mut array = empty(shape, dtype).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 3;
            *ptr.add(1) = 1;
            *ptr.add(2) = 2;
            *ptr.add(3) = 1;
            *ptr.add(4) = 3;
            *ptr.add(5) = 2;
        }
        
        let unique_arr = unique(&array).unwrap();
        
        assert_eq!(unique_arr.size(), 3);
        unsafe {
            let unique_ptr = unique_arr.data_ptr() as *const i32;
            // Should be sorted: [1, 2, 3]
            assert_eq!(*unique_ptr.add(0), 1);
            assert_eq!(*unique_ptr.add(1), 2);
            assert_eq!(*unique_ptr.add(2), 3);
        }
    }

    #[test]
    fn test_union1d() {
        let dtype = DType::new(NpyType::Int);
        let mut a1 = empty(vec![3], dtype.clone()).unwrap();
        let mut a2 = empty(vec![3], dtype).unwrap();
        
        unsafe {
            let ptr1 = a1.data_ptr_mut() as *mut i32;
            *ptr1.add(0) = 1;
            *ptr1.add(1) = 2;
            *ptr1.add(2) = 3;
            
            let ptr2 = a2.data_ptr_mut() as *mut i32;
            *ptr2.add(0) = 3;
            *ptr2.add(1) = 4;
            *ptr2.add(2) = 5;
        }
        
        let union = union1d(&a1, &a2).unwrap();
        
        assert!(union.size() >= 3);
    }

    #[test]
    fn test_intersect1d() {
        let dtype = DType::new(NpyType::Int);
        let mut a1 = empty(vec![3], dtype.clone()).unwrap();
        let mut a2 = empty(vec![3], dtype).unwrap();
        
        unsafe {
            let ptr1 = a1.data_ptr_mut() as *mut i32;
            *ptr1.add(0) = 1;
            *ptr1.add(1) = 2;
            *ptr1.add(2) = 3;
            
            let ptr2 = a2.data_ptr_mut() as *mut i32;
            *ptr2.add(0) = 2;
            *ptr2.add(1) = 3;
            *ptr2.add(2) = 4;
        }
        
        let intersect = intersect1d(&a1, &a2).unwrap();
        
        assert!(intersect.size() >= 1);
    }
}

