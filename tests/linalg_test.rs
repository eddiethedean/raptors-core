//! Tests for linear algebra operations

#[cfg(test)]
mod tests {
    use raptors_core::zeros;
    use raptors_core::linalg::{dot, matmul};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_dot_1d_1d() {
        let dtype = DType::new(NpyType::Double);
        
        let shape = vec![3];
        let mut a = zeros(shape.clone(), dtype.clone()).unwrap();
        unsafe {
            let ptr = a.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
        }
        
        let mut b = zeros(shape, dtype).unwrap();
        unsafe {
            let ptr = b.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 4.0;
            *ptr.add(1) = 5.0;
            *ptr.add(2) = 6.0;
        }
        
        let result = dot(&a, &b).unwrap();
        
        assert_eq!(result.size(), 1);
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assert_eq!(*ptr, 32.0);
        }
    }

    #[test]
    fn test_dot_2d_2d() {
        let dtype = DType::new(NpyType::Double);
        
        // 2x3 matrix
        let shape_a = vec![2, 3];
        let mut a = zeros(shape_a, dtype.clone()).unwrap();
        unsafe {
            let ptr = a.data_ptr_mut() as *mut f64;
            // [[1, 2, 3],
            //  [4, 5, 6]]
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
            *ptr.add(4) = 5.0;
            *ptr.add(5) = 6.0;
        }
        
        // 3x2 matrix
        let shape_b = vec![3, 2];
        let mut b = zeros(shape_b, dtype).unwrap();
        unsafe {
            let ptr = b.data_ptr_mut() as *mut f64;
            // [[7, 8],
            //  [9, 10],
            //  [11, 12]]
            *ptr.add(0) = 7.0;
            *ptr.add(1) = 8.0;
            *ptr.add(2) = 9.0;
            *ptr.add(3) = 10.0;
            *ptr.add(4) = 11.0;
            *ptr.add(5) = 12.0;
        }
        
        let result = dot(&a, &b).unwrap();
        
        assert_eq!(result.shape(), &[2, 2]);
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            // First row: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
            assert!((*ptr.add(0) - 58.0).abs() < 1e-10);
            assert!((*ptr.add(1) - 64.0).abs() < 1e-10);
            // Second row: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
            assert!((*ptr.add(2) - 139.0).abs() < 1e-10);
            assert!((*ptr.add(3) - 154.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matmul_2d_2d() {
        let dtype = DType::new(NpyType::Double);
        
        let shape_a = vec![2, 2];
        let mut a = zeros(shape_a, dtype.clone()).unwrap();
        unsafe {
            let ptr = a.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
        }
        
        let shape_b = vec![2, 2];
        let mut b = zeros(shape_b, dtype).unwrap();
        unsafe {
            let ptr = b.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 5.0;
            *ptr.add(1) = 6.0;
            *ptr.add(2) = 7.0;
            *ptr.add(3) = 8.0;
        }
        
        let result = matmul(&a, &b).unwrap();
        
        assert_eq!(result.shape(), &[2, 2]);
        unsafe {
            let ptr = result.data_ptr() as *const f64;
            // [[19, 22],
            //  [43, 50]]
            assert!((*ptr.add(0) - 19.0).abs() < 1e-10);
            assert!((*ptr.add(1) - 22.0).abs() < 1e-10);
            assert!((*ptr.add(2) - 43.0).abs() < 1e-10);
            assert!((*ptr.add(3) - 50.0).abs() < 1e-10);
        }
    }
}

