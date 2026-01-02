//! Tests for advanced ufunc functionality

#[cfg(test)]
mod tests {
    use raptors_core::{Array, zeros, empty};
    use raptors_core::ufunc::{create_sin_ufunc, create_cos_ufunc, create_exp_ufunc, create_log_ufunc, create_sqrt_ufunc, create_abs_ufunc, create_floor_ufunc, create_ceil_ufunc, create_unary_ufunc_loop};
    use raptors_core::types::{DType, NpyType};

    fn create_test_array() -> Array {
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(shape, dtype).unwrap();
        
        // Set some test values
        unsafe {
            let data_ptr = array.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 0.0;
            *data_ptr.add(1) = std::f64::consts::PI / 4.0; // π/4
            *data_ptr.add(2) = std::f64::consts::PI / 2.0; // π/2
            *data_ptr.add(3) = 1.0;
        }
        
        array
    }

    #[test]
    fn test_sin_ufunc() {
        let input = create_test_array();
        let mut output = empty(input.shape().to_vec(), input.dtype().clone()).unwrap();
        
        let ufunc = create_sin_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            // sin(0) = 0
            assert!((*out_ptr.add(0)).abs() < 1e-10);
            // sin(π/4) ≈ 0.707
            assert!((*out_ptr.add(1) - 0.7071067811865475).abs() < 1e-10);
            // sin(π/2) = 1
            assert!((*out_ptr.add(2) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cos_ufunc() {
        let input = create_test_array();
        let mut output = empty(input.shape().to_vec(), input.dtype().clone()).unwrap();
        
        let ufunc = create_cos_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            // cos(0) = 1
            assert!((*out_ptr.add(0) - 1.0).abs() < 1e-10);
            // cos(π/4) ≈ 0.707
            assert!((*out_ptr.add(1) - 0.7071067811865475).abs() < 1e-10);
            // cos(π/2) ≈ 0
            assert!((*out_ptr.add(2)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exp_ufunc() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut input = zeros(shape, dtype.clone()).unwrap();
        
        unsafe {
            let data_ptr = input.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 0.0;  // exp(0) = 1
            *data_ptr.add(1) = 1.0;  // exp(1) ≈ e
            *data_ptr.add(2) = 2.0;  // exp(2)
        }
        
        let mut output = empty(input.shape().to_vec(), dtype).unwrap();
        let ufunc = create_exp_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            assert!((*out_ptr.add(0) - 1.0).abs() < 1e-10);
            assert!((*out_ptr.add(1) - std::f64::consts::E).abs() < 1e-10);
            assert!((*out_ptr.add(2) - (std::f64::consts::E * std::f64::consts::E)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_log_ufunc() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let mut input = zeros(shape, dtype.clone()).unwrap();
        
        unsafe {
            let data_ptr = input.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 1.0;  // ln(1) = 0
            *data_ptr.add(1) = std::f64::consts::E;  // ln(e) = 1
            *data_ptr.add(2) = 10.0;  // ln(10)
        }
        
        let mut output = empty(input.shape().to_vec(), dtype).unwrap();
        let ufunc = create_log_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            assert!((*out_ptr.add(0)).abs() < 1e-10);
            assert!((*out_ptr.add(1) - 1.0).abs() < 1e-10);
            assert!((*out_ptr.add(2) - 10.0_f64.ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sqrt_ufunc() {
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut input = zeros(shape, dtype.clone()).unwrap();
        
        unsafe {
            let data_ptr = input.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 0.0;  // sqrt(0) = 0
            *data_ptr.add(1) = 1.0;  // sqrt(1) = 1
            *data_ptr.add(2) = 4.0;  // sqrt(4) = 2
            *data_ptr.add(3) = 9.0;  // sqrt(9) = 3
        }
        
        let mut output = empty(input.shape().to_vec(), dtype).unwrap();
        let ufunc = create_sqrt_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            assert!((*out_ptr.add(0)).abs() < 1e-10);
            assert!((*out_ptr.add(1) - 1.0).abs() < 1e-10);
            assert!((*out_ptr.add(2) - 2.0).abs() < 1e-10);
            assert!((*out_ptr.add(3) - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_abs_ufunc() {
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut input = zeros(shape, dtype.clone()).unwrap();
        
        unsafe {
            let data_ptr = input.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = -5.0;
            *data_ptr.add(1) = 5.0;
            *data_ptr.add(2) = 0.0;
            *data_ptr.add(3) = -std::f64::consts::PI;
        }
        
        let mut output = empty(input.shape().to_vec(), dtype).unwrap();
        let ufunc = create_abs_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            assert!((*out_ptr.add(0) - 5.0).abs() < 1e-10);
            assert!((*out_ptr.add(1) - 5.0).abs() < 1e-10);
            assert!((*out_ptr.add(2)).abs() < 1e-10);
            assert!((*out_ptr.add(3) - std::f64::consts::PI).abs() < 1e-10);
        }
    }

    #[test]
    fn test_floor_ufunc() {
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut input = zeros(shape, dtype.clone()).unwrap();
        
        unsafe {
            let data_ptr = input.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 3.7;
            *data_ptr.add(1) = -3.7;
            *data_ptr.add(2) = 5.0;
            *data_ptr.add(3) = -5.0;
        }
        
        let mut output = empty(input.shape().to_vec(), dtype).unwrap();
        let ufunc = create_floor_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            assert!((*out_ptr.add(0) - 3.0).abs() < 1e-10);
            assert!((*out_ptr.add(1) + 4.0).abs() < 1e-10); // floor(-3.7) = -4
            assert!((*out_ptr.add(2) - 5.0).abs() < 1e-10);
            assert!((*out_ptr.add(3) + 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ceil_ufunc() {
        let shape = vec![4];
        let dtype = DType::new(NpyType::Double);
        let mut input = zeros(shape, dtype.clone()).unwrap();
        
        unsafe {
            let data_ptr = input.data_ptr_mut() as *mut f64;
            *data_ptr.add(0) = 3.2;
            *data_ptr.add(1) = -3.2;
            *data_ptr.add(2) = 5.0;
            *data_ptr.add(3) = -5.0;
        }
        
        let mut output = empty(input.shape().to_vec(), dtype).unwrap();
        let ufunc = create_ceil_ufunc();
        create_unary_ufunc_loop(&ufunc, &input, &mut output).unwrap();
        
        unsafe {
            let out_ptr = output.data_ptr() as *const f64;
            assert!((*out_ptr.add(0) - 4.0).abs() < 1e-10);
            assert!((*out_ptr.add(1) + 3.0).abs() < 1e-10); // ceil(-3.2) = -3
            assert!((*out_ptr.add(2) - 5.0).abs() < 1e-10);
            assert!((*out_ptr.add(3) + 5.0).abs() < 1e-10);
        }
    }
}

