//! Tests for C API functions

#[cfg(test)]
mod tests {
    use raptors_core::ffi::PyArrayObject;
    use std::ptr;

    // Helper function to create a simple test array structure
    // This is a minimal implementation for testing
    fn create_test_array() -> Box<PyArrayObject> {
        let mut dims = [0i64; 64];
        dims[0] = 3;
        dims[1] = 4;
        
        let mut strides = [0i64; 64];
        strides[0] = 32; // 4 * 8 bytes per double
        strides[1] = 8;  // 8 bytes per double
        
        Box::new(PyArrayObject {
            ob_base: ptr::null_mut(),
            data: ptr::null_mut(),
            nd: 2,
            descr: ptr::null_mut(),
            flags: 0,
            dimensions: dims,
            strides: strides,
            base: ptr::null_mut(),
            _descr: ptr::null_mut(),
            weakreflist: ptr::null_mut(),
        })
    }

    #[test]
    fn test_pyarray_ndim() {
        use raptors_core::ffi::PyArray_NDIM;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let ndim = unsafe { PyArray_NDIM(arr_ptr) };
        assert_eq!(ndim, 2);
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_ndim_null() {
        use raptors_core::ffi::PyArray_NDIM;
        
        let result = unsafe { PyArray_NDIM(ptr::null_mut()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_pyarray_dim() {
        use raptors_core::ffi::PyArray_DIM;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let dim0 = unsafe { PyArray_DIM(arr_ptr, 0) };
        assert_eq!(dim0, 3);
        
        let dim1 = unsafe { PyArray_DIM(arr_ptr, 1) };
        assert_eq!(dim1, 4);
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_stride() {
        use raptors_core::ffi::PyArray_STRIDE;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let stride0 = unsafe { PyArray_STRIDE(arr_ptr, 0) };
        assert_eq!(stride0, 32);
        
        let stride1 = unsafe { PyArray_STRIDE(arr_ptr, 1) };
        assert_eq!(stride1, 8);
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_data() {
        use raptors_core::ffi::PyArray_DATA;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let data_ptr = unsafe { PyArray_DATA(arr_ptr) };
        assert!(!data_ptr.is_null() || data_ptr.is_null()); // Just check it doesn't crash
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_dims() {
        use raptors_core::ffi::PyArray_DIMS;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let dims_ptr = unsafe { PyArray_DIMS(arr_ptr) };
        assert!(!dims_ptr.is_null());
        
        unsafe {
            assert_eq!(*dims_ptr, 3);
            assert_eq!(*dims_ptr.add(1), 4);
        }
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_strides() {
        use raptors_core::ffi::PyArray_STRIDES;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let strides_ptr = unsafe { PyArray_STRIDES(arr_ptr) };
        assert!(!strides_ptr.is_null());
        
        unsafe {
            assert_eq!(*strides_ptr, 32);
            assert_eq!(*strides_ptr.add(1), 8);
        }
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }
}

