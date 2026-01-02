//! Tests for array functionality

#[cfg(test)]
mod tests {
    use raptors_core::array::Array;
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_array_creation() {
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Double);
        let array = Array::new(shape.clone(), dtype.clone());
        
        assert!(array.is_ok());
        let arr = array.unwrap();
        assert_eq!(arr.shape(), &shape);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.size(), 6);
        assert_eq!(arr.itemsize(), 8); // double is 8 bytes
        assert_eq!(arr.dtype().type_(), NpyType::Double);
    }

    #[test]
    fn test_array_shape() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let array = Array::new(shape.clone(), dtype).unwrap();
        
        assert_eq!(array.shape(), &[5]);
        assert_eq!(array.size(), 5);
        assert_eq!(array.ndim(), 1);
    }

    #[test]
    fn test_dtype_properties() {
        let dtype = DType::new(NpyType::Float);
        assert_eq!(dtype.itemsize(), 4);
        assert_eq!(dtype.name(), "float32");
        assert_eq!(dtype.type_(), NpyType::Float);
    }

    #[test]
    fn test_array_contiguity() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = Array::new(shape, dtype).unwrap();
        
        // New arrays should be C-contiguous
        assert!(array.is_c_contiguous());
        assert!(array.owns_data());
        assert!(array.is_writeable());
    }

    #[test]
    fn test_array_flags() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let array = Array::new(shape, dtype).unwrap();
        
        let flags = array.flags();
        assert!(flags.contains(raptors_core::array::ArrayFlags::C_CONTIGUOUS));
        assert!(flags.contains(raptors_core::array::ArrayFlags::OWNDATA));
        assert!(flags.contains(raptors_core::array::ArrayFlags::WRITEABLE));
    }
}

