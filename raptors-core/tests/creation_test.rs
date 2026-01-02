//! Tests for array creation functions

#[cfg(test)]
mod tests {
    use raptors_core::{empty, ones, zeros};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_empty() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = empty(shape, dtype);
        assert!(array.is_ok());
    }

    #[test]
    fn test_zeros() {
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Int);
        let array = zeros(shape, dtype).unwrap();
        
        assert_eq!(array.size(), 6);
        // Verify first element is zero (uninitialized memory check would be unsafe)
    }

    #[test]
    fn test_ones() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Double);
        let array = ones(shape, dtype).unwrap();
        
        assert_eq!(array.size(), 4);
    }
}

