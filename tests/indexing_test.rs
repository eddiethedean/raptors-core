//! Tests for indexing functionality

#[cfg(test)]
mod tests {
    use raptors_core::Array;
    use raptors_core::zeros;
    use raptors_core::indexing::*;
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_index_array() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let indices = vec![1, 2];
        let result = index_array(&array, &indices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_index_array_out_of_bounds() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let indices = vec![5, 2]; // Out of bounds
        let result = index_array(&array, &indices);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IndexError::OutOfBounds));
    }

    #[test]
    fn test_index_array_dimension_mismatch() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let indices = vec![1, 2, 3]; // Wrong number of dimensions
        let result = index_array(&array, &indices);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IndexError::DimensionMismatch));
    }

    #[test]
    fn test_validate_index() {
        assert!(validate_index(0, 5).is_ok());
        assert!(validate_index(4, 5).is_ok());
        assert!(validate_index(5, 5).is_err());
        assert!(validate_index(-1, 5).is_err());
    }
}

