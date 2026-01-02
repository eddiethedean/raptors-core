//! Tests for shape manipulation operations

#[cfg(test)]
mod tests {
    use raptors_core::shape::*;

    #[test]
    fn test_compute_size() {
        assert_eq!(compute_size(&[2, 3]), 6);
        assert_eq!(compute_size(&[2, 3, 4]), 24);
        assert_eq!(compute_size(&[5]), 5);
        assert_eq!(compute_size(&[]), 1); // Empty shape = scalar
    }

    #[test]
    fn test_validate_reshape_shape() {
        // Valid reshape
        assert!(validate_reshape_shape(&[2, 3], &[6]).is_ok());
        assert!(validate_reshape_shape(&[2, 3], &[3, 2]).is_ok());
        assert!(validate_reshape_shape(&[2, 3, 4], &[24]).is_ok());
        assert!(validate_reshape_shape(&[2, 3, 4], &[4, 6]).is_ok());

        // Invalid reshape (size mismatch)
        assert!(validate_reshape_shape(&[2, 3], &[5]).is_err());
        assert!(validate_reshape_shape(&[2, 3], &[7]).is_err());
    }

    #[test]
    fn test_compute_reshape_strides() {
        let strides = compute_reshape_strides(&[2, 3], 8);
        assert_eq!(strides, vec![24, 8]); // C-order: last stride = itemsize, previous = itemsize * shape[1]

        let strides = compute_reshape_strides(&[2, 3, 4], 8);
        assert_eq!(strides, vec![96, 32, 8]); // C-order strides
    }

    #[test]
    fn test_transpose_dimensions_default() {
        // Default transpose (reverse axes)
        let shape = vec![2, 3];
        let result = transpose_dimensions(&shape, None).unwrap();
        assert_eq!(result.0, vec![3, 2]); // Reversed

        let shape = vec![2, 3, 4];
        let result = transpose_dimensions(&shape, None).unwrap();
        assert_eq!(result.0, vec![4, 3, 2]); // Reversed
    }

    #[test]
    fn test_transpose_dimensions_with_axes() {
        // Transpose with explicit axes
        let shape = vec![2, 3, 4];
        let axes = Some(&[1, 0, 2][..]);
        let result = transpose_dimensions(&shape, axes).unwrap();
        assert_eq!(result.0, vec![3, 2, 4]); // Swapped first two dimensions

        let axes = Some(&[2, 1, 0][..]);
        let result = transpose_dimensions(&shape, axes).unwrap();
        assert_eq!(result.0, vec![4, 3, 2]); // Full reversal
    }

    #[test]
    fn test_transpose_dimensions_errors() {
        // Invalid: wrong number of axes
        let shape = vec![2, 3];
        let axes = Some(&[0, 1, 2][..]);
        assert!(transpose_dimensions(&shape, axes).is_err());

        // Invalid: duplicate axes
        let axes = Some(&[0, 0][..]);
        assert!(transpose_dimensions(&shape, axes).is_err());

        // Invalid: axis out of bounds
        let axes = Some(&[0, 3][..]);
        assert!(transpose_dimensions(&shape, axes).is_err());
    }

    #[test]
    fn test_squeeze_dims_all() {
        // Squeeze all dimensions of size 1
        assert_eq!(squeeze_dims(&[1, 3, 1], None), vec![3]);
        assert_eq!(squeeze_dims(&[2, 1, 3, 1], None), vec![2, 3]);
        assert_eq!(squeeze_dims(&[1, 1, 1], None), vec![]);
        assert_eq!(squeeze_dims(&[2, 3], None), vec![2, 3]); // No size-1 dims
    }

    #[test]
    fn test_squeeze_dims_specific_axis() {
        // Squeeze specific axis
        assert_eq!(squeeze_dims(&[1, 3, 4], Some(0)), vec![3, 4]); // Remove axis 0
        assert_eq!(squeeze_dims(&[2, 1, 4], Some(1)), vec![2, 4]); // Remove axis 1
        assert_eq!(squeeze_dims(&[2, 3, 1], Some(2)), vec![2, 3]); // Remove axis 2

        // Axis doesn't have size 1, should return unchanged
        assert_eq!(squeeze_dims(&[2, 3, 4], Some(1)), vec![2, 3, 4]);
    }

    #[test]
    fn test_expand_dims() {
        // Expand at beginning
        let result = expand_dims(&[2, 3], 0).unwrap();
        assert_eq!(result, vec![1, 2, 3]);

        // Expand in middle
        let result = expand_dims(&[2, 3], 1).unwrap();
        assert_eq!(result, vec![2, 1, 3]);

        // Expand at end
        let result = expand_dims(&[2, 3], 2).unwrap();
        assert_eq!(result, vec![2, 3, 1]);
    }

    #[test]
    fn test_expand_dims_errors() {
        // Invalid: axis too large
        assert!(expand_dims(&[2, 3], 3).is_err());
        assert!(expand_dims(&[2, 3], 4).is_err());
    }

    #[test]
    fn test_flatten_shape() {
        assert_eq!(flatten_shape(&[2, 3]), vec![6]);
        assert_eq!(flatten_shape(&[2, 3, 4]), vec![24]);
        assert_eq!(flatten_shape(&[5]), vec![5]);
        assert_eq!(flatten_shape(&[]), vec![1]); // Empty shape = scalar
    }
}

