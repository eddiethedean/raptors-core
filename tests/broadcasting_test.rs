//! Tests for broadcasting functionality

#[cfg(test)]
mod tests {
    use raptors_core::broadcasting::*;

    #[test]
    fn test_can_broadcast_compatible() {
        assert!(can_broadcast(&[5], &[1]));
        assert!(can_broadcast(&[1], &[5]));
        assert!(can_broadcast(&[5, 1], &[1, 3]));
        assert!(can_broadcast(&[4, 1, 3], &[1, 3]));
    }

    #[test]
    fn test_can_broadcast_incompatible() {
        assert!(!can_broadcast(&[5], &[3]));
        assert!(!can_broadcast(&[5, 4], &[3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_same() {
        let result = broadcast_shapes(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_with_one() {
        let result = broadcast_shapes(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_different_lengths() {
        let result = broadcast_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(result, vec![3, 4]);
        
        let result = broadcast_shapes(&[4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        let result = broadcast_shapes(&[3, 4], &[2, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_shapes_multi() {
        let shapes = vec![&[1, 3, 1][..], &[3, 1, 5][..], &[1, 1, 5][..]];
        let result = broadcast_shapes_multi(&shapes).unwrap();
        assert_eq!(result, vec![3, 3, 5]);
    }

    #[test]
    fn test_broadcast_strides() {
        let orig_shape = vec![1, 3];
        let orig_strides = vec![0, 8]; // Stride 0 for dimension of size 1
        let target_shape = vec![4, 1, 3];
        
        let result = broadcast_strides(&orig_shape, &orig_strides, &target_shape).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0); // Broadcast dimension
        assert_eq!(result[1], 0); // Original dimension of size 1
        assert_eq!(result[2], 8); // Original stride
    }
}

