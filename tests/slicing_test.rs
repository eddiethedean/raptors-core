//! Tests for slicing functionality

#[cfg(test)]
mod tests {
    use raptors_core::indexing::{Slice, normalize_slice, slice_length, compute_slice_shape};

    #[test]
    fn test_slice_creation() {
        let slice = Slice::new(Some(1), Some(5), Some(2));
        assert_eq!(slice.start, Some(1));
        assert_eq!(slice.stop, Some(5));
        assert_eq!(slice.step, Some(2));
    }

    #[test]
    fn test_normalize_slice_positive_step() {
        let slice = Slice::new(Some(1), Some(5), Some(1));
        let (start, stop, step) = normalize_slice(&slice, 10).unwrap();
        assert_eq!(start, 1);
        assert_eq!(stop, 5);
        assert_eq!(step, 1);
    }

    #[test]
    fn test_normalize_slice_no_start() {
        let slice = Slice::new(None, Some(5), Some(1));
        let (start, stop, step) = normalize_slice(&slice, 10).unwrap();
        assert_eq!(start, 0);
        assert_eq!(stop, 5);
        assert_eq!(step, 1);
    }

    #[test]
    fn test_normalize_slice_negative_indices() {
        let slice = Slice::new(Some(-3), Some(-1), Some(1));
        let (start, stop, step) = normalize_slice(&slice, 10).unwrap();
        assert_eq!(start, 7); // 10 - 3
        assert_eq!(stop, 9);  // 10 - 1
    }

    #[test]
    fn test_slice_length() {
        assert_eq!(slice_length(0, 10, 1), 10);
        assert_eq!(slice_length(0, 10, 2), 5);
        assert_eq!(slice_length(5, 10, 1), 5);
        assert_eq!(slice_length(0, 0, 1), 0);
    }

    #[test]
    fn test_compute_slice_shape() {
        let slices = vec![
            Slice::new(Some(1), Some(5), Some(1)),
            Slice::new(Some(0), Some(3), Some(1)),
        ];
        let array_shape = vec![10, 5];
        
        let result = compute_slice_shape(&slices, &array_shape).unwrap();
        assert_eq!(result, vec![4, 3]);
    }
}

