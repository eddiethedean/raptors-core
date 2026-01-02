//! Tests for iterator functionality

#[cfg(test)]
mod tests {
    use raptors_core::zeros;
    use raptors_core::iterators::{ArrayIterator, FlatIterator};
    use raptors_core::types::{DType, NpyType};

    #[test]
    fn test_array_iterator_new() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let iter = ArrayIterator::new(&array);
        assert_eq!(iter.index(), 0);
        assert!(!iter.is_exhausted());
        assert_eq!(iter.coordinates(), &[0, 0]);
    }

    #[test]
    fn test_array_iterator_next() {
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Int);
        let array = zeros(shape, dtype).unwrap();
        
        let mut iter = ArrayIterator::new(&array);
        
        // Should be able to iterate through all elements
        let mut count = 0;
        while iter.next() {
            count += 1;
        }
        
        assert_eq!(count, 6); // 2 * 3 = 6 elements
        assert!(iter.is_exhausted());
    }

    #[test]
    fn test_array_iterator_reset() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let mut iter = ArrayIterator::new(&array);
        
        // Advance a few steps
        iter.next();
        iter.next();
        assert_eq!(iter.index(), 2);
        
        // Reset
        iter.reset();
        assert_eq!(iter.index(), 0);
        assert_eq!(iter.coordinates(), &[0, 0]);
    }

    #[test]
    fn test_flat_iterator() {
        let shape = vec![3, 2];
        let dtype = DType::new(NpyType::Int);
        let array = zeros(shape, dtype).unwrap();
        
        let mut iter = FlatIterator::new(&array);
        
        let mut count = 0;
        while iter.next() {
            count += 1;
        }
        
        assert_eq!(count, 6); // 3 * 2 = 6 elements
    }

    #[test]
    fn test_iterator_trait() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let mut iter = ArrayIterator::new(&array);
        
        // Use Iterator trait
        let count = iter.by_ref().take(4).count();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_nditer_basic() {
        use raptors_core::iterators::advanced::{NdIter, IterFlags};
        
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(shape.clone(), dtype.clone()).unwrap();
        let array2 = zeros(shape, dtype).unwrap();
        
        let arrays = vec![&array1, &array2];
        let mut iter = NdIter::new(arrays, IterFlags::READONLY).unwrap();
        
        assert_eq!(iter.n_arrays(), 2);
        assert_eq!(iter.broadcast_shape(), &[2, 3]);
        assert!(!iter.is_exhausted());
        
        let mut count = 0;
        while iter.next() {
            count += 1;
        }
        
        assert_eq!(count, 6); // 2 * 3 = 6 elements
        assert!(iter.is_exhausted());
    }

    #[test]
    fn test_nditer_broadcasting() {
        use raptors_core::iterators::advanced::{NdIter, IterFlags};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![3, 1], dtype.clone()).unwrap();
        let array2 = zeros(vec![1, 4], dtype).unwrap();
        
        let arrays = vec![&array1, &array2];
        let mut iter = NdIter::new(arrays, IterFlags::READONLY).unwrap();
        
        // Broadcast shape should be [3, 4]
        assert_eq!(iter.broadcast_shape(), &[3, 4]);
        
        let mut count = 0;
        while iter.next() {
            count += 1;
        }
        
        assert_eq!(count, 12); // 3 * 4 = 12 elements
    }

    #[test]
    fn test_nditer_reset() {
        use raptors_core::iterators::advanced::{NdIter, IterFlags};
        
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Double);
        let array = zeros(shape, dtype).unwrap();
        
        let arrays = vec![&array];
        let mut iter = NdIter::new(arrays, IterFlags::READONLY).unwrap();
        
        iter.next();
        iter.next();
        assert_eq!(iter.index(), 2);
        
        iter.reset();
        assert_eq!(iter.index(), 0);
        assert!(!iter.is_exhausted());
    }

    #[test]
    fn test_nditer_data_ptrs() {
        use raptors_core::iterators::advanced::{NdIter, IterFlags};
        
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(shape.clone(), dtype.clone()).unwrap();
        let array2 = zeros(shape, dtype).unwrap();
        
        let arrays = vec![&array1, &array2];
        let mut iter = NdIter::new(arrays, IterFlags::READONLY).unwrap();
        
        assert!(iter.next());
        let ptrs = iter.get_data_ptrs();
        assert_eq!(ptrs.len(), 2);
        assert!(!ptrs[0].is_null());
        assert!(!ptrs[1].is_null());
    }
}

