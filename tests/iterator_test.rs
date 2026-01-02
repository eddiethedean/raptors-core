//! Tests for iterator functionality

#[cfg(test)]
mod tests {
    use raptors_core::{Array, zeros};
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
}

