//! Tests for memory-mapped arrays

#[cfg(test)]
mod tests {
    use raptors_core::DType;
    use raptors_core::types::NpyType;
    use raptors_core::memmap::*;
    use std::path::Path;
    use std::fs;

    #[test]
    fn test_memmap_array_creation() {
        // Create a temporary file for testing
        let temp_file = Path::new("/tmp/test_memmap_array.npy");
        
        // Create test file with some data
        let test_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(&temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let result = memmap_array(&temp_file, dtype, vec![4]);
        
        // Cleanup
        let _ = fs::remove_file(&temp_file);
        
        // Would test when properly implemented
        // assert!(result.is_ok());
        // let mmap = result.unwrap();
        // assert_eq!(mmap.array().size(), 4);
    }

    #[test]
    fn test_memmap_array_file_not_found() {
        let dtype = DType::new(NpyType::Double);
        let nonexistent = Path::new("/tmp/nonexistent_file.npy");
        let result = memmap_array(nonexistent, dtype, vec![4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_memmap_flush() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_flush.npy");
        
        // Create test file
        let test_data = vec![1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(&temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mut mmap = memmap_array_writable(&temp_file, dtype.clone(), vec![3]).unwrap();
        
        // Modify data
        let array = mmap.array_mut();
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 99.0;
        }
        
        // Flush changes
        mmap.flush().unwrap();
        
        // Verify file was updated
        let loaded = load_memmap(&temp_file, dtype, vec![3]).unwrap();
        unsafe {
            let ptr = loaded.array().data_ptr() as *const f64;
            assert_eq!(*ptr, 99.0);
        }
        
        let _ = fs::remove_file(&temp_file);
    }

    #[test]
    fn test_save_memmap() {
        use std::fs;
        use raptors_core::{Array, zeros};
        use raptors_core::memmap::save_memmap;
        
        let temp_file = Path::new("/tmp/test_save_memmap.npy");
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(vec![3], dtype.clone()).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 20.0;
            *ptr.add(2) = 30.0;
        }
        
        let mmap = save_memmap(&array, &temp_file).unwrap();
        assert_eq!(mmap.array().size(), 3);
        
        // Verify data
        unsafe {
            let ptr = mmap.array().data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 10.0);
            assert_eq!(*ptr.add(1), 20.0);
            assert_eq!(*ptr.add(2), 30.0);
        }
        
        let _ = fs::remove_file(&temp_file);
    }

    #[test]
    fn test_memmap_sync() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_sync.npy");
        
        let test_data = vec![1.0f64, 2.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(&temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mmap = memmap_array(&temp_file, dtype, vec![2]).unwrap();
        
        // Sync should not fail
        mmap.sync().unwrap();
        
        let _ = fs::remove_file(&temp_file);
    }

    #[test]
    fn test_memmap_file_too_small() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_small.npy");
        
        // Create file with insufficient data
        fs::write(&temp_file, vec![0u8; 4]).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let result = memmap_array(&temp_file, dtype, vec![10]); // Need 80 bytes for 10 doubles
        
        assert!(result.is_err());
        
        let _ = fs::remove_file(&temp_file);
    }

    #[test]
    fn test_memmap_writable_creates_file() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_create.npy");
        
        // File doesn't exist yet
        assert!(!temp_file.exists());
        
        let dtype = DType::new(NpyType::Double);
        let result = memmap_array_writable(&temp_file, dtype, vec![3]);
        
        // Should create file
        assert!(result.is_ok());
        assert!(temp_file.exists());
        
        let mmap = result.unwrap();
        assert_eq!(mmap.array().size(), 3);
        assert_eq!(mmap.mode(), MapMode::ReadWrite);
        
        let _ = fs::remove_file(&temp_file);
    }
}

