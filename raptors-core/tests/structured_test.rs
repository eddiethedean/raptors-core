//! Tests for structured arrays

#[cfg(test)]
mod tests {
    use raptors_core::DType;
    use raptors_core::types::NpyType;
    use raptors_core::structured::*;

    #[test]
    fn test_structured_dtype_creation() {
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)),
            ("y".to_string(), DType::new(NpyType::Double)),
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        assert_eq!(dtype.num_fields(), 2);
        assert_eq!(dtype.field_names(), vec!["x", "y"]);
    }

    #[test]
    fn test_structured_dtype_get_field() {
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)),
            ("y".to_string(), DType::new(NpyType::Int)),
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        
        let field_x = dtype.get_field_by_name("x").unwrap();
        assert_eq!(field_x.name, "x");
        assert_eq!(field_x.dtype.type_(), NpyType::Double);
        
        let field_y = dtype.get_field_by_name("y").unwrap();
        assert_eq!(field_y.name, "y");
        assert_eq!(field_y.dtype.type_(), NpyType::Int);
    }

    #[test]
    fn test_structured_dtype_field_not_found() {
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)),
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        assert!(dtype.get_field_by_name("y").is_none());
    }

    #[test]
    fn test_structured_dtype_empty_fields() {
        let fields = vec![];
        let result = StructuredDType::new(fields);
        assert!(result.is_err());
    }

    #[test]
    fn test_structured_dtype_itemsize() {
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)), // 8 bytes
            ("y".to_string(), DType::new(NpyType::Int)),    // 4 bytes
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        // Itemsize should be at least 12 bytes (8 + 4), plus alignment
        assert!(dtype.itemsize() >= 12);
    }

    #[test]
    fn test_structured_dtype_field_offsets() {
        let fields = vec![
            ("a".to_string(), DType::new(NpyType::Byte)),  // 1 byte
            ("b".to_string(), DType::new(NpyType::Int)),   // 4 bytes (aligned)
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        let field_a = dtype.get_field_by_name("a").unwrap();
        let field_b = dtype.get_field_by_name("b").unwrap();
        
        assert_eq!(field_a.offset, 0);
        // Field b should be aligned (offset should be multiple of 4)
        assert!(field_b.offset >= 1);
        assert_eq!(field_b.offset % 4, 0);
    }

    #[test]
    fn test_structured_dtype_get_field_by_index() {
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)),
            ("y".to_string(), DType::new(NpyType::Int)),
            ("z".to_string(), DType::new(NpyType::Float)),
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        
        let field0 = dtype.get_field(0).unwrap();
        assert_eq!(field0.name, "x");
        
        let field1 = dtype.get_field(1).unwrap();
        assert_eq!(field1.name, "y");
        
        let field2 = dtype.get_field(2).unwrap();
        assert_eq!(field2.name, "z");
        
        assert!(dtype.get_field(3).is_none());
    }

    #[test]
    fn test_structured_dtype_duplicate_field_names() {
        // This should be allowed in our implementation (NumPy allows it too)
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)),
            ("x".to_string(), DType::new(NpyType::Int)), // Duplicate name
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        // get_field_by_name returns the first match
        let field = dtype.get_field_by_name("x").unwrap();
        assert_eq!(field.dtype.type_(), NpyType::Double);
    }

    #[test]
    fn test_structured_dtype_empty_field_name() {
        let fields = vec![
            ("".to_string(), DType::new(NpyType::Double)),
        ];
        
        let result = StructuredDType::new(fields);
        assert!(result.is_err());
    }

    #[test]
    fn test_structured_array_creation() {
        let fields = vec![
            ("x".to_string(), DType::new(NpyType::Double)),
            ("y".to_string(), DType::new(NpyType::Int)),
        ];
        
        // Create test data (simplified - would need proper byte layout)
        let itemsize = StructuredDType::new(fields.clone()).unwrap().itemsize();
        let data = vec![0u8; itemsize * 2]; // 2 elements
        
        let result = structured_array(fields, &data, vec![2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_structured_dtype_alignment() {
        // Test that fields are properly aligned
        let fields = vec![
            ("a".to_string(), DType::new(NpyType::Byte)),   // 1 byte, align 1
            ("b".to_string(), DType::new(NpyType::Short)), // 2 bytes, align 2
            ("c".to_string(), DType::new(NpyType::Int)),   // 4 bytes, align 4
        ];
        
        let dtype = StructuredDType::new(fields).unwrap();
        let field_a = dtype.get_field(0).unwrap();
        let field_b = dtype.get_field(1).unwrap();
        let field_c = dtype.get_field(2).unwrap();
        
        assert_eq!(field_a.offset, 0);
        // b should be aligned to 2
        assert_eq!(field_b.offset % 2, 0);
        // c should be aligned to 4
        assert_eq!(field_c.offset % 4, 0);
    }
}

