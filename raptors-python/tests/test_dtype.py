"""Python pytest tests for DType bindings"""

import pytest
import raptors


class TestDTypeCreation:
    """Tests for dtype creation"""
    
    def test_create_from_string(self):
        """Test creating dtype from string"""
        dtype = raptors.DType("float64")
        assert dtype.name == "float64"
        assert dtype.itemsize == 8
    
    def test_create_float32(self):
        """Test creating float32 dtype"""
        dtype = raptors.DType("float32")
        assert dtype.name == "float32"
        assert dtype.itemsize == 4
    
    def test_create_int64(self):
        """Test creating int64 dtype"""
        dtype = raptors.DType("int64")
        assert dtype.name == "int64"
        assert dtype.itemsize == 8
    
    def test_invalid_dtype(self):
        """Test invalid dtype creation"""
        with pytest.raises(ValueError):
            raptors.DType("invalid_type")


class TestDTypeConstants:
    """Tests for dtype constants"""
    
    def test_float64_constant(self):
        """Test float64 constant"""
        assert raptors.float64.name == "float64"
        assert raptors.float64.itemsize == 8
    
    def test_float32_constant(self):
        """Test float32 constant"""
        assert raptors.float32.name == "float32"
        assert raptors.float32.itemsize == 4
    
    def test_int64_constant(self):
        """Test int64 constant"""
        assert raptors.int64.name == "int64"
        assert raptors.int64.itemsize == 8
    
    def test_int32_constant(self):
        """Test int32 constant"""
        assert raptors.int32.name == "int32"
        assert raptors.int32.itemsize == 4


class TestDTypeProperties:
    """Tests for dtype properties"""
    
    def test_name(self):
        """Test dtype name property"""
        dtype = raptors.float64
        assert dtype.name == "float64"
    
    def test_itemsize(self):
        """Test dtype itemsize property"""
        assert raptors.float64.itemsize == 8
        assert raptors.float32.itemsize == 4
        assert raptors.int64.itemsize == 8
        assert raptors.int32.itemsize == 4
    
    def test_kind(self):
        """Test dtype kind property"""
        assert raptors.float64.kind == "f"
        assert raptors.int64.kind == "i"
        assert raptors.bool_.kind == "b"
    
    def test_repr(self):
        """Test dtype string representation"""
        dtype = raptors.float64
        repr_str = repr(dtype)
        assert "float64" in repr_str

