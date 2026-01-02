"""Python pytest tests for Array bindings"""

import pytest
import raptors
import numpy as np


class TestArrayCreation:
    """Tests for array creation"""
    
    def test_zeros(self):
        """Test zeros array creation"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        assert arr.shape == [3, 4]
        assert arr.size == 12
        assert arr.ndim == 2
        assert arr.dtype.name == "float64"
    
    def test_ones(self):
        """Test ones array creation"""
        arr = raptors.ones([2, 3], dtype=raptors.float64)
        assert arr.shape == [2, 3]
        assert arr.size == 6
    
    def test_empty(self):
        """Test empty array creation"""
        arr = raptors.Array.empty([3, 4], dtype=raptors.float64)
        assert arr.shape == [3, 4]
        assert arr.size == 12


class TestArrayProperties:
    """Tests for array properties"""
    
    def test_shape(self):
        """Test shape property"""
        arr = raptors.zeros([3, 4, 5], dtype=raptors.float64)
        assert arr.shape == [3, 4, 5]
    
    def test_size(self):
        """Test size property"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        assert arr.size == 12
    
    def test_ndim(self):
        """Test ndim property"""
        arr = raptors.zeros([3, 4, 5], dtype=raptors.float64)
        assert arr.ndim == 3
    
    def test_dtype(self):
        """Test dtype property"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        assert arr.dtype.name == "float64"
        
        arr = raptors.zeros([3, 4], dtype=raptors.float32)
        assert arr.dtype.name == "float32"
    
    def test_strides(self):
        """Test strides property"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        strides = arr.strides
        assert len(strides) == 2
        assert strides[1] == 8  # itemsize for float64
    
    def test_contiguity(self):
        """Test contiguity flags"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        assert arr.is_c_contiguous == True
        assert arr.is_writeable == True


class TestArrayOperations:
    """Tests for array operations"""
    
    def test_addition(self):
        """Test array addition"""
        a = raptors.ones([2, 3], dtype=raptors.float64)
        b = raptors.ones([2, 3], dtype=raptors.float64)
        result = a + b
        assert result.shape == [2, 3]
    
    def test_subtraction(self):
        """Test array subtraction"""
        a = raptors.ones([2, 3], dtype=raptors.float64)
        b = raptors.ones([2, 3], dtype=raptors.float64)
        result = a - b
        assert result.shape == [2, 3]
    
    def test_multiplication(self):
        """Test array multiplication"""
        a = raptors.ones([2, 3], dtype=raptors.float64)
        b = raptors.ones([2, 3], dtype=raptors.float64)
        result = a * b
        assert result.shape == [2, 3]
    
    def test_division(self):
        """Test array division"""
        a = raptors.ones([2, 3], dtype=raptors.float64)
        b = raptors.ones([2, 3], dtype=raptors.float64)
        result = a / b
        assert result.shape == [2, 3]
    
    def test_comparison(self):
        """Test array comparison"""
        a = raptors.ones([2, 3], dtype=raptors.float64)
        b = raptors.ones([2, 3], dtype=raptors.float64)
        result = a == b
        assert result.shape == [2, 3]
        assert result.dtype.name == "bool"


class TestArrayManipulation:
    """Tests for array manipulation"""
    
    def test_reshape(self):
        """Test array reshape"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        reshaped = arr.reshape([12])
        assert reshaped.shape == [12]
        assert reshaped.size == 12
    
    def test_transpose(self):
        """Test array transpose"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        transposed = arr.transpose()
        assert transposed.shape == [4, 3]
    
    def test_copy(self):
        """Test array copy"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        copied = arr.copy()
        assert copied.shape == arr.shape
        assert copied.size == arr.size
    
    def test_view(self):
        """Test array view"""
        arr = raptors.zeros([3, 4], dtype=raptors.float64)
        view = arr.view()
        assert view.shape == arr.shape


class TestArrayIndexing:
    """Tests for array indexing"""
    
    def test_getitem_1d(self):
        """Test 1D array indexing"""
        arr = raptors.ones([5], dtype=raptors.float64)
        value = arr[2]
        assert value == 1.0
    
    def test_setitem_1d(self):
        """Test 1D array item assignment"""
        arr = raptors.zeros([5], dtype=raptors.float64)
        arr[2] = 42.0
        value = arr[2]
        assert value == 42.0


class TestArrayIterator:
    """Tests for array iteration"""
    
    def test_iteration(self):
        """Test array iteration"""
        arr = raptors.ones([5], dtype=raptors.float64)
        values = [x for x in arr]
        assert len(values) == 5
        assert all(v == 1.0 for v in values)
    
    def test_iterator_exhaustion(self):
        """Test iterator exhaustion"""
        arr = raptors.ones([3], dtype=raptors.float64)
        iterator = iter(arr)
        values = list(iterator)
        assert len(values) == 3
        
        # Iterator should be exhausted
        with pytest.raises(StopIteration):
            next(iterator)

