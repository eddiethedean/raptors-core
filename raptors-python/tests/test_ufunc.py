"""Python pytest tests for ufunc bindings"""

import pytest
import raptors
import numpy as np


class TestArithmeticUfuncs:
    """Tests for arithmetic ufuncs"""
    
    def test_add(self):
        """Test add ufunc"""
        a = raptors.ones([3, 3], dtype=raptors.float64)
        b = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.add_arrays(a, b)
        assert result.shape == [3, 3]
    
    def test_subtract(self):
        """Test subtract ufunc"""
        a = raptors.ones([3, 3], dtype=raptors.float64)
        b = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.subtract_arrays(a, b)
        assert result.shape == [3, 3]
    
    def test_multiply(self):
        """Test multiply ufunc"""
        a = raptors.ones([3, 3], dtype=raptors.float64)
        b = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.multiply_arrays(a, b)
        assert result.shape == [3, 3]
    
    def test_divide(self):
        """Test divide ufunc"""
        a = raptors.ones([3, 3], dtype=raptors.float64)
        b = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.divide_arrays(a, b)
        assert result.shape == [3, 3]


class TestComparisonUfuncs:
    """Tests for comparison ufuncs"""
    
    def test_equal(self):
        """Test equal ufunc"""
        a = raptors.ones([3, 3], dtype=raptors.float64)
        b = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.equal_arrays(a, b)
        assert result.shape == [3, 3]
        assert result.dtype.name == "bool"
    
    def test_less(self):
        """Test less ufunc"""
        a = raptors.zeros([3, 3], dtype=raptors.float64)
        b = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.less_arrays(a, b)
        assert result.shape == [3, 3]
        assert result.dtype.name == "bool"


class TestMathUfuncs:
    """Tests for mathematical ufuncs"""
    
    def test_sin(self):
        """Test sin ufunc"""
        arr = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.sin(arr)
        assert result.shape == [3, 3]
        assert result.dtype.name == "float64"
    
    def test_cos(self):
        """Test cos ufunc"""
        arr = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.cos(arr)
        assert result.shape == [3, 3]
    
    def test_exp(self):
        """Test exp ufunc"""
        arr = raptors.zeros([3, 3], dtype=raptors.float64)
        result = raptors.exp(arr)
        assert result.shape == [3, 3]
    
    def test_log(self):
        """Test log ufunc"""
        arr = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.log(arr)
        assert result.shape == [3, 3]
    
    def test_sqrt(self):
        """Test sqrt ufunc"""
        arr = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.sqrt(arr)
        assert result.shape == [3, 3]
    
    def test_abs(self):
        """Test abs ufunc"""
        arr = raptors.ones([3, 3], dtype=raptors.float64)
        result = raptors.abs(arr)
        assert result.shape == [3, 3]


class TestReductionUfuncs:
    """Tests for reduction ufuncs"""
    
    def test_sum(self):
        """Test sum reduction"""
        arr = raptors.ones([3, 4], dtype=raptors.float64)
        result = raptors.sum(arr)
        # Sum without axis should reduce to scalar
        assert result.shape == [] or result.size == 1
    
    def test_sum_axis(self):
        """Test sum with axis"""
        arr = raptors.ones([3, 4], dtype=raptors.float64)
        result = raptors.sum(arr, axis=0)
        assert result.shape == [4]
    
    def test_mean(self):
        """Test mean reduction"""
        arr = raptors.ones([3, 4], dtype=raptors.float64)
        result = raptors.mean(arr)
        assert result.size == 1
    
    def test_min(self):
        """Test min reduction"""
        arr = raptors.ones([3, 4], dtype=raptors.float64)
        result = raptors.min(arr)
        assert result.size == 1
    
    def test_max(self):
        """Test max reduction"""
        arr = raptors.ones([3, 4], dtype=raptors.float64)
        result = raptors.max(arr)
        assert result.size == 1

