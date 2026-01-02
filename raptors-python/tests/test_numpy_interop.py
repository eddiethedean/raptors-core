"""Python pytest tests for NumPy interoperability"""

import pytest
import raptors
import numpy as np


class TestNumPyInterop:
    """Tests for NumPy array interoperability"""
    
    def test_from_numpy(self):
        """Test converting NumPy array to Raptors array"""
        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        raptors_arr = raptors.from_numpy(np_arr)
        assert raptors_arr.shape == [2, 2]
        assert raptors_arr.size == 4
        assert raptors_arr.dtype.name == "float64"
    
    def test_to_numpy(self):
        """Test converting Raptors array to NumPy array"""
        raptors_arr = raptors.ones([2, 2], dtype=raptors.float64)
        np_arr = raptors.to_numpy(raptors_arr)
        assert np_arr.shape == (2, 2)
        assert np_arr.dtype == np.float64
        # Verify data is correct
        assert np.allclose(np_arr, np.ones((2, 2)))
    
    def test_compatible_dtypes(self):
        """Test that Raptors dtypes are compatible with NumPy concepts"""
        # Verify dtype names match NumPy conventions
        assert raptors.float64.name == "float64"
        assert raptors.float32.name == "float32"
        assert raptors.int64.name == "int64"
        assert raptors.int32.name == "int32"

