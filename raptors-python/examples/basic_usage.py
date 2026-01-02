"""Basic usage examples for Raptors Python bindings"""

import raptors
import numpy as np

print("=== Raptors Python Basic Usage Examples ===\n")

# Example 1: Array Creation
print("1. Array Creation:")
arr = raptors.zeros([3, 4], dtype=raptors.float64)
print(f"   Zeros array shape: {arr.shape}")

arr = raptors.ones([3, 4], dtype=raptors.float64)
print(f"   Ones array shape: {arr.shape}")

arr = raptors.Array.empty([3, 4], dtype=raptors.float64)
print(f"   Empty array created\n")

# Example 2: Array Properties
print("2. Array Properties:")
arr = raptors.zeros([3, 4], dtype=raptors.float64)
print(f"   Shape: {arr.shape}")
print(f"   Size: {arr.size}")
print(f"   NDim: {arr.ndim}")
print(f"   DType: {arr.dtype}")
print(f"   Is C-contiguous: {arr.is_c_contiguous}\n")

# Example 3: Array Operations
print("3. Array Operations:")
a = raptors.ones([3, 4], dtype=raptors.float64)
b = raptors.ones([3, 4], dtype=raptors.float64)

sum_arr = a + b
print(f"   Sum array shape: {sum_arr.shape}")

prod_arr = a * b
print(f"   Product array shape: {prod_arr.shape}\n")

# Example 4: Shape Manipulation
print("4. Shape Manipulation:")
arr = raptors.zeros([3, 4], dtype=raptors.float64)

reshaped = arr.reshape([12])
print(f"   Reshaped from [3, 4] to {reshaped.shape}")

transposed = arr.transpose()
print(f"   Transposed shape: {transposed.shape}\n")

print("=== Examples Complete ===")

