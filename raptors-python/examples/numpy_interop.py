"""NumPy interoperability examples"""

import raptors
import numpy as np

print("=== Raptors NumPy Interoperability Examples ===\n")

# Example 1: Create Raptors array
print("1. Raptors Array Creation:")
arr = raptors.zeros([3, 4], dtype=raptors.float64)
print(f"   Raptors array shape: {arr.shape}\n")

# Example 2: NumPy array creation (for comparison)
print("2. NumPy Array Creation:")
np_arr = np.zeros((3, 4), dtype=np.float64)
print(f"   NumPy array shape: {np_arr.shape}\n")

# Example 3: Operations work similarly
print("3. Similar Operations:")
raptors_sum = arr + arr
np_sum = np_arr + np_arr
print(f"   Raptors sum shape: {raptors_sum.shape}")
print(f"   NumPy sum shape: {np_sum.shape}\n")

print("=== Interoperability Examples Complete ===")

