import pandas as pd
import time
import numpy as np

# the None value in the list is converted to a np.nan (not a number)
c_list = [12.5, 14.2, 10.1, None, 16.8, 20.0]
arr_c = np.array(c_list, dtype=float)
arr_f = arr_c * 9/5 + 32 # -> vectorized operation: applies to the whole vector

arr_c, arr_f, arr_c.dtype, arr_c.shape

# **** VECTORIZATION AND MASKING *******


# ~ indicates a NOT condition
# "give me all the elements in this array that are NON null"
# mask == True for valid values
# mask == False for NaN values
mask = ~np.isnan(arr_c)
mask

# mask is a subset if arr_c, based on a condition
mean_c_np = np.mean(arr_c[mask])
mean_c_np

labels_np = np.where(arr_c < 5, 'cold', np.where(arr_c < 15, 'mild', 'warm'))
labels_np

# vectorising is very efficient:
# Python loop vs Numpy vectporization
# Generate large synthetic data
rng = np.random.default_rng(42)
large_c = rng.normal(loc=12.0, scale=8.0, size=2_000_000)
large_c_list = large_c.tolist()

print(large_c_list[:10])

%%timeit -n 3 -r 3
# Python loop timing
out = []
for v in large_c_list:
    out.append(v * 9/5 + 32)

%%timeit -n 3 -r 3
# NumPy vectorized timing
out_np = large_c * 9/5 + 32