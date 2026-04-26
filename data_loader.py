import numpy as np

X = np.load('data/X_timeseries.npy')
y = np.load('data/y_yield.npy')

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")
print(f"y min: {y.min()}")
print(f"y max: {y.max()}")

assert X.ndim == 3, f"X must be 3D, got {X.ndim}D"
assert y.ndim == 1, f"y must be 1D, got {y.ndim}D"
assert len(X) == len(y), f"X and y must have same length, got {len(X)} and {len(y)}"


