import numpy as np

X = np.load("data/X_timeseries.npy")
y = np.load("data/y_yield.npy")

print("Initial X shape:", X.shape)
print("Initial y shape:", y.shape)

# --- FIX: truncate X to match y ---
n = len(y)
X_aligned = X[:n]

assert len(X_aligned) == len(y)

print("Aligned X shape:", X_aligned.shape)
print("Aligned y shape:", y.shape)

np.save("data/X_final.npy", X_aligned)
np.save("data/y_final.npy", y)

print("Saved X_final.npy and y_final.npy")
