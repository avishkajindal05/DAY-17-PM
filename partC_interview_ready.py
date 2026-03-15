
import numpy as np

# Q1 Vectorization
data = np.random.rand(1000,1000)

vectorized = data**2 + 2*data + 1


# Q2 K nearest neighbors
def k_nearest(data: np.ndarray, point: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(data - point, axis=1)
    return np.argsort(distances)[:k]


# Example
points = np.random.rand(10,2)
p = np.array([0.5,0.5])

print("Nearest:", k_nearest(points, p, 3))


# Q3 Debug fix
data = np.random.randn(100,5)

means = data.mean(axis=0, keepdims=True)
stds = data.std(axis=0, keepdims=True)

normalized = (data - means) / stds

print(normalized.shape)
