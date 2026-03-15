
import numpy as np

# 1. Matrix operations
A = np.array([
    [4, 2, 1],
    [0, 5, 3],
    [2, 1, 3]
])

detA = np.linalg.det(A)
invA = np.linalg.inv(A)
eigvals, eigvecs = np.linalg.eig(A)

identity_check = np.allclose(A @ invA, np.eye(3))

print("Determinant:", detA)
print("Inverse:\n", invA)
print("Eigenvalues:", eigvals)
print("A * A_inv equals Identity:", identity_check)

# 2. Solve linear system
A_sys = np.array([[2,3],
                  [4,1]])

b = np.array([8,10])

solution = np.linalg.solve(A_sys, b)

print("Solution (x,y):", solution)

# 3. SVD explanation
explanation = '''
np.linalg.svd() performs Singular Value Decomposition, which factorizes a matrix
into three matrices: U, Sigma, and V^T.

In machine learning it is used for dimensionality reduction techniques such as
Principal Component Analysis (PCA).

It is also used in recommendation systems and latent factor models to discover
hidden structure in large datasets.
'''

print(explanation)
