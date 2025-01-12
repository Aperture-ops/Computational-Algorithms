import numpy as np

def modified_cholesky_decomposition(A):
    """
    Performs a modified Cholesky decomposition of a symmetric matrix A.
    Returns the matrices L and E such that A = L * L.T + E
    """
    n = A.shape[0]
    L = np.zeros_like(A)
    E = np.zeros_like(A)
    mu =  1e-10  # Small value to avoid division by zero
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:  # Diagonal elements
                if A[i][i] - sum_k > mu:
                    L[i][j] = np.sqrt(A[i][i] - sum_k)
                else:
                    L[i][j] = np.sqrt(mu)
                    E[i][i] = A[i][i] - sum_k
            else:
                if L[j][j] != 0:
                    L[i][j] = (A[i][j] - sum_k) / L[j][j]
                else:
                    E[i][j] = A[i][j] - sum_k
    
    return L, E

# Example usage
A = np.array([[4, 12, -16], 
              [12, 37, -43], 
              [-16, -43, 98]])

L, E = modified_cholesky_decomposition(A)
print("Lower triangular matrix L:\n", L)
print("Error matrix E:\n", E)
print("Reconstructed matrix A:\n", L @ L.T + E)
