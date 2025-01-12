import numpy as np
from .Cholesky_Decomposition import modified_cholesky_decomposition  # Import the method

def invert_matrix(A):
    """
    Inverts a symmetric matrix A using the modified Cholesky decomposition.
    """
    L, E = modified_cholesky_decomposition(A)
    if np.any(E != 0):
        raise ValueError("Matrix is not invertible due to non-zero error matrix E.")
    
    L_inv = np.linalg.inv(L)
    A_inv = L_inv.T @ L_inv
    return A_inv

def damped_newton(f, grad_f, hess_f, x0, tol=1e-4, max_iter=100, mu=0.1, nu=0.9):  # Set tolerance to 10^-4
    """
    Performs the Damped Newton algorithm to find a local minimum of the function f.
    
    Parameters:
    f       : function to minimize
    grad_f  : gradient of the function
    hess_f  : Hessian of the function
    x0      : initial guess
    tol     : tolerance for stopping criterion
    max_iter: maximum number of iterations
    mu      : Wolfe condition parameter for sufficient decrease
    nu      : Wolfe condition parameter for curvature condition
    
    Returns:
    x       : the found local minimum
    """
    x = x0
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Solve for the Newton direction
        try:
            hess_inv = invert_matrix(hess)
            p = hess_inv @ -grad
        except ValueError as e:
            print(e)
            break
        
        # Line search for damping satisfying Wolfe conditions
        alpha = 1
        while f(x + alpha * p) > f(x) + mu * alpha * np.dot(grad, p) or np.dot(grad_f(x + alpha * p), p) < nu * np.dot(grad, p):
            alpha *= 0.5
        
        x_new = x + alpha * p
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    return x

# Example usage for a 3-dimensional quadratic objective function
def f(x):
    return x[0]**2 + 3*x[1]**2 + 2*x[2]**2 + 2*x[0]*x[1] + 2*x[0]*x[2] + 2*x[1]*x[2] - 4*x[0] - 6*x[1] - 8*x[2]

def grad_f(x):
    return np.array([2*x[0] + 2*x[1] + 2*x[2] - 4, 6*x[1] + 2*x[0] + 2*x[2] - 6, 4*x[2] + 2*x[0] + 2*x[1] - 8])

def hess_f(x):
    return np.array([[2, 2, 2], [2, 6, 2], [2, 2, 4]])

x0 = np.array([0, 0, 0])
x_min = damped_newton(f, grad_f, hess_f, x0)
print("Found local minimum:", x_min)

# Remark: This implementation of the Damped Newton algorithm uses the modified Cholesky decomposition to invert the Hessian matrix.
# For least-squares problems, it is better to choose Gauss-Newton method or Levenberg-Marquardt method where no inversion is needed.