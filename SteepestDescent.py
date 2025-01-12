import numpy as np

def steepest_descent(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    Performs the Steepest Descent algorithm to find a local minimum of the function f.
    
    Parameters:
    f       : function to minimize
    grad_f  : gradient of the function
    x0      : initial guess
    tol     : tolerance for stopping criterion
    max_iter: maximum number of iterations
    
    Returns:
    x       : the found local minimum
    """
    x = x0
    for _ in range(max_iter):
        grad = grad_f(x)
        
        # Line search for optimal step size
        alpha = 1
        while f(x - alpha * grad) > f(x) - 0.1 * alpha * np.dot(grad, grad):
            alpha *= 0.5
        
        x_new = x - alpha * grad
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    return x

# Example usage for a quadratic objective function
def f(x):
    return x[0]**2 + 3*x[1]**2 + 2*x[0]*x[1] - 4*x[0] - 6*x[1]

def grad_f(x):
    return np.array([2*x[0] + 2*x[1] - 4, 6*x[1] + 2*x[0] - 6])

x0 = np.array([0, 0])
x_min = steepest_descent(f, grad_f, x0)
print("Found local minimum:", x_min)
