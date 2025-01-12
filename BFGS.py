import numpy as np

def wolfe_conditions(f, grad_f, xk, pk, grad_fk, alpha, mu, eta):
    # Implements the Armijo and Curvature conditions
    f_new = f(xk + alpha * pk)
    armijo = f_new <= f(xk) + mu * alpha * np.dot(grad_fk, pk)
    grad_f_new = grad_f(xk + alpha * pk)
    curvature = np.dot(grad_f_new, pk) >= eta * np.dot(grad_fk, pk)
    return armijo and curvature

def line_search(f, grad_f, xk, pk, grad_fk, mu=1e-4, eta=0.1, max_iterations=100):
    # Implements a backtracking line search using the Wolfe conditions.
    alpha = 1.0
    i = 0
    while i < max_iterations:
        if wolfe_conditions(f, grad_f, xk, pk, grad_fk, alpha, mu, eta):
            return alpha
        alpha /= 2.0
        i += 1
    return alpha

def bfgs(f, grad_f, x0, tol=1e-6, mu=1e-4, eta=0.1, max_iterations=1000):
    # Implements the BFGS algorithm.
    n = len(x0)
    Bk_inv = np.eye(n)  # Initial inverse Hessian approximation
    xk = x0
    k = 0

    while k < max_iterations:
        grad_fk = grad_f(xk)
        if np.linalg.norm(grad_fk) < tol:
            break

        pk = -np.dot(Bk_inv, grad_fk)  # Search direction
        alpha_k = line_search(f, grad_f, xk, pk, grad_fk, mu, eta)  # Step size using line search

        xk_new = xk + alpha_k * pk  # Update solution
        grad_fk_new = grad_f(xk_new)

        sk = xk_new - xk  # compute sk
        yk = grad_fk_new - grad_fk  # compute yk

        rho_k = 1.0 / np.dot(yk, sk)  # compute rho_k

        # Update the inverse Hessian approximation
        Bk_inv = (np.eye(n) - rho_k * np.outer(sk, yk)) @ Bk_inv @ (np.eye(n) - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)
        xk = xk_new
        k += 1

    return xk, f(xk), k

# Example usage for the Beale function
def beale_function(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

def grad_beale_function(x):
    df_dx0 = 2 * (1.5 - x[0] + x[0] * x[1]) * (-1 + x[1]) + 2 * (2.25 - x[0] + x[0] * x[1]**2) * (-1 + x[1]**2) + 2 * (2.625 - x[0] + x[0] * x[1]**3) * (-1 + x[1]**3)
    df_dx1 = 2 * (1.5 - x[0] + x[0] * x[1]) * x[0] + 2 * (2.25 - x[0] + x[0] * x[1]**2) * 2 * x[0] * x[1] + 2 * (2.625 - x[0] + x[0] * x[1]**3) * 3 * x[0] * x[1]**2
    return np.array([df_dx0, df_dx1])

x0 = np.array([3, -1], dtype=float)
x_min, f_min, iterations = bfgs(beale_function, grad_beale_function, x0)
print(f"Minimum found at x = {x_min}, f(x) = {f_min}, after {iterations} iterations")