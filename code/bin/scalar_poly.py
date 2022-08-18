import numpy as np

def lanczos_poly_approx(f,A,b,k):
    """
    compute degree k Lanczos approximation to f(A)b
    """
    
    Q,(a,b) = exact_lanczos(A,b,k+1)
    theta = sp.linalg.eigvalsh_tridiagonal(a,b,tol=1e-30)
    
    w = compute_barycentric_weights(theta)
    
    return lambda x: barycentric(x,theta,f(theta),w)
