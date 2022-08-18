import numpy as np
import scipy as sp
from IPython.display import clear_output

from .barycentric import compute_barycentric_weights, barycentric

def lanczos_reorth(A,v,k,reorth=0):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    v : starting vector
    k : number of iterations (matvecs)
    reorth : number of iterations to reorthogonalize
    
    Output
    ------
    Q : Lanczos vectors
    α : diagonal coefficients
    β : off diagonal coefficients 
    """
    
    n = A.shape[0]
    
    Q = np.zeros((n,k+1),dtype=A.dtype)
    α = np.zeros(k,dtype=A.dtype)
    β = np.zeros(k,dtype=A.dtype)
    
    Q[:,0] = v / np.linalg.norm(v)
    
    for i in range(k):

        qip1 = A@Q[:,i] - β[i-1]*Q[:,i-1] if i>0 else A@Q[:,i]
        
        α[i] = Q[:,i].conj().T@qip1
        qip1 -= α[i]*Q[:,i]
        
        if reorth>i:
            qip1 -= Q[:,:i-1]@(Q[:,:i-1].conj().T@qip1)
            
        β[i] = np.linalg.norm(qip1)
        Q[:,i+1] = qip1 / β[i]
                
    return Q,(α,β)



def lanczos_block(A,v,k,verbose=False):
    
    if len(v.shape)==1:
        n = v.shape
        α = np.zeros(k,dtype=np.float64)
        β = np.zeros(k,dtype=np.float64)
    else:
        n,bs = v.shape
        α = np.zeros((k,bs),dtype=np.float64)
        β = np.zeros((k,bs),dtype=np.float64)
    
    q = v / np.linalg.norm(v,axis=0)
    
    for i in range(0,k):
        if verbose:
            clear_output(wait=True)            
            print(f'iter={i}')
            
        q__ = np.copy(q)
        q = A@q - β[i-1]*q_ if i>0 else A@q
        q_ = q__
        
        α[i] = np.sum(q_.conj()*q,axis=0)
        q -= α[i]*q_
        
        β[i] = np.linalg.norm(q,axis=0)
        q = q / β[i]
        
    return (α,β)


def lanczos_poly_approx(f,α,β):
    """
    compute degree k Lanczos approximation to f(A)b
    """
    
    theta = sp.linalg.eigvalsh_tridiagonal(α,β[:-1],tol=1e-30)
    
    w = compute_barycentric_weights(theta)
    
    return lambda x: barycentric(x,theta,f(theta),w)


def lanczos_FA(f,Q,α,β,k,normb=1):
    """
    compute Lanczos-FA iterate
    
    Input
    -----
    k : degree of approximation
    
    """
    
    theta,S = sp.linalg.eigh_tridiagonal(α[:k+1],β[:k],tol=1e-30)
    
    return normb*(Q[:,:k+1]@(S@(f(theta)*(S.T[:,0]))))
