import numpy as np
import scipy as sp
from scipy import integrate
from .barycentric import compute_barycentric_weights, barycentric

def opt_poly_approx(f,A,b,k,B=None):
    n = len(A)

    unif_ones = np.ones(n,dtype=np.longdouble) / np.sqrt(np.ones(n)*B@np.ones(n))

    Q,(a_,b_) = exact_lanczos(A,unif_ones,k+1,B)

    p_opt = lambda x: polyval_A_equil(Q[:,:k+1].T@(B*f(A)),a_[:k+1],b_[:k+1],x) * Q[0,0]

    return p_opt

def opt_FA(fAb,Q,B=None,normb=1):
    
    if B is None:
        B = np.ones(len(A))
        
    return Q@Q.T@(B*fAb)


def opt_fAb(f,A,b,k,B=None):
    """
    get optimal p_k(A)b over K_k(A,b) in B norm (B has same eigenvalues as A)
    """
    
    if B is None:
        B = np.ones(len(A),dtype=np.longdouble)
        
    Q,(a_,b_) = exact_lanczos(A,b,k+1,B)

    return Q[:,:k+1]@(Q[:,:k+1].T@(B*f(A)*b))

def Q_wz(w,z,lmin,lmax):
    """
    max_{x\in[lmin,lmax]} |x-w|/|z-w|
    """
    
    if np.real(z) - w != 0:
        b_hat = ( np.abs(z)**2 - np.real(z)*w ) / (np.real(z) - w)
    else:
        b_hat = np.inf
        
    if lmin < b_hat <= lmax:
        return np.abs((z-w)/np.imag(z))
    else:
        return np.max([np.abs((lmax-w)/(lmax-z)), np.abs((lmin-w)/(lmin-z))])
    
def Q_z(z,lmin,lmax):
    """
    max_{x\in[lmin,lmax]} 1/|z-w|
    """
    
    b_hat = np.real(z)
        
    if lmin < b_hat <= lmax:
        return np.abs(1/np.imag(z))
    
    else:
        return np.max([np.abs(1/(lmax-z)), np.abs(1/(lmin-z))])
    
def get_a_priori_bound(f,gamma,endpts,k,w,lmin,lmax,epsabs=0,limit=200):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| (Q_{w,z})^{k+1} |dz|
    """
    
    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * Q_wz(w,z,lmin,lmax)**(k+1) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=epsabs,limit=limit) 
    
    return integral


def get_a_posteriori_bound(f,gamma,endpts,a_,b_,w,lmin,lmax,epsabs=0,limit=200):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| |D_{k,w,z}| Q_{w,z} |dz|
    """
    
    if len(a_)>0:
        theta = sp.linalg.eigvalsh_tridiagonal(a_,b_,tol=1e-30)
    else:
        theta = np.zeros(0)
        
    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * np.abs(np.prod((theta-w)/(theta-z))) * Q_wz(w,z,lmin,lmax) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=epsabs,limit=limit) 
    
    return integral

def get_exact_bound(f,gamma,endpts,a_,b_,w,lam,epsabs=0,limit=200):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| (Q_{w,z})^{k+1} |dz|
    """
    
    if len(a_)>0:
        theta = sp.linalg.eigvalsh_tridiagonal(a_,b_,tol=1e-30)
    else:
        theta = np.zeros(0)
        
    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * np.abs(np.prod((theta-w)/(theta-z))) * np.max(np.abs((lam-w)/(lam-z))) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=epsabs,limit=limit) 
    
    return integral

def get_a_priori_bound_Qz(f,gamma,endpts,k,w,lminl,lmaxl,lminr,lmaxr):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| (Q_{w,z})^{k+1} |dz|
    """
    
    def F(t):
        z,dz = gamma(t)
                
        return (1/(2*np.pi)) * np.abs(f(z)) * Q_wz(w,z,lminl,lmaxr)**(2*k) * np.max([Q_z(z,lminl,lmaxl),Q_z(z,lminr,lmaxr)]) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200) 
    
    return integral

def get_a_posteriori_bound_Qz(f,gamma,endpts,a_,b_,w,lminl,lmaxl,lminr,lmaxr):
    """
    (1/2pi) \oint_{\Gamma} |f(z)| |D_{k,z}| Q_{w,z} |dz|
    """
    
    theta = sp.linalg.eigvalsh_tridiagonal(a_,b_,tol=1e-30)

    def F(t):
        z,dz = gamma(t)
        
        return (1/(2*np.pi)) * np.abs(f(z)) * np.abs(np.prod((theta-w)/(theta-z)))**2 * np.max([Q_z(z,lminl,lmaxl),Q_z(z,lminr,lmaxr)]) * np.abs(dz)
    
    integral = sp.integrate.quad(F,endpts[0],endpts[1],epsabs=0,limit=200) 
    
    return integral