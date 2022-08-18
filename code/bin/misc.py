import numpy as np

def model_problem_spectrum(n,ρ,κ=1e3,dtype='float'):
    '''
    returns spectrum of model problem
    '''
    λ1 = 1
    λn = κ
    Λ = λ1+(λn-λ1)*np.arange(n)/(n-1)*ρ**np.arange(n-1,-1,-1,dtype=dtype)
    
    return Λ


def get_discrete_nodes(L,k,X=None):
    
    Xn = np.zeros(k,dtype=np.longdouble)
    Xn[:] = np.nan
    
    i0 = 0
    if X is not None:
        Xn[:len(X)] = X
        i0 = len(X)
    
    for i in range(i0,k):
        for l in L[::-1,0]:
            if np.all(Xn != l):
                Xn[i] = l
                break
    
    return np.sort(Xn)

def get_cheb_nodes(a,b,k):
    """
    Get k Chebyshev T nodes on [a,b]
    """
    nodes = (a+b)/2 + (b-a)*np.cos((2*np.arange(k,0,-1)-1)/(2*k)*np.pi)/2
    return nodes.astype(np.longdouble)

class mat_samp_full():
    
    def __init__(self,N,d,σ,dist='Gaussian'):
    
        M = int(np.floor(N/d))
        if dist=='Rademacher':
            self.X = (1-2*np.random.randint(2,size=M*N).reshape(N,M))/np.sqrt(M)
        elif dist=='Gaussian':
            self.X = np.random.randn(N,M)/np.sqrt(M)
        N2 = int(np.floor(N/2))
        self.X[:N2,:] *= np.sqrt(σ)
        self.shape = (N,N)
        self.dtype = np.double
    
    def __matmul__(self,v):
        
        return self.X@(self.X.T@v)
    
def support_gen(N,d,σ): # newtons' method to find all roots of df
    
    f = lambda x: -1/x + d/2*1/(x + 1) + d/2*1/(x + 1/σ)
    df = lambda x: 1/x**2 - d/2*1/(x + 1)**2 - d/2*1/(x + 1/σ)**2
    ddf = lambda x: -2/x**3 + d/(x + 1)**3 + d/(x + 1/σ)**3

    x0 = np.arange(-4,4,.005)
    for i in range(400):
        x0 = x0 - df(x0)/ddf(x0)
    x0 = f(x0[np.abs(x0)<100])
    
    return np.reshape(x0[np.unique(x0.round(decimals=10),return_index=True)[1]],(2,2))