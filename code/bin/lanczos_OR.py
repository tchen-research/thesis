import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# jj = j+q


class streaming_tridiag_square():

    def __init__(self,k):
        
        self.T = np.zeros((2,k))
        self.Tp2 = np.zeros((3,k))
        self.iter = 0
        
    def read_coeff(self,alpha_j,beta_j):

        j = self.iter; T = self.T; Tp2 = self.Tp2
        
        T[0,j] = alpha_j
        T[1,j] = beta_j
        
        if j == 0:
            Tp2[0,j] = T[0,j]**2+T[1,j]**2
        else:
            Tp2[0,j] = T[0,j]**2+T[1,j]**2+T[1,j-1]**2
            Tp2[1,j-1] = (T[0,j]+T[0,j-1])*T[1,j-1]
            Tp2[2,j-1] = T[1,j]*T[1,j-1]

        self.iter += 1
        
class streaming_LDL():
    
    def __init__(self,k,q):
        self.L = np.zeros((q,k))
        self.d =  np.zeros(k)

        self.k = k
        self.q = q
        self.iter = 0
        
    def read_stream(self,N_colj):
        # M_colj is below lower triangle, starting at diagonal
        
        L = self.L; d = self.d; k = self.k; q = self.q; jj = self.iter
                
        d[jj] = N_colj[jj-jj] - sum([L[jj-l-1,l]**2*d[l] for l in range(max(0,jj-q),jj)])

        for i in range(jj+1,min(jj+q+1,k)):
            L[i-jj-1,jj] = (1/d[jj])*(N_colj[i-jj] - sum([L[i-l-1,l]*L[jj-l-1,l]*d[l] for l in range(max(0,i-q),i)]))

        self.iter += 1
        
class streaming_banded_prod():
    
    def __init__(self,n,q,k):

        self.n = n
        self.q = q

        self.k = k
        self.X_ = np.zeros((n,q+1))
        self.y_ = np.zeros(q+1)
        self.out = np.zeros(n)

        self.iter = -1
        
    def read_stream(self,q_jj,l_jj,d_jj,yin):

        jj = self.iter; q = self.q; X_ = self.X_; y_ = self.y_

        if jj==-1:
            self.X_[:,:q] = q_jj
            
        else:
            if jj==0:
                self.y_[:] = yin # no need to deal with y

            self.out += (y_[0]/d_jj)*self.X_[:,0]

            # this is unecessary the last step, but li will be zero so its fine..
            y_[:q] = y_[1:] + y_[0]*l_jj
            y_[-1] = 0
            X_[:,-1] = q_jj
            X_[:,:q] = X_[:,1:] + np.outer(X_[:,0],l_jj)

        self.iter += 1

class streaming_banded_inv():
    
    def __init__(self,n,k,q):
       
        self.n = n
        self.k = k
        self.q = q
                
        self.LDL = streaming_LDL(self.k,self.q)
        self.Q_init = np.zeros((n,q))

        self.iter = 0
        
    def read_stream(self,q_j,N_colj,yin):
        
        j = self.iter; q = self.q        
                
        if j<q:
            self.Q_init[:,j] = q_j
            if j==q-1:
                self.b_prod = streaming_banded_prod(self.n,self.q,self.k)
                self.b_prod.read_stream(self.Q_init,None,None,None)
        else:
            self.LDL.read_stream(N_colj)
            self.b_prod.read_stream(q_j,\
                                    -self.LDL.L[:,j-q],\
                                    self.LDL.d[j-q],\
                                    yin) # sometimes vi will be None but that is okay

        self.iter += 1
        
        
def get_matrix_poly(P_coeff,STp2,j):
    """
    return nonzero entries of jth column of aT^2 + bT + cI where T is tridiagonal matrix maintained by STp2
    """
    
    a,b,c = P_coeff
    
    N_colj = np.zeros(3)
    N_colj[:] = a*STp2.Tp2[:,j]
    N_colj[:2] += b*STp2.T[:,j]
    N_colj[0] += c

    return N_colj

class streaming_banded_rational():
    
    def __init__(self,n,k,M_coeff,N_coeff):
       
        self.n = n
        self.k = k
        self.q = 2
        self.M_coeff = M_coeff
        self.N_coeff = N_coeff
        
        self.STp2 = streaming_tridiag_square(k)
        self.b_inv = streaming_banded_inv(n,k,self.q)
        
        self.iter = 0
    
    def read_stream(self,Q_col,alpha,beta):

        j = self.iter; k = self.k; q = self.q
                
        if j<k:
            self.STp2.read_coeff(alpha,beta)

            self.b_inv.read_stream(Q_col,\
                                   get_matrix_poly(self.N_coeff,self.STp2,j-q) if j>=q else None,\
                                   get_matrix_poly(self.M_coeff,self.STp2,0) if j==q else None)  
        
        self.iter +=1
        
    def finish_up(self):
        j = self.iter; k = self.k; q = self.q

        for i in range(k,k+q):
            self.b_inv.read_stream(None,\
                       get_matrix_poly(self.N_coeff,self.STp2,i-q),\
                       None) 

    def __call__(self):
        return self.b_inv.b_prod.out