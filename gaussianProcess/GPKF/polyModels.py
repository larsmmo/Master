import numpy as np

class ARXSS:
    def __init__(self, na, nb, nk, n_endog, n_exog, init = None, var = 0, adaptive = True, ID = 0):
        self.na = na # Order of autoregressive model
        self.nb = nb # Zeros
        self.nk = nk # Delay of inputs
        self.init = init
        self.var = var
        self.adaptive = adaptive
        
        self.n_exog = n_exog
        self.n_endog = n_endog
        
        if nb > 0:
            self.exogenous = True
        else:
            self.exogenous = False
        
    def measurement_matrix(self, time, endog, exog):
        """
        Yt = lambda t: endog[(t-1-self.nk)::-1, :] if t == self.na+self.nk else endog[(t-1-self.nk):(t-1-self.na):-1, :]
        Ut = lambda t: exog[(t-1)::-1, :] if t == self.nb+self.nk else exog[(t-1):(t-1-self.nb-self.nk):-1, :]
        Y = Yt(time).append(Ut(time)).reshape(1, -1)
        Cn = np.kron(np.eye(endog.shape[1]), Y.T)
        """
        max_lag = self.get_init_time()
        
        n_endog = endog.shape[0]
        n_exog = exog.shape[0]
        
        Cn = np.zeros(self.na*n_endog + self.nb*n_exog)
        Cn[0 : self.na*n_endog] = endog[:,time + max_lag - 1::-1][:,0:self.na].reshape(1,-1)
        Cn[self.na*n_endog : self.na*n_endog + self.nb*n_exog] = exog[:,max_lag + time - 1::-1][:,self.nk:self.nb + self.nk].reshape(1,-1)
        
        Cn = np.kron(np.eye(n_endog), Cn)
        
        return  Cn
    
    def createDiscreteTimeSys(self,Ts = 1.0):
        """
        See p118 of "System identification - theory for the user" by Ljung for clear equations
        """
        if self.init is None:
            self.init = np.zeros(self.na+self.nb)
        
        if self.adaptive:
            dim_x = self.na * (self.n_endog) + self.nb*(self.n_exog) # State variables
            A = np.eye(dim_x)
            C = np.zeros(dim_x)
            V = np.eye(dim_x)
            B = np.zeros(dim_x)
            #Q = np.eye(dim_x)
            Q = np.eye(dim_x) * self.var
        else:
            A = np.zeros((self.na,self.na))
            A[:,0] = self.init[:self.na]
            Ia = np.eye(self.na-1)
            A[:self.na-1,1:] = Ia
            
            B = np.zeros((self.na,1))
            B[:self.nb] = self.init[self.na : self.na+self.nb]
            
            C = np.eye(1, self.na)[0]
            
            V = np.ones((self.na,self.na)) * 10000
            Q = np.zeros(self.na)
        
        
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.V = V
        
        return A, B, C, V, Q
    
    def updateDiscreteTimeSys(self,time):
        
        self.C = measurement_matrix(time)

        #self.A[:,0] = params[0:self.na]
        #self.B[:self.nb] = params[self.na:self.nb]
        
        return self.A, self.B, self.C
    
    def get_init_time(self):
        """
        Get first time instant for kalman iteration (largest delay required)
        """
        return max(self.na, self.nb+self.nk)
    
    def sample(self, x1):
        """
        Placeholder. Can't sample ARX model (todo)
        """
        return np.array([[0]])