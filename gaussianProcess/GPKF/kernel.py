import numpy as np
from numpy.linalg import multi_dot
from sklearn.gaussian_process.kernels import Matern
from scipy.linalg import solve_continuous_lyapunov, block_diag
import GPy

from abc import ABC, abstractmethod

class Kernel(ABC):
    """
    Abstract basececlass for kernels. Implemented as a wapper around GPy kernels.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.lengthscale = self.kernel.lengthscale[0]
        self.variance = self.kernel.variance[0]
        #for parameter, value in parameters.items():
        #   setattr(self, parameter, value)
        
    """def get_params(self, deep = True):
        return {'kernel':self.kernel}"""
           
    def get_hyperparams(self, deep = True):
        """
        Get parameters of kernel. If deep = True, also return parameters for contained kernels
        
        Output: Dict of parameter names mapped to their values
        """
        print('Got kernel params')
        return self.kernel[:]
        #params = dict() 
        
    def set_hyperparams(self, hyperparams):
        self.variance = hyperparams[0]
        self.lengthscale = hyperparams[1]
        
        self.kernel[:] = hyperparams
    
    @abstractmethod
    def get_psd(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
        
    @abstractmethod
    def get_state_transition(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
        
    def createDiscreteTimeSys(self, Ts):
        """
        Builds the discrete time state-space system in canonical form, using numerator and
        denominator coefficients of the companion form

        INPUT:  Sampling time Ts for the discretization

        OUTPUT: matrix A,C,V,Q: state matrix, output matrix,
                state variance matrix (solution of lyapunov equation), 
                and measurement variance matrix
        """
        
        # state dimension
        num_coeff, den_coeff = self.get_psd()
        stateDim  = np.max(den_coeff.shape)
        if stateDim ==1:
            F = -den_coeff       # state matrix
            A = np.exp(F * Ts)   # Discretization
            G = np.array([1])    #L
        else:
            F = np.diag(np.ones((1,stateDim-1)).flatten(),1).copy()
            F[stateDim-1, :] = -den_coeff
            A = self.get_state_transition(Ts) #expm(F * Ts)  # state matrix
            G = np.append(np.zeros((stateDim-1,1)),1.0)[np.newaxis].T # input matrix

        # output matrix
        C = np.zeros((1,stateDim))[0]
        C[0:np.max(num_coeff.shape)] = num_coeff

        # state variance as solution of the lyapunov equation
        V = solve_continuous_lyapunov(F, -np.matmul(G, G.conj().T))

        # discretization of the noise matrix
        Q = np.zeros(stateDim)
        Ns = 10000        
        t = Ts/Ns
        if stateDim == 1:
            for n in np.arange(t, Ts+t, step=t):
                Q = Q + t * np.exp(F * n) * np.dot(G,G.conj().T) * np.exp(F * n).conj().T
        else:
            for n in np.arange(t, Ts+t, step=t):
                #Q = Q + np.linalg.multi_dot([t, expm(np.dot(F,n)), np.dot(G,G.conj().T), expm(np.dot(F,n)).conj().T])
                #Q = Q + t * expm(F * n) @ (G @G.conj().T) @ (expm(F * n)).conj().T
                Fn = self.get_state_transition(n)   # REMEMBER TO WRITE ABOUT THIS OPTIMIZATION
                Q = Q + t * Fn @ (G @G.conj().T) @ Fn.conj().T

        return A, C, V, Q
    
    def sample(self, X1, X2 = None):
        """
           Returns the kernel sampled across the desired input set
        """
        K = self.kernel.K(X1, X2)
        if X2 is None:
            K[np.diag_indices_from(K)] += 1e-4 # Add epsilon to diagonal for numerical stability (positive definite requred for cholesky)
        
        return K
        
    def __add__(self, other):
        """
        Add another kernel to this kernel
        INPUT:
            other: the other kernel to add
        OOUTPUT:
            combined kernel
        """
        assert isinstance(other, Kernel), "Can only add other kernels to a kernel..."
        return AddedKernels([self, other])
    
    def __str__(self):
        return str(vars(self))
    
    
class CombinationKernel(Kernel):
    def __init__(self, kernels):
        self.parts = kernels
        
    def set_hyperparams(self, hyperparams):
        self.kernel[:] = hyperparams
        for idx, kern in enumerate(self.parts):
            kern.set_params(kern.kernel[:])
            
    def get_hyperparams(self):
        params = []
        for kern in self.parts:
            params.append(kern.get_hyperparams()[0])
        
    
class AddedKernels(CombinationKernel):
    def __init__(self, kernels):
        subkerns = []
        for kern in kernels:
            if isinstance(kern, Kernel):
                subkerns.append(kern)
        
        super(AddedKernels, self).__init__(subkerns)
        self.kernel = kernels[0].kernel + kernels[1].kernel
        
    def get_psd(self):
        raise NotImplementedError("Attempting to get psd of added kernels not implemented. Try getting psd for each separate kernel instead...")
        
    def createDiscreteTimeSys(self, Ts):
        A = None
        C = None
        V = None
        Q = None
        
        for part in self.parts:
            At, Ct, Vt, Qt = part.createDiscreteTimeSys(Ts)
            
            A = block_diag(A, At) if (A is not None) else At
            C = np.hstack((C, Ct)) if (C is not None) else Ct
            V = block_diag(V, Vt) if (V is not None) else Vt
            Q = block_diag(Q, Qt) if (Q is not None) else Qt
        
        return A, C, V, Q
    
    def get_state_transition():
        raise NotImplementedError("Attempting to get state transition matrix of added kernels not implemented. Try getting it for each separate kernel instead...")
            
    
class Matern32Kernel(Kernel):
    def __init__(self, input_dim, variance, lengthscale, active_dims = None, ARD=False):
        super().__init__(GPy.kern.Matern32(input_dim = input_dim, active_dims = active_dims, lengthscale = lengthscale, variance = variance, ARD = ARD))
    
    def get_psd(self):
        lam = np.sqrt(3.0)/self.lengthscale
        num = np.array([np.sqrt(12.0 * 3.0**0.5/ self.lengthscale **3.0 * self.variance)])
        den = np.array([lam ** 2, 2*lam])
        
        return num, den
    
    def get_state_transition(self, Ts):
        lam = np.sqrt(3.0)/self.lengthscale
        return np.exp(-Ts * lam) * (Ts * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
    
class ExponentialKernel(Kernel):
    def __init__(self, input_dim, variance, lengthscale):
        super().__init__(GPy.kern.Exponential(input_dim = input_dim, lengthscale = lengthscale, variance = variance))
    
    def get_psd(self):
        num = np.array([np.sqrt(2*self.variance / self.lengthscale)])
        den = np.array([1/self.lengthscale])
        
        return num, den
    
    def get_state_transition(self, Ts):
        return np.broadcast_to(np.exp(-Ts/self.lengthscale), [1,1])
    
    
class PeriodicKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        self.frequency = self.kernel.period[0]
        
    def kernelFunction(x1,x2):
        return self.lengthscale * np.cos(2*pi*self.frequency * np.linalg.norm(x1-x2)) * np.exp(-np.linalg.norm(x1-x2)/self.variance)
    
    def get_psd(self):
        num = np.array([np.sqrt(2*self.lengthscale / self.variance) * np.array([np.sqrt((1/self.variance)**2 + (2*np.pi*self.frequency)**2) , 1])])         
        den = np.array([((1/self.variance)**2 + (2*np.pi*self.frequency)**2 ), 2/self.variance])
        
        return num, den
    
    
class GaussianKernel(Kernel):
    def __init__(self, input_dim, lengthscale, variance):
        super().__init__(GPy.kern.RBF(input_dim=input_dim, variance=variance, lengthscale=lengthscale))
        
    def kernelFunction(self, x1, x2):
        return self.lengthscale * (np.exp(-np.linalg.norm(x1-x2)**2 / (2*self.variance**2)))
    
    def get_psd(self):
        return 0
    
    def get_state_transition(self, Ts):
        return 0

"""
class SeparableKernel(Kernel):
    def __init__(self, params):
        super().__init__(self, params)
        #self.ks = 
        
    def kernelFunction(self, x1, x2):
        ks = kernelFunction(params['space']['type'], params['space']['scale'], params['space']['variance'])
        kt = kernelFunction(params['time']['type'], params['time']['scale'], params['time']['variance'])
        
        return np.matmul(kt(x1[0],x2[0]), ks(x1[1:end],x2[1:end]))
"""  