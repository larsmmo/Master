import numpy as np
from numpy.linalg import multi_dot
from sklearn.gaussian_process.kernels import Matern
from scipy.linalg import solve_continuous_lyapunov, block_diag, expm
from scipy import special as special
import GPy

from abc import ABC, abstractmethod

class Kernel(ABC):
    """
    Abstract basececlass for kernels. Implemented as a wapper around GPy kernels.
    """
    def __init__(self, kernel):
        self.kernel = kernel   # GPy kernel
        self.lengthscale = self.kernel.lengthscale[0]
        self.variance = self.kernel.variance[0]
        #for parameter, value in parameters.items():
        #   setattr(self, parameter, value)
        
    """def params(self, deep = True):
        return {'kernel':self.kernel}"""
           
    def get_hyperparams(self, deep = True):
        """
        Get parameters of kernel. If deep = True, also return parameters for contained kernels
        
        Output: Dict of parameter names mapped to their values
        """
        return self.kernel[:]
        
    def set_hyperparams(self, hyperparams):
        self.variance = hyperparams[0]
        self.lengthscale = hyperparams[1]
        
        self.kernel[:] = hyperparams
    
    @abstractmethod
    def psd(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
        
    @abstractmethod
    def state_transition(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
        
    def state_matrix(self):
        """
        Creates a companion-form state-space matrix from numerator and denumerator coefficients
        of the spectral density function of a kernel.
        
        Outputs:
            F: state-space matrix
            state_dim: number of states
        """
        num_coeff, den_coeff = self.psd()
        state_dim  = np.max(den_coeff.shape)
        
        if state_dim == 1:
            F = -den_coeff
        else:
            F = np.diag(np.ones((1,state_dim-1)).flatten(), 1).copy()  # Hopefully there is a better way to do this
            F[state_dim-1, :] = -den_coeff
        
        return F, state_dim
        
    def createDiscreteTimeSys(self, Ts = 1.0):
        """
        Builds the discrete time state-space system in canonical form, using numerator and
        denominator coefficients of the companion form

        INPUT:  Sampling time Ts for the discretization

        OUTPUT: matrix A,C,V,Q: state matrix, output matrix,
                state variance matrix (solution of lyapunov equation), 
                and measurement variance matrix
        """
        
        # state dimension
        num_coeff, den_coeff = self.psd()

        F, state_dim = self.state_matrix()          # State matrix
        A = self.state_transition(Ts)               # State transition matrix = expm(F*Ts)
        G = np.eye(1, state_dim, k = state_dim-1).T # Input matrix
        
        # output matrix
        C = np.zeros(state_dim)
        C[0:np.max(num_coeff.shape)] = num_coeff

        # state variance as solution of the lyapunov equation
        V = solve_continuous_lyapunov(F, -np.matmul(G, G.conj().T))

        # discretization of the noise matrix
        Q = np.zeros(state_dim)
        Ns = 10000        
        t = Ts/Ns
        if state_dim == 1:
            for n in np.arange(t, Ts+t, step=t):
                Q = Q + t * np.exp(F*n) * np.dot(G,G.conj().T) * np.exp(F*n).conj().T
        else:
            for n in np.arange(t, Ts+t, step=t):
                #Q = Q + np.linalg.multi_dot([t, expm(np.dot(F,n)), np.dot(G,G.conj().T), expm(np.dot(F,n)).conj().T])
                Fn = self.state_transition(n)   # REMEMBER TO WRITE ABOUT THIS OPTIMIZATION
                Q = Q + t * Fn @ (G @ G.conj().T) @ Fn.conj().T

        return A, C, V, Q
    
    
    
    def sample(self, X1, X2 = None):
        """
           Returns the kernel sampled across the desired input set
        """
        K = self.kernel.K(X1, X2)   # .K = GPy kernel sampling, see their documentation
        if X2 is None:
            K[np.diag_indices_from(K)] += 1e-5 # Add epsilon to diagonal for numerical stability (helps ensure positive-definiteness)
        
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
    
    def __mul__(self, other):
        """
        Add another kernel to this kernel
        INPUT:
            other: the other kernel to add
        OOUTPUT:
            combined kernel
        """
        assert isinstance(other, Kernel), "Can only add other kernels to a kernel..."
        return ProductKernels([self, other])
    
    def __str__(self):
        return str(vars(self))
    
    def to_dict(self):
        return vars(self)
    
    
class CombinationKernel(Kernel):
    def __init__(self, kernels):
        self.parts = kernels
        
    def set_hyperparams(self, hyperparams):
        self.kernel[:] = hyperparams
        i = 0
        for idx, kern in enumerate(self.parts):
            kern_params_n = len(kern.kernel[:])
            kern.set_hyperparams(hyperparams[i : i + kern_params_n])
            i += kern_params_n
            
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
        
    def psd(self):
        raise NotImplementedError("Getting psd of added kernels is not implemented. Try getting psd for each separate kernel instead...")
        
    def createDiscreteTimeSys(self, Ts = 1.0):
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
    
    def state_transition():
        raise NotImplementedError("Attempting to get state transition matrix of added kernels not implemented. Try getting it for each separate kernel instead...")
       
    
class ProductKernels(CombinationKernel):
    def __init__(self, kernels):
        subkerns = []
        for kern in kernels:
            if isinstance(kern, Kernel):
                subkerns.append(kern)
        
        super(ProductKernels, self).__init__(subkerns)
        self.kernel = kernels[0].kernel * kernels[1].kernel
        
    def psd(self):
        raise NotImplementedError("Attempting to get psd of multiplied kernels is not implemented. Try getting psd for each separate kernel instead...")
        
    def createDiscreteTimeSys(self, Ts = 1.0):
        A = np.array((0,), ndmin=2)
        C = np.array((1,), ndmin=1)
        V = np.array((1,), ndmin=2)
        Q = np.array((1,), ndmin=2)
        
        for part in self.parts:
            At, Ct, Vt, Qt = part.createDiscreteTimeSys(Ts)
            
            #if part.__class__.__name__ == 'CosineKernel':
            
            A = np.kron(A, np.eye(At.shape[0])) + np.kron(np.eye(A.shape[0]), At)
            C = np.kron(C, Ct)
            V = np.kron(V, Vt)
            Q = np.kron(Q, Qt)
        
        return A, C, V, Q
    
    def state_matrix():
        """
        Computes the state matrix as the kronecker sum between multiplied kernels
        Output:
            F: State matrix
        """
        F = np.array((0,), ndmin=2)
        for part in self.parts:
            Ft = part.state_matrix()
            F = np.kron(F, np.eye(Ft.shape[0])) + np.kron(np.eye(F.shape[0]), Ft)
        
        return F
    
    def state_transition():
        raise NotImplementedError("Attempting to get state transition matrix of multiplied kernels is not implemented. Try getting it for each separate kernel instead...")
    
    
class Exponential(Kernel):
    def __init__(self, input_dim, variance, lengthscale):
        super().__init__(GPy.kern.Exponential(input_dim = input_dim, lengthscale = lengthscale, variance = variance))
    
    def psd(self):
        num = np.array([np.sqrt(2*self.variance / self.lengthscale)])
        den = np.array([1/self.lengthscale])
        
        return num, den
    
    def state_transition(self, Ts):
        return np.broadcast_to(np.exp(-Ts/self.lengthscale), [1,1])
    
    
class Matern32(Kernel):
    def __init__(self, input_dim,  variance, lengthscale, active_dims = None, ARD=False):
        super().__init__(GPy.kern.Matern32(input_dim = input_dim, active_dims = active_dims, lengthscale = lengthscale, variance = variance, ARD = ARD))
    
    def psd(self):
        lam = np.sqrt(3.0)/self.lengthscale
        num = np.array([np.sqrt(12.0 * 3.0**0.5/ self.lengthscale **3.0 * self.variance)])
        den = np.array([lam ** 2, 2*lam])
        
        return num, den
    
    def state_transition(self, Ts):
        lam = np.sqrt(3.0)/self.lengthscale
        return np.exp(-Ts * lam) * (Ts * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
    
    
class Matern52(Kernel):
    def __init__(self, input_dim, variance, lengthscale, active_dims = None, ARD=False):
        super().__init__(GPy.kern.Matern52(input_dim = input_dim, active_dims = active_dims, lengthscale = lengthscale, variance = variance, ARD = ARD))
    
    def psd(self):
        lam = np.sqrt(5.0)/self.lengthscale
        num = np.array([np.sqrt(self.variance * 400.0 * 5.0**0.5/ 3.0 / self.lengthscale **5.0)])
        den = np.array([lam ** 3.0, 3.0*lam**2.0, 3.0*lam])
        
        return num, den
    
    def state_transition(self, Ts):
        lam = np.sqrt(5.0)/self.lengthscale
        TsLam = Ts * lam
        return np.exp(-TsLam) \
            * (Ts * np.array([[lam * (0.5 * TsLam + 1.0),      TsLam + 1.0,            0.5 * Ts],
                              [-0.5 * TsLam * lam ** 2,        lam * (1.0 - TsLam),    1.0 - 0.5 * TsLam],
                              [lam ** 3 * (0.5 * TsLam - 1.0), lam ** 2 * (TsLam - 3), lam * (0.5 * TsLam - 2.0)]])
               + np.eye(3))

    
class Periodic(Kernel):
    def __init__(self, input_dim, variance, lengthscale, period):
        super().__init__(GPy.kern.StdPeriodic(input_dim=input_dim, variance = variance, lengthscale=lengthscale, period = period))
        self.variance = variance
        self.lengthscale = lengthscale
        self.freq= period
        
    def set_hyperparams(self, hyperparams):
        self.variance = hyperparams[0]
        self.freq = hyperparams[1]
        self.lengthscale = hyperparams[2]
        
        self.kernel[:] = hyperparams
    
    def psd(self):
        num = np.sqrt(2*self.variance / self.lengthscale) * np.array([np.sqrt((1/self.lengthscale)**2 + (2*np.pi/self.freq)**2) , 1])       
        den = np.array([((1/self.lengthscale)**2 + (2*np.pi/self.freq)**2 ), 2/self.lengthscale])
        
        return num, den
    
    def state_transition(self, Ts):
        num, den = self.psd()
        F = np.diag(np.ones((1,1)).flatten(),1).copy()
        F[1, :] = -den
        return expm(F * Ts)

class CanonicalPeriodic(Kernel):
    """
    Periodic state-space representation (Inspired by GPy class: sde_standard_periodic)
    
    Reference:
    [1] Arno Solin and Simo Särkkä (2014). Explicit link between periodic 
        covariance functions and state space models. In Proceedings of the 
        Seventeenth International Conference on Artifcial Intelligence and 
        Statistics (AISTATS 2014). JMLR: W&CP, volume 33.  
    """
    def __init__(self, input_dim, variance, lengthscale, period, order = 6):
        super().__init__(GPy.kern.StdPeriodic(input_dim=input_dim, variance = variance, lengthscale=lengthscale, period = period))
        self.variance = variance
        self.lengthscale = lengthscale
        self.freq= period
        self.order = order
        
    def set_hyperparams(self, hyperparams):
        self.variance = hyperparams[0]
        self.freq = hyperparams[1]
        self.lengthscale = hyperparams[2]
        
        self.kernel[:] = hyperparams
        
    def createDiscreteTimeSys(self, Ts = 1.0):

        F, state_dim = self.state_matrix()          # State matrix
        A = self.state_transition(Ts)               # State transition matrix 
        G = np.eye(1, state_dim, k = state_dim-1).T # Input matrix
        
        # output matrix
        C = np.kron(np.ones((1, self.order +1)),np.array((1,0)))[0]
        
        # Coefficients
        q2 = 2*self.variance*np.exp(-(self.lengthscale*2)**(-2) ) * special.iv(range(0,self.order+1),1.0/(self.lengthscale*2)**(2))
        q2[0] = 0.5*q2[0]
        
        #V = solve_continuous_lyapunov(F, -np.matmul(G, G.conj().T))
        V = np.kron(np.diag(q2), np.eye(2))
        
        Q = np.zeros((state_dim, state_dim))

        return A, C, V, Q
        
    def state_matrix(self):
        w0 = 2*np.pi/self.freq
        F = np.kron(np.diag(range(0,self.order+1)),np.array(((0, -w0), (w0, 0))))
        state_dim = 2*(self.order+1)
        return F, state_dim
    
    def state_transition(self, Ts = 1.0):
        w0 = 2*np.pi/self.freq
        harmonics = np.arange(self.order + 1) * w0
        rot = np.zeros((len(harmonics),2,2))
        
        for i,h in enumerate(harmonics):
            rot[i] = np.array([[np.cos(h * Ts), -np.sin(h * Ts)],
                                [np.sin(h* Ts),  np.cos(h * Ts)]])

        A = np.block([
            [rot[0], np.zeros([2, 12])],
            [np.zeros([2, 2]),  rot[1], np.zeros([2, 10])],
            [np.zeros([2, 4]),  rot[2], np.zeros([2, 8])],
            [np.zeros([2, 6]),  rot[3], np.zeros([2, 6])],
            [np.zeros([2, 8]),  rot[4], np.zeros([2, 4])],
            [np.zeros([2, 10]), rot[5], np.zeros([2, 2])],
            [np.zeros([2, 12]), rot[6]]
        ])
        return A
    
    def psd(self):
        return 0
    
    
class PeriodicMatern32(Kernel):
    def __init__(self, variance, lengthscale, period):
        super().__init__(Gpy.kern.PeriodicMatern32(input_dim = input_dim, variance = variance, period = period))
        self.freq = freq
        
    def psd(self):
        return 0
    
    def state_transition_matrix(self):
        lam = np.sqrt(3) / self.lengthscale
        F = np.array([[0, -self.freq, 1, 0],
                      [self.freq, 0, 0, 1],
                      []])
    
    
class RBF(Kernel):
    def __init__(self, input_dim, lengthscale, variance, active_dims = None, ARD = False):
        super().__init__(GPy.kern.RBF(input_dim=input_dim, active_dims=active_dims, variance=variance, lengthscale=lengthscale, ARD = ARD))
        
    def kernelFunction(self, x1, x2):
        return self.lengthscale * (np.exp(-np.linalg.norm(x1-x2)**2 / (2*self.variance**2)))
    
    def psd(self):
        return 0
    
    def state_transition(self, Ts):
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