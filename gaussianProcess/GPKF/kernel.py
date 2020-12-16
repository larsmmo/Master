import numpy as np
from numpy.linalg import multi_dot
from sklearn.gaussian_process.kernels import Matern
import GPy

from abc import ABC, abstractmethod

class Kernel(ABC):
    """
    Abstract baseclass for kernels.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.lengthscale = self.kernel.lengthscale[0]
        self.variance = self.kernel.variance[0]
            
    def get_params(self, deep = True):
        """
        Get parameters of kernel. If deep = True, also return parameters for contained kernels
        
        Output: Dict of parameter names mapped to their values
        """
        #params = dict()
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            setattr(self.kernel, parameter, value)
        
    @abstractmethod
    def kernelFunction(self):
         raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
    
    @abstractmethod
    def get_psd(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
        
    @abstractmethod
    def get_state_transition(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
    
    def sample(self, X1, X2):
        """
           Returns the kernel sampled across the desired input set
        """
        dist = self.kernel._scaled_dist(X1, X2)
        return self.kernel.K_of_r(dist)
        
    def __str__(self):
        return str(vars(self))
    
class Matern32Kernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(self, x1, x2):
        print('Matern func not implemented yet. Only PSD')
        return None # No need to implement yet
    
    def get_psd(self):
        lam = np.sqrt(3.0)/self.lengthscale
        num = np.array([np.sqrt(12.0 * 3.0**0.5/ self.lengthscale **3.0 * self.variance)])
        den = np.array([lam ** 2, 2*lam])
        
        return num, den
    
    def get_state_transition(self, Ts):
        lam = np.sqrt(3.0)/self.lengthscale
        return np.exp(-Ts * lam) * (Ts * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
    
class ExponentialKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(self, x1, x2):
        return self.lengthscale * np.exp(-np.linalg.norm(x1-x2) / self.variance)
    
    def get_psd(self):
        num = np.array([np.sqrt(2*self.lengthscale / self.variance)])
        den = np.array([1/self.variance])
        
        return num, den
    
    
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
    def __init__(self, params):
        super().__init__(params)
        
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