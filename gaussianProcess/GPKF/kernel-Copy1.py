import numpy as np
from numpy.linalg import multi_dot
from sklearn.gaussian_process.kernels import Matern
import GPy

from abc import ABC, abstractmethod

def get_psd(kernel):
    """
    Retrieve numerator and denominator for power spectral density representation of kernel. 
    """
    type = kernel.__class__.__name__
    scale = kernel.lengthscale[0]
    std = kernel.variance[0]
    
    if type == 'Exponential':
        num = np.array([np.sqrt(2*scale / std)])
        den = np.array([1/std])
        
    elif type == 'SineExpSquared':
        frequency = kernel.period[0]
        num = np.array([np.sqrt(2*scale / std) * np.array([np.sqrt((1/std)**2 + (2*np.pi*frequency)**2) , 1])])         
        den = np.array([((1/std)**2 + (2*np.pi*frequency)**2 ), 2/std])
        
    elif type =='RBF':
        raise NotImplementedError("RBF approximation not yet implemented")
        
    elif type =='Matern32':
        lam = np.sqrt(3.0)/scale
        num = np.array([np.sqrt(12.0 * 3.0**0.5/ scale **3.0 * std)])
        den = np.array([lam ** 2, 2*lam])
        print("new exp:")
        print(np.exp(-1 * lam) * (1 * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2)))
        
    elif type =='Matern52':
        lam = np.sqrt(5.0)/scale
        num = np.array([np.sqrt(std*400.0*np.sqrt(5.0)/3.0/scale**5)])
        den = np.array([lam ** 3, 3.0*lam**2, 2*lam])
        
    else:
        raise NotImplementedError("Unknown type of kernel. Might not be implemented")

    return num, den

def sample_kernel(kernel, X1, X2):
    """
    Wrapper for sampling kernels. Useful if you want to change where you get your kernels from (e.g. sklearn vs GPy)
    Currently implemented to sample GPy kernels
    """
    dist = kernel._scaled_dist(X1, X2)
    return kernel.K_of_r(dist)


class KernelFactory(object):
    """
    Kernel factory instantiating the desired class
    """
    def get_kernel(self, type, params):
        if type == 'exponential':
            return ExponentialKernel(params)
        elif type == 'periodic':
            return PeriodicKernel(params)
        elif type =='gaussian':
            return GaussianKernel(params)
        elif type =='matern32':
            return Matern32Kernel(params)
        else:
            raise NotImplementedError("Unknown type of kernel. Might not be implemented")

class Kernel(ABC):
    """
    Abstract baseclass for kernels.
    """
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)
            
    def get_params(self, deep = True):
        """
        Get parameters of kernel. If deep = True, also return parameters for contained kernels
        
        Output: Dict of parameter names mapped to their values
        """
        params = dict()
        
        
    @abstractmethod
    def kernelFunction(self):
         raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")
            
    def get_psd(self):
        raise NotImplementedError("No default kernel. Please specify a type (exponential, periodic etc.)")

    def sampled(self, input_set_1, input_set_2):
        """
        kernelSampled returns sampled kernel
           K = kernelSampled(kernel_function, *args) returns the kernel 
           (given as input function) sampled across the desired input set
           Consistency among kernel function and input sets must be priorly ensure
           by the user
        """

        # cardinalities of input sets
        n_input1 = input_set_1.shape[0]
        n_input2 = input_set_2.shape[0]

        # initialize
        K = np.zeros((n_input1, n_input2))

        for i in np.arange(0, n_input1):
            for j in np.arange(0, n_input2):
                #todo: fix [i,:]
                K[i,j] = self.kernelFunction(input_set_1[i] , input_set_2[j])
        
        return K

    def spaceTimeSampled(self, kernel_space, kernel_time, space_locs1, space_locs2, time_instants1, time_instants2, param):
        """
        K = kernelSpaceTimeSampled(space_locs, time_instants, kernel_param) build the 
        space and time kernels, sample them in the desired set of input 
        locations and returns the kernel (given as input function) sampled 
        across the desired input set.
        Consistency among kernel function and input sets must be priorly ensure
        by the user

        This way of sampling the space-time kernel is just one possibility. It
        would be possible to directly use the function kernelSampled by properly
        specifying the input sets. However, this implementation is more efficient
        """
        # sampled kernels
        Ks = kernel_space.sampled(space_locs1, space_locs2)
        Kt = kernel_time.sampled(time_instants1, time_instants2)

        # overall space-time kernel
        K = np.kron(Kt,Ks)

        return K
    
    def __str__(self):
        return str(vars(self))
    
class Matern32Kernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(self, x1, x2):
        print('Matern func not implemented yet. Only PSD')
        return None # No need to implement yet
    
    def get_psd(self):
        lam = np.sqrt(3.0)/self.scale
        num = np.array([np.sqrt(12.0 * 3.0**0.5/ self.scale **3.0 * self.std)])
        den = np.array([lam ** 2, 2*lam])
        
        return num, den
    
class ExponentialKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(self, x1, x2):
        return self.scale * np.exp(-np.linalg.norm(x1-x2) / self.std)
    
    def get_psd(self):
        num = np.array([np.sqrt(2*self.scale / self.std)])
        den = np.array([1/self.std])
        
        return num, den
    
    
class PeriodicKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(x1,x2):
        return self.scale * np.cos(2*pi*self.frequency * np.linalg.norm(x1-x2)) * np.exp(-np.linalg.norm(x1-x2)/self.std)
    
    def get_psd(self):
        num = np.array([np.sqrt(2*self.scale / self.std) * np.array([np.sqrt((1/self.std)**2 + (2*np.pi*self.frequency)**2) , 1])])         
        den = np.array([((1/self.std)**2 + (2*np.pi*self.frequency)**2 ), 2/self.std])
        
        return num, den
    
    
class GaussianKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(self, x1, x2):
        return self.scale * (np.exp(-np.linalg.norm(x1-x2)**2 / (2*self.std**2)))

"""
class SeparableKernel(Kernel):
    def __init__(self, params):
        super().__init__(self, params)
        #self.ks = 
        
    def kernelFunction(self, x1, x2):
        ks = kernelFunction(params['space']['type'], params['space']['scale'], params['space']['std'])
        kt = kernelFunction(params['time']['type'], params['time']['scale'], params['time']['std'])
        
        return np.matmul(kt(x1[0],x2[0]), ks(x1[1:end],x2[1:end]))
"""  