import numpy as np
from numpy.linalg import multi_dot

from abc import ABC, abstractmethod

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
        else:
            raise NotImplementedError("Unknown type of kernel. Might not be implemented")

class Kernel(ABC):
    """
    Abstract baseclass for kernels.
    """
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)
        
    @abstractmethod
    def kernelFunction(self):
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

    def spaceTimeSampled(self, space_locs1, space_locs2, time_instants1, time_instants2, param):
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

        # compute kernel functions
        kernel_space = kernelFunction(param['space']['type'], param['space'])
        kernel_time = kernelFunction(param['time']['type'] , param['time'])

        # sampled kernels
        Ks = kernelSampled(space_locs1, space_locs2, kernel_space)
        Kt = kernelSampled(time_instants1, time_instants2, kernel_time)

        # overall space-time kernel
        K = np.kron(Kt,Ks)

        return K
    
class ExponentialKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
        self.num = np.array([np.sqrt(2*self.scale / self.std)])
        self.den = np.array([1/self.std])
        
    def kernelFunction(self, x1, x2):
        return self.scale * np.exp(-np.linalg.norm(x1-x2) / self.std)
    
    
class PeriodicKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
        self.num = np.array([np.sqrt(2*self.scale / self.std) * np.array([np.sqrt((1/self.std)**2 + (2*np.pi*self.frequency)**2) , 1])])         
        self.den = np.array([((1/self.std)**2 + (2*np.pi*self.frequency)**2 ), 2/self.std])
    
    def kernelFunction(x1,x2):
        return self.scale * np.cos(2*pi*self.frequency * np.linalg.norm(x1-x2)) * np.exp(-np.linalg.norm(x1-x2)/self.std)
    
    
class GaussianKernel(Kernel):
    def __init__(self, params):
        super().__init__(params)
        
    def kernelFunction(self, x1, x2):
        return self.scale *(np.exp(-np.linalg.norm(x1-x2)**2 / (2*self.std**2)))

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
    

def kernelFunction(type, params):
    """
    kernel = kernelFunction(type,kernel_specific_parameter) returns a function
    describing the desired kernel.
    INPUT: type: kernel type
           args: necessary (kernel specific) paremeters. 
                 Consistency must be ensured by the user
    OUTPUT: kernel: kernel function
    """
    if type == 'separable':
        ks = kernelFunction(params['space']['type'], params['space']['scale'], params['space']['std'])
        kt = kernelFunction(params['time']['type'], params['time']['scale'], params['time']['std'])
        def kernel(x1,x2):
            return np.matmul(kt(x1[0],x2[0]), ks(x1[1:end],x2[1:end]))
    
    elif type == 'exponential':
        scale = params['scale']
        std_dev = params['std']
        def kernel(x1,x2):
            return scale *np.exp(-np.linalg.norm(x1-x2) / std_dev)

    elif type == 'gaussian':
        scale = params['scale']
        std_dev = params['std']
        def kernel(x1,x2):
            return np.dot(scale, (np.exp(-np.linalg.norm(x1-x2)**2 / (2*std_dev**2))))
        
    elif type == 'periodic':
        scale = params['scale']
        std_dev = params['std']
        frequency = params['frequency']
        def kernel(x1,x2):
            return np.linalg.multi_dot([scale, np.cos(2*pi*frequency * np.linalg.norm(x1-x2)), np.exp(-np.linalg.norm(x1-x2)/std_dev)])
        
    else:
        print('Unknown type of kernel')

    return kernel


def kernelSampled(input_set_1, input_set_2, kernel_func):
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
    K = np.zeros((n_input1,n_input2))

    for i in np.arange(0, n_input1):
        for j in np.arange(0, n_input2):
            #todo: fix [i,:]
            K[i,j] = kernel_func(input_set_1[i] , input_set_2[j])

    return K

def kernelSpaceTimeSampled(space_locs1, space_locs2, time_instants1, time_instants2, param):
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

    # compute kernel functions
    kernel_space = kernelFunction(param['space']['type'], param['space'])
    kernel_time = kernelFunction(param['time']['type'] , param['time'])

    # sampled kernels
    Ks = kernelSampled(space_locs1, space_locs2, kernel_space)
    Kt = kernelSampled(time_instants1, time_instants2, kernel_time)

    # overall space-time kernel
    K = np.kron(Kt,Ks)

    return K
