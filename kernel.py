import numpy as np

#class Kernel:
	#def __init__(self, type, varargin):

def kernelFunction(type, *args):
	"""
	kernel = kernelFunction(type,kernel_specific_parameter) returns a function
	describing the desired kernel.

	INPUT: type: kernel type
	       args: necessary (kernel specific) paremeters. 
	             Consistency must be ensured by the user

	OUTPUT: kernel: kernel function
	"""
	params = args

	if type == 'separable':
		ks = kernelFunction(params.space.type, params.space.scale, params.space.std)
        kt = kernelFunction(params.time.type, params.time.scale, params.time.std)
        def kernel(x1,x2):
        	return kt(x1[0],x2[0]) * ks(x1[1:end],x2[1:end])
            
    elif type == 'exponential':
        scale = params.scale
        std_dev = params.std
        def kernel(x1,x2):
        	return scale * np.exp(-np.norm(x1-x2) / std_dev)      

    elif type == 'gaussian':
        scale = params.scale
        std_dev = params.std
        def kernel(x1,x2):
        	return scale * np.exp(-np.norm(x1-x2)**2 / (2*std_dev**2))
        
    elif type == 'periodic':
        scale = params.scale
        std_dev = params.std
        frequency = params.frequency
        def kernel(x1,x2):
        	return scale * np.cos(2*pi*frequency * np.norm(x1-x2)) * np.exp(-np.norm(x1-x2)/std_dev)
        
    else
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
	        K[i,j] = kernel_func(input_set_1[i,:] , input_set_2[j,:])

	return K