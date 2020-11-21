import numpy as np
from scipy.optimize import minimize


def nll_fn(X_train, Y_train, noise, params, theta):
	"""
	Algorithm 2.1 from http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, section 2.2
	INPUTS:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        params: GP parameters

    OUTPUTS:
        Minimization objective.
    """
    Y_train = Y_train.ravel()

	def ls(a, b):
	    return lstsq(a, b, rcond=-1)[0]

	K = kernelFunction(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
	    noise**2 * np.eye(len(X_train))
	L = np.linalg.cholesky(K)

	return np.sum(np.log(np.diagonal(L))) + \
	       0.5 * Y_train.dot(ls(L.T, ls(L, Y_train))) + \
	       0.5 * len(X_train) * np.log(2*np.pi)