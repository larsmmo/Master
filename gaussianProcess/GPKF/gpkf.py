import numpy as np
from numpy.linalg import multi_dot

from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize
from scipy.stats import loguniform

import sklearn.gaussian_process.kernels

from kernel import KernelFactory, Kernel, kernelFunction, kernelSampled
from params import Params

class Gpkf:
    def __init__(self, params):
        self.params = params # Todo: fix structure
        
        # create DT state space model
        self.a, self.c, self.v0, self.q = createDiscreteTimeSys(self.params.gpkf['kernel']['time']['num'], self.params.gpkf['kernel']['time']['den'], self.params.data['samplingTime'])
        
        # create time kernel
        self.factory = KernelFactory()
        self.kernel_time = self.factory.get_kernel(self.params.gpkf['kernel']['time']['type'], self.params.gpkf['kernel']['time']) 
        
        # create space kernel
        self.kernel_space = self.factory.get_kernel(self.params.gpkf['kernel']['space']['type'], self.params.gpkf['kernel']['space'])
        self.Ks_chol = np.linalg.cholesky(self.kernel_space.sampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'])).conj().T
        
    def optimize(self, meas, noiseVar, n_restarts):
        
        def nll(theta):
            self.params.data['kernel']['time']['scale'] = theta[0]
            self.params.data['kernel']['time']['std'] = theta[1]
            self.params.data['kernel']['space']['scale'] = theta[2]
            self.params.data['kernel']['space']['std'] = theta[3]
            
            self.kernel_time = self.factory.get_kernel(self.params.gpkf['kernel']['time']['type'], self.params.gpkf['kernel']['time'])
            self.kernel_space = self.factory.get_kernel(self.params.gpkf['kernel']['space']['type'], self.params.gpkf['kernel']['space'])
            self.Ks_chol = np.linalg.cholesky(self.kernel_space.sampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'])).conj().T
            
            self.a, self.c, self.v0, self.q = createDiscreteTimeSys(self.kernel_time.num, self.kernel_time.den, self.params.data['samplingTime'])
            
            # number of measured locations and time instants
            numSpaceLocs,numTimeInstants = meas.shape

            # initialize quantities needed for kalman estimation
            I = np.eye(numSpaceLocs)
            A = np.kron(I,self.a)
            C = self.Ks_chol @ np.kron(I,self.c)
            V0 = np.kron(I,self.v0)
            Q = np.kron(I,self.q)
            R = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
            for t in np.arange(0, numTimeInstants):
                R[:,:,t] = np.diag(noiseVar[:,t]).copy()

            x,V,xp,Vp,logMarginal = kalmanEst(A,C,Q,V0,meas,R)
            
            return logMarginal[0]
        
        res = minimize(nll, [self.params.data['kernel']['time']['scale'], self.params.data['kernel']['time']['std'], 
                             self.params.data['kernel']['space']['scale'], self.params.data['kernel']['space']['std']], 
               bounds=((1e-5, 1e+5), (1e-5, 1e+5), (1e-5, 1e+5), (1e-5, 1e+5)),
               method='L-BFGS-B')
        
        print('First run:', res.fun)
        
        for r in np.arange(0, n_restarts):
            print('Optimizer restart:', r+1, ' of ',  n_restarts)
            
            random_theta0 = loguniform.rvs(1e-5, 1e+5, size= 4)
            new_res = minimize(nll, random_theta0, 
               bounds=((1e-5, 1e+5), (1e-5, 1e+5), (1e-5, 1e+5), (1e-5, 1e+5)),
               method='L-BFGS-B')
            if new_res.fun < res.fun:
                res = new_res
            print(res.fun)
        
        # Update kernel parameters with best results
        self.params.data['kernel']['time']['scale'], self.params.data['kernel']['time']['std'], self.params.data['kernel']['space']['scale'], self.params.data['kernel']['space']['std'] = res.x
        self.params.gpkf['kernel'] = self.params.data['kernel']
        
        self.kernel_time = self.factory.get_kernel(self.params.gpkf['kernel']['time']['type'], self.params.gpkf['kernel']['time'])
        self.kernel_space = self.factory.get_kernel(self.params.gpkf['kernel']['space']['type'], self.params.gpkf['kernel']['space'])
        self.Ks_chol = np.linalg.cholesky(self.kernel_space.sampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'])).conj().T
        
        print(res.x)
        print(res.fun)
                   
        
    def estimation(self, meas, noiseVar):
        """
        Returns the estimate of the GPKF algorithm

        INPUT:  Measurements and corresponding noise variance matrix
        
        OUTPUT: PosteriorMean, posteriorCov: kalman estimates
                logMarginal: value of the final marginal log-likelihood
        """
        # number of measured locations and time instants
        numSpaceLocs,numTimeInstants = meas.shape

        # initialize quantities needed for kalman estimation
        I = np.eye(numSpaceLocs)
        A = np.kron(I,self.a)
        C = self.Ks_chol @ np.kron(I,self.c)
        V0 = np.kron(I,self.v0)
        Q = np.kron(I,self.q)
        R = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
        for t in np.arange(0, numTimeInstants):
            R[:,:,t] = np.diag(noiseVar[:,t]).copy()
        
        # compute kalman estimate
        x,V,xp,Vp,logMarginal = kalmanEst(A,C,Q,V0,meas,R)

        # output function
        posteriorMean = np.matmul(C,x)
        posteriorMeanPred = np.matmul(C,xp)

        # posterior variance
        O3 = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
        posteriorCov = O3.copy()
        posteriorCovPred = O3.copy()
        outputCov = O3.copy()
        outputCovPred = O3.copy()

        for t in np.arange(0, numTimeInstants):
            
            # extract variance
            #posteriorCov[:,:,t] = np.linalg.multi_dot([C, V[:,:,t], C.conj().T])
            #posteriorCovPred[:,:,t] = np.linalg.multi_dot([C, Vp[:,:,t], C.conj().T])
            posteriorCov[:,:,t] = C @ V[:,:,t] @ C.conj().T
            posteriorCovPred[:,:,t] = C @ Vp[:,:,t] @ C.conj().T
            
            # compute output variance
            outputCov[:,:,t] = posteriorCov[:,:,t] + R[:,:,t]
            outputCovPred[:,:,t] = posteriorCovPred[:,:,t] + R[:,:,t]
        
        return posteriorMean, posteriorCov, logMarginal


    def prediction(self, meas, noiseVar):
        """
        Returns kalman prediction accoring to the GPKF algorithm
        
        INPUT: data and Gpkf specific parameters, measurements and corresponding
        noise variance matrix
        
        OUTPUT: predictedMean, predictedCov: kalman estimates
        """

        postMean, postCov, logMarginal = self.estimation(meas, noiseVar)

        kernelSection = self.kernel_space.sampled(self.params.data['spaceLocsPred'], self.params.data['spaceLocsMeas'])
        kernelPrediction = self.kernel_space.sampled(self.params.data['spaceLocsPred'], self.params.data['spaceLocsPred'])
        Ks = self.kernel_space.sampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'])
        Ks_inv = np.linalg.inv(Ks)

        numSpaceLocsPred = np.max(self.params.data['spaceLocsPred'].shape)
        numTimeInsts = np.max(self.params.data['timeInstants'].shape)
        predictedCov = np.zeros((numSpaceLocsPred, numSpaceLocsPred, numTimeInsts))
        scale = self.params.gpkf['kernel']['time']['scale']
        predictedMean = np.matmul(kernelSection, np.matmul(Ks_inv, postMean))
                
        for t in np.arange(0, numTimeInsts):
            W = Ks_inv @ (Ks - postCov[:,:,t].conj().T / scale) @ Ks_inv
            predictedCov[:,:,t] = np.linalg.multi_dot([scale, (kernelPrediction - kernelSection @ W @ kernelSection.conj().T)])

        return predictedMean, predictedCov
    
    def draw_samples(sample_range, n_samples):
        mu = np.zeros(sample_range.shape)
        cov = self.kernel

def createDiscreteTimeSys(num_coeff, den_coeff, Ts):
    """
    builds the discrete time ss system in canonical form, given the numerator and
    denominator coefficients of the companion form

    INPUT:  numerator, denominator coefficients and sampling time Ts for the
            discretization

    OUTPUT: matrix A,C,V,Q: state matrix, output matrix,
            state variance matrix (solution of lyapunov equation), 
            and measurement variance matrix
    """
    # state dimension
    stateDim  = np.max(den_coeff.shape)
    if stateDim ==1:
        F = -den_coeff       # state matrix
        A = np.exp(F * Ts)   # Discretization
        G = np.array([1])
    else:
        F = np.diag(np.ones((1,stateDim-1)).flatten(),1).copy()
        F[stateDim-1, :] = -den_coeff
        A = expm(F * Ts)  # state matrix
        G = np.vstack([np.zeros((stateDim-1,1)),1]) # input matrix
    
    # output matrix
    C = np.zeros((1,stateDim))
    C[0:np.max(num_coeff.shape)] = num_coeff

    # state variance as solution of the lyapunov equation
    V = solve_continuous_lyapunov(F, -np.matmul(G, G.conj().T))

    # discretization of the noise matrix
    Q = np.zeros(stateDim)
    Ns = 10000        
    t = Ts/Ns
    if stateDim == 1:
        for n in np.arange(t, Ts+t, step=t):
            Q = Q + t * np.exp(np.dot(F,n)) * np.dot(G,G.conj().T) * np.exp(np.dot(F,n)).conj().T
    else:
        for n in np.arange(t, Ts+t, step=t):
            #Q = Q + np.linalg.multi_dot([t, expm(np.dot(F,n)), np.dot(G,G.conj().T), expm(np.dot(F,n)).conj().T])
            Q = Q + t * expm(F * n) @ (G @G.conj().T) @ (expm(F * n)).conj().T
    
    return A, C, V, Q

def kalmanEst(A, C, Q, V0, meas, noiseVar, optimize='False'):
    """
    Returns the prediction and corrections computed by means of standard
    iterative kalman filtering procedure

    INPUT:  (A,C,Q,V0) model matrices, meas vector of measuremenst, noiseVar
            corresponding noise variance matrix

    OUTPUT: x,V,xp,Vp estimates and predictions with corresponding covariance.
            logMarginal marginal log-likelihood
    """
    numTimeInstants = meas.shape[1]
    stateDim = A.shape[0]
    I = np.eye(stateDim)
    times = np.zeros((numTimeInstants,1))
    logMarginal = 0

    # initialization
    xt = np.zeros((stateDim,1))
    Vt = V0
    xp = np.zeros((stateDim,numTimeInstants))
    Vp = np.zeros((stateDim,stateDim,numTimeInstants))
    x = np.zeros((stateDim,numTimeInstants))
    V = np.zeros((stateDim,stateDim,numTimeInstants))

    for t in np.arange(0, numTimeInstants):
        # prediction
        xpt = np.matmul(A,xt)
        Vpt = A @ Vt @ A.conj().T + Q

        # correction
        notNanPos = np.logical_not(np.isnan(meas[:,t]))
        Ct = C[notNanPos,:]
        Rt = noiseVar[:,:,t][np.ix_(notNanPos, notNanPos)]

        innovation = meas[:,t][np.newaxis].T[np.ix_(notNanPos)] -  (Ct @ xpt)
        innovVar = Ct @ Vpt @ Ct.conj().T + Rt
        K = np.linalg.solve(innovVar.conj().T,(np.matmul(Vpt, Ct.conj().T)).conj().T).conj().T   # kalman gain
        correction = np.matmul(K, innovation)
        
        xt = xpt + correction

        Vt = np.linalg.multi_dot([(I - np.matmul(K,Ct)), Vpt ,(I - np.matmul(K,Ct)).conj().T]) + np.linalg.multi_dot([K, Rt, K.conj().T])

        # save values
        xp[:,t] = xpt[:,0]
        Vp[:,:,t] = Vpt[:,:]
        x[:,t] = xt[:,0]
        V[:,:,t] = Vt[:,:]
          
        # computations for the marginal likelihood
        l1 = np.sum(np.log(np.linalg.eig(innovVar)[0]))
        l2 = np.matmul(innovation.conj().T, np.linalg.solve(innovVar, innovation))

        logMarginal = logMarginal +  0.5*(np.max(innovation.shape) * np.log(2*np.pi) + l1 + l2)
    
    if optimize == True:
        return logMarginal[0]
    
    return x, V, xp, Vp, logMarginal[0]