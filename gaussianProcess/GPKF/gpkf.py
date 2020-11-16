import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import expm, solve_continuous_lyapunov

from kernel import kernelFunction, kernelSampled
from params import Params

class Gpkf:
    def __init__(self, params):
        self.params = params # Todo: fix structure

    def estimation(self, meas, noiseVar):

        # number of measured locations and time instants
        numSpaceLocs,numTimeInstants = meas.shape

        # create DT state space model
        a,c,v0,q = createDiscreteTimeSys(self.params.gpkf['kernel']['time']['num'], self.params.gpkf['kernel']['time']['den'], self.params.data['samplingTime'])

        # create space kernel
        kernel_space = kernelFunction(self.params.gpkf['kernel']['space']['type'], self.params.gpkf['kernel']['space'])
        Ks_chol = np.linalg.cholesky(kernelSampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'], kernel_space)).conj().T

        # initialize quantities needed for kalman estimation
        I = np.eye(numSpaceLocs)
        A = np.kron(I,a)
        C = np.matmul(Ks_chol, np.kron(I,c))
        V0 = np.kron(I,v0)
        Q = np.kron(I,q)
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
            posteriorCov[:,:,t] = np.linalg.multi_dot([C, V[:,:,t], C.conj().T])
            posteriorCovPred[:,:,t] = np.linalg.multi_dot([C, Vp[:,:,t], C.conj().T])
            
            # compute output variance
            outputCov[:,:,t] = posteriorCov[:,:,t] + R[:,:,t]
            outputCovPred[:,:,t] = posteriorCovPred[:,:,t] + R[:,:,t]
        
        return posteriorMean, posteriorCov, logMarginal


    def prediction(self, meas, noiseVar):

        postMean, postCov, logMarginal = self.estimation(meas, noiseVar)

        kernel_space = kernelFunction(self.params.gpkf['kernel']['space']['type'], self.params.gpkf['kernel']['space'])
        kernelSection = kernelSampled(self.params.data['spaceLocsPred'], self.params.data['spaceLocsMeas'], kernel_space)
        kernelPrediction = kernelSampled(self.params.data['spaceLocsPred'], self.params.data['spaceLocsPred'], kernel_space)
        Ks = kernelSampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'], kernel_space)
        Ks_inv = np.linalg.inv(Ks)

        numSpaceLocsPred = np.max(self.params.data['spaceLocsPred'].shape)
        numTimeInsts = np.max(self.params.data['timeInstants'].shape)
        predictedCov = np.zeros((numSpaceLocsPred, numSpaceLocsPred, numTimeInsts))
        scale = self.params.gpkf['kernel']['time']['scale']

        predictedMean = np.matmul(kernelSection, np.matmul(Ks_inv, postMean))
                
        for t in np.arange(0, numTimeInsts):
            W = Ks_inv @ (Ks - postCov[:,:,t].conj().T / scale) @ Ks_inv
            predictedCov[:,:,t] = np.linalg.multi_dot([scale, (kernelPrediction - np.linalg.multi_dot([kernelSection, W, kernelSection.conj().T]))])

        return predictedMean, predictedCov

def createDiscreteTimeSys(num_coeff, den_coeff, Ts):
    # state dimension
    stateDim  = np.max(den_coeff.shape)

    if stateDim ==1:
        print('1 dim')
        F = -den_coeff       # state matrix
        A = np.exp(F * Ts)   # Discretization
        G = np.array([1])
    else:
        print('more dims')
        F = np.diag(np.ones((1,stateDim-1)),1).copy()
        F[stateDim-1] = -den_coeff
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
            Q = Q + np.linalg.multi_dot([t, expm(np.dot(F,n)), np.dot(G,G.conj().T), expm(np.dot(F,n)).conj().T])
    
    return A, C, V, Q

def kalmanEst(A, C, Q, V0, meas, noiseVar):

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
    
    #print(noiseVar)
    for t in np.arange(0, numTimeInstants):
        # prediction
        xpt = np.matmul(A,xt)
        Vpt = A @ Vt @ A.conj().T + Q

        # correction
        notNanPos = np.logical_not(np.isnan(meas[:,t]))
        Ct = C[notNanPos,:]
        Rt = noiseVar[:,:,t][np.ix_(notNanPos, notNanPos)]

        innovation = meas[:,t][np.newaxis].T[np.ix_(notNanPos)] -  (Ct @ xpt)
        innovVar = np.linalg.multi_dot([Ct, Vpt, Ct.conj().T]) + Rt
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
        
    return x, V, xp, Vp, logMarginal