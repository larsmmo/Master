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
        Ks_chol = np.linalg.chol(kernelSampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'], kernel_space)).conj().T

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
        x,V,xp,Vp,exeTime,logMarginal = kalmanEst(A,C,Q,V0,meas,R)

        # output function
        posteriorMean = np.matmul(C,x)
        posteriorMeanPred = np.matmul(C,xp)

        # posterior variance
        O3 = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
        posteriorCov = O3
        posteriorCovPred = O3
        outputCov = O3
        outputCovPred = O3

        for t in np.arange(0, numTimeInstants):
            
            # extract variance
            posteriorCov[:,:,t] = np.linalg.multi_dot([C, V[:,:,t], C.conj().T])
            posteriorCovPred[:,:,t] = np.linalg.multi_dot([C, Vp[:,:,t], C.conj().T])
            
            # compute output variance
            outputCov[:,:,t] = posteriorCov[:,:,t] + R[:,:,t]
            outputCovPred[:,:,t] = posteriorCovPred[:,:,t] + R[:,:,t]

        return posteriorMean, posteriorCov, logMarginal


    def prediction(self, meas, noiseVar):

        postMean, postCov, logMarginal = self.estimation(paramData, paramGpkf, meas, noiseVar)

        kernel_space = kernelFunction(self.params.gpkf['kernel']['space']['type'], self.params.gpkf['kernel']['space'])
        kernelSection = kernelSampled(self.params.data['spaceLocsPred'], self.params.data['spaceLocsMeas'], kernel_space)
        kernelPrediction = kernelSampled(self.params.data['spaceLocsPred'], self.params.data['spaceLocsPred'], kernel_space)
        Ks = kernelSampled(self.params.data['spaceLocsMeas'], self.params.data['spaceLocsMeas'], kernel_space)
        I = np.eye(Ks.shape[0])
        Ks_inv = I/Ks

        numSpaceLocsPred = np.max(self.params.data['spaceLocsPred'].shape)
        numTimeInsts = np.max(params.data['timeInstants'].shape)
        predictedCov = np.zeros((numSpaceLocsPred, numSpaceLocsPred, numTimeInsts))
        scale = params.gpkf['kernel']['time']['scale']

        predictedMean = np.matmul(kernelSection, np.matmul(Ks_inv, postMean))
                
        for t in np.arange(0, numTimeInsts):
            W = np.linalg.multi_dot([Ks_inv, (Ks - postCov[:,:,t]/scale), Ks_inv])
            predictedCov[:,:,t] = np.linalg.multi_dot([scale, (kernelPrediction - np.linalg.multi_dot([kernelSection, W, kernelSection.conj().T]))])

        return preditedMean, predictedCov

def createDiscreteTimeSys(num_coeff, den_coeff, Ts):
    # state dimension
    stateDim  = np.max(den_coeff.shape)

    # state matrix
    F = np.diag(np.ones((1,stateDim-1)).tolist(),1).copy()
    print(np.ones((1,stateDim-1)).tolist())
    F[stateDim-1] = -den_coeff

    # input matrix
    G = np.array([np.zeros((stateDim-1 ,1)).tolist() , 1])
    print(G)

    # output matrix
    C = np.zeros((1,stateDim))
    C[0:np.max(num_coeff.shape)] = num_coeff

    # discretization
    A = expm(np.matmul(F,Ts))  # state matrix

    # state variance as solution of the lyapunov equation
    print(F)
    print('-----')
    print(np.matmul(G, G.conj().T[0]))
    V = solve_continuous_lyapunov(F, np.matmul(G, G.conj().T))

    # discretization of the noise matrix
    Q = np.zeros(stateDim)
    Ns = 10000        
    t = Ts/Ns
    for n in np.arange(t, Ts+t, step=t):
        Q = Q + np.linalg.multi_dot([t, expm(np.matmul(F,n)), np.matmul(G,G.conj().T),expm(np.matmul(F,n)).conj().T])

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

    for t in arange(1,numTimeInstants+1):
      
        # prediction
        xpt = np.matmul(A,xt)
        Vpt = np.linalg.multi_dot([A,Vt,A.conj().T]) + Q
        
        # correction
        notNanPos = np.logical_not(np.isnan(meas[:,t]))
        Ct = C[notNanPos,:]
        Rt = noiseVar[notNanPos, notNanPos, t]
        
        innovation = meas[notNanPos,t] -  np.matmul(Ct, xpt)
        innovVar = np.linalg.multi_dot([Ct, Vpt, Ct.conj().T]) + Rt
        K = (np.matmul(Vpt, Ct.conj().T))/innovVar   # kalman gain
        correction = np.matmul(K, innovation)
        
        xt = xpt + correction
        Vt = np.linalg.multi_dot([(I - np.matmul(K,Ct)), Vpt ,(I - np.matmul(K,Ct)).conj().T]) + np.linalg.multi_dot([K, Rt, K.conj().T])
        
        # save values
        xp[:,t] = xpt
        Vp[:,:,t] = Vpt
        x[:,t] = xt
        V[:,:,t] = Vt
          
        # computations for the marginal likelihood
        l1 = np.sum(np.log(np.linalg.eig(innovVar)))
        l2 = np.matmul(innovation.conj().T, np.linalg.lstsq(innovVar, innovation)[0])
        logMarginal = logMarginal +  0.5*(np.matmul(np.size(innovation, axis=1),log(2*pi)) + l1 + l2)
        
    return x, V, xp, Vp, logMarginal