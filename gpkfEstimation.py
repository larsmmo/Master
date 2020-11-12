import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

def gpkfEstimation(paramData, paramGpkf, meas, noiseVar):

	# number of measured locations and time instants
	numSpaceLocs,numTimeInstants = meas.shape

	# create DT state space model
	a,c,v0,q = createDiscreteTimeSys(paramGpkf.kernel.time.num, paramGpkf.kernel.time.den, paramData.samplingTime)

	# create space kernel
	kernel_space = kernelFunction(paramGpkf.kernel.space.type, paramGpkf.kernel.space)
	Ks_chol = np.linalg.chol(kernelSampled(paramData.spaceLocsMeas, paramData.spaceLocsMeas, kernel_space)).conj().T

	# initialize quantities needed for kalman estimation
	I = np.eye(numSpaceLocs)
	A = np.kron(I,a)
	C = Ks_chol * np.kron(I,c)
	V0 = np.kron(I,v0)
	Q = np.kron(I,q)
	R = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
	for t in np.arange(1, numTimeInstant):
	    R[:,:,t] = np.diag(noiseVar[:,t])

	# compute kalman estimate
	x,V,xp,Vp,exeTime,logMarginal = kalmanEst(A,C,Q,V0,meas,R)

	# output function
	posteriorMean = C*x
	posteriorMeanPred = C*xp

	# posterior variance
	O3 = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
	posteriorCov = O3
	posteriorCovPred = O3
	outputCov = O3
	outputCovPred = O3

	for t in np.arange(0, numTimeInstants):
	    
	    # extract variance
	    posteriorCov[:,:,t] = C * V[:,:,t] * C.conj().T
	    posteriorCovPred[:,:,t] = C * Vp[:,:,t] * C.conj().T
	    
	    # compute output variance
	    outputCov[:,:,t] = posteriorCov[:,:,t] + R[:,:,t]
	    outputCovPred[:,:,t] = posteriorCovPred[:,:,t] + R[:,:,t]

	return posteriorMean, posteriorCov, logMarginal


def gpkfPrediction(paramData, paramGpkf, meas, noiseVar):

	postMean, postCov, logMarginal = gpkfEstimation(paramData, paramGpkf, meas, noiseVar)

	kernel_space = kernelFunction(paramGpkf.kernel.space.type, paramGpkf.kernel.space)
	kernelSection = kernelSampled(paramData.spaceLocsPred, paramData.spaceLocsMeas, kernel_space)
	kernelPrediction = kernelSampled(paramData.spaceLocsPred, paramData.spaceLocsPred, kernel_space)
	Ks = kernelSampled(paramData.spaceLocsMeas, paramData.spaceLocsMeas, kernel_space)
	I = np.eye(Ks.shape)
	Ks_inv = I/Ks

	numSpaceLocsPred = np.max(paramData.spaceLocsPred.shape)
	numTimeInsts = np.max(paramData.timeInstants.shape)
	predictedCov = np.zeros((numSpaceLocsPred, numSpaceLocsPred, numTimeInsts))
	scale = paramGpkf.kernel.time.scale

	predictedMean = kernelSection * (Ks_inv * postMean)
	        
	for t in np.arange(0,np.max(paramData.timeInstants.shape)):
	    W = Ks_inv * (Ks - postCov[:,:,t]/scale) * Ks_inv
	    predictedCov[:,:,t] = scale * (kernelPrediction - (kernelSection * W * kernelSection.conj().T))

	return preditedMean, predictedCov

def createDiscreteTimeSys(num_coeff, den_coeff, Ts):
	# state dimension
	stateDim  = np.max(den_coeff.shape)

	# state matrix
	F = np.diag(np.ones((1,stateDim-1)),1)
	F[stateDim,:] = -den_coeff

	# input matrix
	G = np.array([np.zeros((stateDim-1 ,1)) , 1])

	# output matrix
	C = np.zeros((1,stateDim))
	C[0:np.max(num_coeff.shape)] = num_coeff

	# discretization
	A = scipy.linalg.expm(F*Ts)  # state matrix

	# state variance as solution of the lyapunov equation
	V = scipy.linalg.solve_continuous_lyapunov(F,G*G.conj().T)

	# discretization of the noise matrix
	Q = np.zeros(stateDim)
	Ns = 10000        
	t = Ts/Ns
	for n in np.arange(t, Ts+t, step=t):
	    Q = Q + t * scipy.linalg.expm(F*n)*(G*G.conj().T)*scipy.linalg.expm(F*n).conj().T;

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
	    xpt = A*xt
	    Vpt = A*Vt*A.conj().T + Q
	    
	    # correction
	    notNanPos = np.logical_not(np.isnan(meas[:,t]))
	    Ct = C[notNanPos,:]
	    Rt = noiseVar[notNanPos, notNanPos, t]
	    
	    innovation = meas[notNanPos,t] -  Ct * xpt
	    innovVar = Ct * Vpt * Ct.conj().T + Rt
	    K = (Vpt * Ct.conj().T)/innovVar   # kalman gain
	    correction = K*innovation
	    
	    xt = xpt + correction
	    Vt = (I - K*Ct) * Vpt *(I - K*Ct).conj().T + K * Rt * K.conj().T
	    
	    # save values
	    xp[:,t] = xpt
	    Vp[:,:,t] = Vpt
	    x[:,t] = xt
	    V[:,:,t] = Vt
	      
	    # computations for the marginal likelihood
	    l1 = np.sum(np.log(np.linalg.eig(innovVar)))
	    l2 = innovation.conj().T * np.linalg.lstsq(innovVar, innovation)[0]
	    logMarginal = logMarginal +  0.5*(np.size(innovation, axis=1)*log(2*pi) + l1 + l2)
	    
	return x, V, xp, Vp, logMarginal