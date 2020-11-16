import numpy as np

def kalmanEst(A, C, Q, V0, meas, noiseVar):

	numTimeInstants = meas.size[1]
	stateDim = A.size[0]
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
	    notNanPos = ~np.isnan(meas[:,t])
	    Ct = C[notNanPos,:]
	    Rt = noiseVar[notNanPos, notNanPos, t]
	    
	    innovation = meas[notNanPos,t] -  Ct * xpt
	    innovVar = Ct * Vpt * Ct.conj().T + Rt
	    K = (Vpt * Ct.conj().T)/innovVar;   # kalman gain
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
	    
	return (x, V, xp, Vp, logMarginal)