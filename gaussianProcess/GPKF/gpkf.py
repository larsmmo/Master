import numpy as np
from numpy.linalg import multi_dot

from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize
from scipy.stats import loguniform

import sklearn.gaussian_process.kernels
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, r2_score

from kernel import Kernel
from params import Params

class Gpkf:
    def __init__(self, kernel_time, kernel_space, alpha = 0.05, normalize_y = True, ID = 0):
        
        self.ID = ID
        self.alpha = alpha
        self.normalize_y = normalize_y
        
        # Set time kernel
        self.kernel_time = kernel_time
        self.init_time = 0
        
        # Set space kernel
        self.kernel_space = kernel_space

    
    """def set_params(self, **parameters):

        Function for setting kernel parameters. Required for cross-validation (sklearn)

        #for parameter, value in parameters.items():
         #   setattr(self, parameter, value)
        
        time_params_n = len(self.kernel_time.kernel[:])
        self.kernel_time.set_params(parameters.values()[:time_params_n])
        self.kernel_space.set_params(parameters.values()[time_params_n:])
        
        return self"""
    
    def get_params(self,deep=True):
        """
        Function for getting parameters. Required for cross-validation (sklearn)
        
        Output: An indexed dict containing all parameters for model instantiation
        """
        
        return {'alpha':self.alpha, 'kernel_time':self.kernel_time,
                'kernel_space':self.kernel_space, 'normalize_y':self.normalize_y}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            print(parameter)
            setattr(self, parameter, value)
        return self
    
    
    def fit(self, spaceLocsMeas, timeInstants, y, u =np.array([]), optimize = False, fun = None, n_restarts = 0, bounds = None, focus = "simulation"):
        """
        INPUT:
            spaceLocsMeas: Spatial indices of measurements to be fitted
            timeInstants: Temporal indices of measurements to be fitted
            y: Target values for the given spatial and temporal indices
            optimize: Whether to optimize kernel parameters or not
            fun: Objective function to be minimized in case of optimization
        """
        def obj(theta):
            # Set hyperparameters to theta
            self.kernel_time.set_hyperparams(theta[:time_params_n])
            self.kernel_space.set_hyperparams(theta[time_params_n:])
            
            #print(theta)
            
            # Calculate covariances for space kernel
            Kss = self.kernel_space.sample(spaceLocsMeas)
            Ks_chol = np.linalg.cholesky(Kss).conj().T
            
            if issubclass(type(self.kernel_time), Kernel):
                self.a, self.c, self.v0, self.q = self.kernel_time.createDiscreteTimeSys()
                self.b = np.array([])
            else:
                # Exogenous input requires a state-space system with a B matrix
                self.a, self.b, self.c, self.v0, self.q = self.kernel_time.createDiscreteTimeSys()

            # initialize quantities needed for kalman estimation
            A = np.kron(I, self.a)
            B = np.vstack([self.b]*I.shape[0])
            C = Ks_chol @ np.kron(I,self.c)
            V0 = np.kron(I, self.v0)
            Q = np.kron(I, self.q)
            
            score = None
            
            if fun == 'nll':
                x,V,xp,Vp,logMarginal = self.kalmanEst(A,B,C,Q,V0,R, self.y_train, computeLikelihood = True)
                score = logMarginal
            elif fun == 'RMSE':
                score = self.score(spaceLocsMeas, timeInstants, y, metric = 'RMSE')
            
            print(score)
            
            return score
        
        self.spaceLocsMeas = spaceLocsMeas
        self.y_train_timeInstants = timeInstants
        
        self.u = u
        
        if issubclass(type(self.kernel_time), Kernel):
            self.a, self.c, self.v0, self.q = self.kernel_time.createDiscreteTimeSys()
            self.b = np.array([])
            self.exogenous = False
            #self.u = np.array([ [] for _ in range(y.shape[1]) ]).T
        else:
            # Exogenous input requires a state-space system with a B matrix
            self.a, self.b, self.c, self.v0, self.q = self.kernel_time.createDiscreteTimeSys()
            self.exogenous = True
            self.init_time = self.kernel_time.get_init_time()
            self.focus = focus # Simulation or prediction
        
        
        # Create noise matrix
        self.noiseVar = np.power((self.alpha * y), 2) #np.ones(y.shape)
        #self.noiseVar = self.alpha**2 * np.ones(y.shape)
        
        if self.normalize_y:
            self.y_train_mean = np.nanmean(y)
            self.y_train_std = np.nanstd(y)
            #self.y_train = y - self.y_train_mean
            self.y_train = (y - self.y_train_mean)#/self.y_train_std
        else:
            self.y_train_mean = 0
            self.y_train_std = 1
            self.y_train = y
            
        
        if optimize is True and fun is not None:
            # Getting some useful values
            numSpaceLocs, numTimeInstants = y.shape
            time_params_n = len(self.kernel_time.kernel[:])
            space_params_n = len(self.kernel_space.kernel[:])

            I = np.eye(numSpaceLocs)
            R = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
            for t in np.arange(0, numTimeInstants):
                R[:,:,t] = np.diag(self.noiseVar[:,t]).copy()
                
            if bounds is None:
                bounds = [(1e-3, 1e+4) for i in range(time_params_n + space_params_n)]
                
            # Use ML to find best kernel parameters using initial guess
            res = minimize(obj, np.concatenate((self.kernel_time.kernel[:], self.kernel_space.kernel[:]), axis=0), 
                   bounds= bounds,
                   method='L-BFGS-B')

            # Repeat for random guesses using loguniform
            for r in np.arange(0, n_restarts):
                print('Optimizer restart:', r+1, ' of ',  n_restarts)

                random_theta0 = loguniform.rvs(1e-4, 1e+4, size= time_params_n + space_params_n)

                new_res = minimize(obj, random_theta0, 
                   bounds= bounds,
                   method='L-BFGS-B')

                # Store best values
                if new_res.fun < res.fun:
                    res = new_res

            # Update parameters with best results
            self.kernel_time.set_hyperparams(res.x[:time_params_n])
            self.kernel_space.set_hyperparams(res.x[time_params_n:])

            # Update state-space representation
            if issubclass(type(self.kernel_time), Kernel):
                self.a, self.c, self.v0, self.q = self.kernel_time.createDiscreteTimeSys()
                self.b = np.array([])
            else:
                # Exogenous input requires a state-space system with a B matrix
                self.a, self.b, self.c, self.v0, self.q = self.kernel_time.createDiscreteTimeSys()

            print('Best hyperparameters and marginal log value')
            print(res.x)
            print(res.fun)
        
        
    def estimate(self, y, u= np.array([]), prediction = True):
        """
        Returns the estimate of the GPKF algorithm

        INPUT:  prediction: decision variable for undoing normalization
        
        OUTPUT: PosteriorMean, posteriorCov: kalman estimates
                logMarginal: value of the final marginal log-likelihood
        """
        # number of measured locations and time instants
        numSpaceLocs,numTimeInstants = y.shape

        # initialize quantities needed for kalman estimation
        I = np.eye(numSpaceLocs)
        A = np.kron(I,self.a)
        B = np.vstack([self.b]*I.shape[0])
        self.Ks_chol = np.linalg.cholesky(self.kernel_space.sample(self.spaceLocsMeas, self.spaceLocsMeas)).conj().T
        C = self.Ks_chol @ np.kron(I, self.c)
        V0 = np.kron(I,self.v0)
        Q = np.kron(I,self.q)
        R = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))

        for t in np.arange(0, numTimeInstants):
            R[:,:,t] = np.diag(self.noiseVar[:,t]).copy()
        
        # compute kalman estimate
        x,V,xp,Vp,logMarginal = self.kalmanEst(A,B,C,Q,V0,R,y,u, computeLikelihood = False)

        # output function
        if self.exogenous:
            posteriorMean = np.zeros((numSpaceLocs, numTimeInstants))
            #posteriorMeanPred = np.zeros((numSpaceLocs, numTimeInstants))
            self.coeffs = x
        else:
            posteriorMean = np.matmul(C,x)# + np.matmul(B,u)
            #posteriorMeanPred = np.matmul(C,xp)# + np.matmul(B,u)
            
        # compute kalman estimate
        x,V,xp,Vp,logMarginal = self.kalmanEst(A,B,C,Q,V0,R,y,u, computeLikelihood = False)

        # posterior variance
        O3 = np.zeros((numSpaceLocs, numSpaceLocs, numTimeInstants))
        posteriorCov = O3.copy()
        #posteriorCovPred = O3.copy()
        outputCov = O3.copy()
        #outputCovPred = O3.copy()

        for t in np.arange(0, numTimeInstants - self.init_time):
            if self.exogenous and self.kernel_time.adaptive:
                posteriorMean[:,t] = np.matmul(self.Cs[:,:,t], x[:,t])# + np.matmul(B,u)
                #posteriorMeanPred[:,t] = np.matmul(self.Cs[:,:,t], xp[:,t])# + np.matmul(B,u)
                
                posteriorCov[:,:,t] = self.Cs[:,:,t] @ V[:,:,t] @ self.Cs[:,:,t].conj().T
                #posteriorCovPred[:,:,t] = self.Cs[:,:,t] @ Vp[:,:,t] @ self.Cs[:,:,t].conj().T
            else:
                # extract variance
                posteriorCov[:,:,t] = C @ V[:,:,t] @ C.conj().T
                #posteriorCovPred[:,:,t] = C @ Vp[:,:,t] @ C.conj().T
            
            # compute output variance
            outputCov[:,:,t] = posteriorCov[:,:,t] + R[:,:,t]
            #outputCovPred[:,:,t] = posteriorCovPred[:,:,t] + R[:,:,t]
            
        # Undo normalization only if we predict at fitted locations
        if self.normalize_y and not prediction:
            posteriorMean = posteriorMean + self.y_train_mean
            #posteriorMean = posteriorMean + self.y_train_mean #+ np.nanmean(y)
            #posteriorCov = posteriorCov * self.y_train_std**2
        
        return posteriorMean, posteriorCov, logMarginal


    def predict(self, spaceLocsPred, timeInstants = None):
        """
        Returns kalman prediction according to the GPKF algorithm
        
        INPUT: data and Gpkf specific parameters
        
        OUTPUT: predictedMean, predictedCov: kalman estimates
        """
        
        if timeInstants is not None:
            if timeInstants[-1] > self.y_train_timeInstants[-1]:  # TODO: rework to add predictions inbetween measurements
                # Add locations we want to predict for that are after t_k (last fitted measurement in time)
                y = np.concatenate((self.y_train, np.full((self.y_train.shape[0], int(timeInstants[-1] - self.y_train_timeInstants[-1])), np.nan)), axis=1)
                self.y_train = y
                
                self.noiseVar = np.concatenate((self.noiseVar,np.full((self.noiseVar.shape[0], int(timeInstants[-1] - self.y_train_timeInstants[-1])), np.nan)), axis=1) 
                
                self.y_train_timeInstants = np.arange(self.y_train_timeInstants[0], timeInstants[-1] + 1)
                
            else:
                y = self.y_train

            #self.noiseVar = (self.alpha**2) * np.ones(y.shape) # todo: zeros?
        """else:
            y = self.y_train"""
            
        postMean, postCov, logMarginal = self.estimate(y, self.u)
        
        #print(logMarginal)

        kernelSection = self.kernel_space.sample(spaceLocsPred, self.spaceLocsMeas)
        kernelPrediction = self.kernel_space.sample(spaceLocsPred, spaceLocsPred)
        Ks = self.kernel_space.sample(self.spaceLocsMeas, self.spaceLocsMeas)
        Ks_inv = np.linalg.inv(Ks)

        numSpaceLocsPred = spaceLocsPred.shape[0]
        numTimeInsts = y.shape[1] # np.max(self.params.data['timeInstants'].shape)
        predictedCov = np.zeros((numSpaceLocsPred, numSpaceLocsPred, numTimeInsts))
        scale = self.kernel_time.sample(np.array([[0]]))[0][0]  # This is the same as gamma(0) = variance
        predictedMean = kernelSection @ (Ks_inv @ postMean)
        
        # Posterior variance
        for t in np.arange(0, numTimeInsts-self.init_time):
            if self.kernel_time.adaptive:
                scale = self.errVar[t]
            W = Ks_inv @ (Ks - postCov[:,:,t].conj().T / scale) @ Ks_inv
            predictedCov[:,:,t] = scale * (kernelPrediction - kernelSection @ W @ kernelSection.conj().T)
            
        # Undo normalization
        if self.normalize_y:
            predictedMean =  predictedMean + self.y_train_mean
            #predictedCov = predictedCov * self.y_train_std**2
        
        # If no timeInstants are given, return predictions for all times between the first and last measurement in y_train
        if timeInstants is not None:
            prediction_indices = np.nonzero(np.in1d(self.y_train_timeInstants, timeInstants))[0]
            
            predictedMean = predictedMean[:, prediction_indices]
            predictedCov = predictedCov[:,:, prediction_indices]
        
        return predictedMean, predictedCov
    
    def kalmanEst(self, A, B, C, Q, V0, noiseVar, y, u = np.array([]), computeLikelihood = True):
        """
        Returns the prediction and corrections computed by means of standard
        iterative kalman filtering procedure

        INPUT:  (A,C,Q,V0,u) = model matrices, noise variance, targets, exogenous variables

        OUTPUT: x,V,xp,Vp estimates and predictions with corresponding covariance.
                logMarginal marginal log-likelihood
        """
        numSpaceLocsMeas, numTimeInstants = y.shape
        stateDim = A.shape[0]
        I = np.eye(stateDim)
        logMarginal = 0

        # initialization
        xt = np.zeros((stateDim,1))
        Vt = V0
        xp = 0#xp = np.zeros((stateDim,numTimeInstants))
        Vp = 0#Vp = np.zeros((stateDim,stateDim,numTimeInstants))
        x = np.zeros((stateDim,numTimeInstants))
        V = np.zeros((stateDim,stateDim,numTimeInstants))
        
        if self.kernel_time.adaptive:
            # Measurement matrix can change in time for adaptive scheme
            self.Cs = np.zeros((C.shape[0], C.shape[1], numTimeInstants - self.init_time))
            
            self.errVar = np.zeros(numTimeInstants - self.init_time)
            errors = np.zeros((numSpaceLocsMeas,1))
            n_errors = 0
            if self.focus == "simulation":
                self.simulated = np.zeros((numSpaceLocsMeas, numTimeInstants))

        for t in np.arange(0, numTimeInstants - self.init_time):
            # Prediction
            xpt = A @ xt
            Vpt = A @ Vt @ A.conj().T + Q
            
            if self.kernel_time.adaptive:
                if self.focus == "simulation":
                    # Feed back estimated values (dynamic)
                    C = self.Ks_chol @ self.kernel_time.measurement_matrix(t, self.simulated, self.u)
                    self.simulated[:,[t + self.init_time]] = C @ xpt
                    
                elif self.focus == "prediction":
                    C = self.Ks_chol @ self.kernel_time.measurement_matrix(t, self.y_train , self.u)
                
                self.Cs[:,:,t] = C[:,:]

            # Correction
            notNanPos = np.logical_not(np.isnan(y[:,t]))
            Ct = C[notNanPos,:]
            
            Rt = noiseVar[:,:,t][np.ix_(notNanPos, notNanPos)]
            
            # Error residual
            innovation = y[:,t][np.newaxis].T[np.ix_(notNanPos)] - np.dot(Ct, xpt)
            
            if self.kernel_time.adaptive:
                # Error variance of ARX
                errors[notNanPos] += innovation
                n_errors += sum(notNanPos)
                self.errVar[t] = (np.linalg.norm(errors, 2)**2) / (2*(n_errors))
            
            # Common subexpression computation for performance
            VptCtt = np.dot(Vpt, Ct.conj().T)   
            
            # Projection of system uncertainty into measurement space
            innovVar = np.dot(Ct, VptCtt) + Rt 
            
            # Mapping system uncertainty into Kalman gain
            K = np.linalg.solve(innovVar.conj().T, VptCtt.conj().T).conj().T 
            
            # Scale residual by Kalman gain and predict new state
            correction = K @ innovation
            xt = xpt + correction
            
            I_KdotCt = I - np.dot(K, Ct) # Common subexpression
            Vt = np.dot(np.dot(I_KdotCt, Vpt), I_KdotCt.conj().T) + np.dot(np.dot(K, Rt), K.conj().T)
            #Vt = I_KdotCt @ Vpt  @ I_KdotCt.conj().T + (K @ Rt @ K.conj().T)

            # save values
            #xp[:,t] = xpt[:,0]
            #Vp[:,:,t] = Vpt[:,:]
            x[:,t] = xt[:,0]
            V[:,:,t] = Vt[:,:]
            
            # Marginal likelihood computations
            if computeLikelihood is True:
                l1 = np.sum(np.log(np.linalg.eig(innovVar)[0]))
                l2 = (innovation.conj().T @ np.linalg.lstsq(innovVar, innovation)[0])[0][0]
                logMarginal = logMarginal +  0.5*(np.max(innovation.shape) * np.log(2*np.pi) + l1 + l2)
        
        return x, V, xp, Vp, logMarginal
        
    def score(self, spaceLocs, timeInstants, targets, metric = None):
        """
        Returns a dict of score metrics for predicted values against target values
        
        Inputs:
            spaceLocsPred: spatial locations of targets
            timeInstants: temporal locations of targets
        Outputs:
            scores: dict of different metrics
        """
        # Append nan values we want to predict
        #meas = np.concatenate((self.y_train, np.full(y.shape, np.nan)), axis=1)
                              
        y_pred, _ = self.predict(spaceLocs, timeInstants)

        y_true = targets[~np.isnan(targets)]
        y_pred = y_pred[~np.isnan(targets)]
        
        if metric == 'RMSE':
            return mean_squared_error(y_true, y_pred, squared=False)
        
        return {'RMSE' : mean_squared_error(y_true, y_pred, squared=False), 'MAE' : mean_absolute_error(y_true, y_pred), 'R2' : r2_score(y_true, y_pred)}
        
        
    """def __str__(self):
        return 'Time kernel: \n'+ str(self.kernel_time) + '\n' + 'Space kernel: \n' + str(self.kernel_space)"""
    
    def to_dict(self):
        return ("Model" + str(self.ID) + 
            "Time kernel: " + vars(self.kernel_time) + "\nSpace kernel: " + vars(self.kernel_space))
