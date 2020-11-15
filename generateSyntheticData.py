import numpy as np
from kernel import kernelSpaceTimeSampled


def generateSyntheticData(param):
    # create space-time sampled kernel
    K = kernelSpaceTimeSampled(param['spaceLocs'], param['spaceLocs'], param['timeInstants'], param['timeInstants'], param['kernel'])

    # sample "true" (zero mean) GP and measurements
    f = np.random.multivariate_normal(np.zeros(K.shape[0]), K, 1).conj().T

    # rearganization in matrix form (row:space, columns:time)
    numSpaceLocs = np.max(param['spaceLocs'].shape)
    numTimeInst = np.max(param['timeInstants'].shape)
    F = np.reshape(f , (numSpaceLocs , numTimeInst))

    # create measurements
    numSpaceLocsMeas = np.max(param['spaceLocsMeas'].shape)
    Y = F[param['spaceLocsMeas'],:] + param['noiseStd'] * np.random.standard_normal((numSpaceLocsMeas, numTimeInst))

    # delete (randomly) some measurements and build measurement cov matrix
    noiseVar = param['noiseStd']**2 * np.ones((numSpaceLocsMeas, numTimeInst))
    for t in np.arange(0,numTimeInst):
        idx = np.sort(np.random.permutation(numSpaceLocsMeas)[:np.random.randint(numSpaceLocsMeas)]).conj().T
        Y[idx,t] = np.nan
        noiseVar[idx,t] = np.inf

    return F, Y, noiseVar
