import numpy as np
from params import Params
from gpkfEstimation import Gpkf
from generateSyntheticData import generateSyntheticData

def start_train():
	params = Params()
	model = Gpkf(params)

	F, Y, noiseVar = generateSyntheticData(params.data)

	# GPKF estimate
	posteriorMean, posteriorCov, logMarginal = model.estimation(Y, noiseVar)

	#GPKF prediction
	preditedMean, predictedCov = model.prediction(Y, noiseVar)
	
	return model

def main():
	model = start_train()

if __name__ == '__main__':
	print('Running train')
	main()
	print('Finished main')