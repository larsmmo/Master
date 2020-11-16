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
	predictedMean, predictedCov = model.prediction(Y, noiseVar)

	return model

def main():
	print('Starting training')
	model = start_train()
	print('Finished training')

if __name__ == '__main__':
	main()