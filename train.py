import numpy as np
from gpkfEstimation import Gpkf

def start_train():
	model = Gpkf()
	params = Params()
	model.estimation(params)
	return model

def main():
	model = start_train()

if __name__ == '__main__':
	print('Running train')
	main()
	print('Finished main')