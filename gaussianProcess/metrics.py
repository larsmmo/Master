import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss


def print_metrics(y_true, y_pred):
	"""
	INPUT:
		y_true: the true measurements
		y_pred: the predicted measurements

	OUTPUT: 
		MSE, RMSE, R2
	"""
	print('Metrics: \n')
	print('MSE: ' + mean_squared_error(y_true, y_pred, squared = False) +'\n')
	print('RMSE: ' + mean_squared_error(y_true, y_pred, squared = True) + '\n')
	print('MAE: ' + mean_absolute_error(y_true, y_pred) + '\n')
	print('R2: ' + r2_score(y_true, y_pred) +'\n')
