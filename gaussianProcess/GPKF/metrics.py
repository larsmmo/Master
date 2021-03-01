import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, r2_score

def print_metrics(y_true, y_pred):
    """
    Utility function for printing metrics
    INPUT:
        y_true: the true measurements
        y_pred: the predicted measurements

    OUTPUT: 
        MSE, RMSE, R2
    """
    y_true = y_true[~np.isnan(y_true)]
    y_pred = y_pred[~np.isnan(y_true)]
    print('---------------------------\n')
    print('Metrics: \n')
    print('MAE: ' + str(mean_absolute_error(y_true, y_pred)))
    print('MSE: ' + str(mean_squared_error(y_true, y_pred, squared = False)))
    print('RMSE: ' + str(mean_squared_error(y_true, y_pred, squared = True)))
    print('R2: ' + str(r2_score(y_true, y_pred)) +'\n')
    print('---------------------------\n')