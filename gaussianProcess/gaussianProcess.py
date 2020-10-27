import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, Matern, WhiteKernel


def run_gaussian_process_regression(X, y, datelist):
    """
    Receives a formatted pandas dataframe,
    and performs a gaussian process regression.
    Returns the root mean squared error on the test set.
    """
    print("\nGaussian Process Regression")
    
    # Format data for fitting
    X_numeric = [pandas.to_numeric(example) for example in X]
    datelist_numeric = [pandas.to_numeric(date) for date in datelist]

    # Create linear regression object
    kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e2)) + WhiteKernel(1e-1)
    regression = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    # Fit the model
    regression.fit(X_numeric, y)

    # Make predictions using the testing set
    y_pred = regression.predict(X_numeric)  # predictions on the domain of X
    y_pred_all = regression.predict(datelist_numeric)

    return y_pred, y_pred_all