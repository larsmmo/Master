import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial as P
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = np.ravel(X)
    mu = np.ravel(mu)
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    fig = plt.subplots(figsize = (8,8))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
"""

def plot_gp(x_mesh, x, y, y_pred, y_cov, samples=[]):
    uncertainty = 1.96 * np.sqrt(np.diag(y_cov))
    print(uncertainty.shape)
    plt.figure(figsize=(16,12))
    plt.plot(x, y, 'r.', markersize=10, label='Measurements')
    plt.plot(x_mesh, y_pred, 'b-', label='Prediction')
    plt.fill_between(np.ravel(x_mesh), y_pred + uncertainty, y_pred - uncertainty, alpha=0.1, label='95% confidence interval')
    """
    plt.fill(np.concatenate([x_mesh, x_mesh[::-1]]),
             np.concatenate([y_pred - 1.9600 * y_cov,
                            (y_pred + 1.9600 * y_cov)[::-1]]),
             alpha=.1, fc='b', ec='None', label='95% confidence interval')
    """
    for i, sample in enumerate(samples):
        plt.plot(x_mesh, sample, lw=1, ls='--', label=f'Sample {i+1}')
        
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    
def plot_gp_example(x_mesh, x, y, y_pred, y_cov, samples=[]):
    uncertainty = 1.96 * np.sqrt(np.diag(y_cov))
    print(uncertainty.shape)
    plt.figure(figsize=(16,12))
    plt.plot(x, y, 'r.', markersize=10, label='Measurements')
    plt.plot(x_mesh, y_pred, 'b-', label='Prediction')

    plt.fill(np.concatenate([x_mesh, x_mesh[::-1]]),
             np.concatenate([y_pred - 1.9600 * y_cov,
                            (y_pred + 1.9600 * y_cov)[::-1]]),
             alpha=.1, fc='b', ec='None', label='95% confidence interval')

    for i, sample in enumerate(samples):
        plt.plot(x_mesh, sample, lw=1, ls='--', label=f'Sample {i+1}')
        
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    
def plot_predictions(x, y, pred_dates=None, lines=None):
    """Data plotted as scatter, and prediction as line"""

    # Resize the graph window
    fig = plt.figure(figsize=(10, 7))

    # Get the axes
    ax = plt.axes()

    # Show the grid
    # ax.grid()

    # Select marker size
    marker_sizes = [4 for _ in x]

    # Plot outputs
    if not(lines is None or pred_dates is None):
        plt.axvline(x=x[-1], linestyle='dashed', color='darkgray')
        for line, color in zip(lines, ['goldenrod', 'green', 'red']):
            plt.plot(pred_dates, line, color=color, linewidth=3, alpha=0.8)
    plt.scatter(x, y, s=marker_sizes, color='darkgray', zorder=100)

    # Format x-axis label angles
    fig.autofmt_xdate()

    # Raise the bottom of the figure to make space for the angled labels
    plt.subplots_adjust(bottom=0.23)

    plt.show()

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nRow, nCol = df.shape
    colNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{colNames[i]}')
    plt.show()