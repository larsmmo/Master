import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial as P
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import animation

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

def plot_gpkf(x_mesh, x, y, y_pred, y_cov, samples=[], **kwargs):
    if y_cov.ndim > 1:
        uncertainty = 3 * np.sqrt(np.abs(np.diag(y_cov)))
    else:
        uncertainty = 3 * np.sqrt(np.abs(np.squeeze(y_cov)))
    #plt.figure(num = None, figsize=(16,12))
    
    plt.plot(x, y, markersize=6, color='r', label='Measurements', **kwargs) # linestyle='none', marker='o',
    plt.plot(x_mesh, y_pred, label='Prediction')
    plt.fill_between(np.ravel(x_mesh), y_pred + uncertainty, y_pred - uncertainty, alpha=0.1, label='95% confidence interval')
    
    for i, sample in enumerate(samples):
        plt.plot(x_mesh, sample, lw=1, ls='--', label=f'Sample {i+1}')
        
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    
def plot_gpkf_space_time(params, F, posteriorMean, posteriorCov, predictedMean, predictedCov, timeIndex =-1, spaceIndex = 0, samples=[]):
    plt.figure(num = None, figsize=(20,12))
    plt.subplot(2,2,1)
    
    # Space
    plot_gpkf(params.data['spaceLocsMeas'], params.data['spaceLocs'], F[:,timeIndex], posteriorMean[:,timeIndex], posteriorCov[:,:,timeIndex], **{'linestyle' : 'None', 'marker': 'x'})
    plt.title('GPKF space estimation')
    plt.subplot(2,2,2)
    plot_gpkf(params.data['spaceLocsPred'], params.data['spaceLocs'], F[:,timeIndex], predictedMean[:,timeIndex], predictedCov[:,:,timeIndex], **{'linestyle' : 'None', 'marker': 'x'})
    plt.title('GPKF space prediction')
    
    # Time
    i = np.where(params.data['spaceLocsPred'] == spaceIndex)[0][0]
    plt.subplot(2,2,3)
    j = np.where(params.data['spaceLocs'] == np.around(spaceIndex - np.mod(spaceIndex,3)))[0][0]
    if j in params.data['spaceLocsMeasIdx']:
        print(j)
        print(params.data['spaceLocsMeas'].shape)
        if j >= np.setdiff1d(np.arange(0, params.data['numLocs']) ,params.data['spaceLocsMeasIdx']):
            print('re-indexing j')
            plot_gpkf(params.data['timeInstants'], params.data['timeInstants'], F[j,:], posteriorMean[j-1,:], posteriorCov[j-1,j-1,:], **{'linestyle' : '--'})
        else:
            plot_gpkf(params.data['timeInstants'], params.data['timeInstants'], F[j,:], posteriorMean[j,:], posteriorCov[j,j,:], **{'linestyle' : '--'})
        plt.title('GPKF time estimation')
    else:
        print('GPKF not fitted for position ', i, ', No GPKF estimation plot for this position')
    plt.subplot(2,2,4)
    plot_gpkf(params.data['timeInstants'], params.data['timeInstants'], F[np.where(params.data['spaceLocs'] == np.around(spaceIndex - np.mod(spaceIndex,3)))[0][0],:], predictedMean[i,:], predictedCov[i,i,:], **{'linestyle' : '--'})
    plt.title('GPKF time prediction')

def plot_gp(x_mesh, x, y, y_pred, y_cov, samples=[]):
    uncertainty = 1.96 * np.sqrt(np.diag(y_cov))
    print(uncertainty.shape)
    plt.figure(figsize=(16,12))
    plt.plot(x, y, label='Measurements', linestyle='none', marker='o', markersize=4, color='r')
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
    
def plot_residuals(y_pred, residuals):
    plt.scatter(y_pred, residuals, s=7, marker ='o', color='r', label='Residual')
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.xlabel('$Predicted$')
    plt.ylabel('$Residual$')
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
    
def animate_heatmap(data):
    fig = plt.figure()
    sns.heatmap(data[0], vmax=.8, square=True)

    def init():
        sns.heatmap(np.zeros((data.shape[0], data.shape[1])), square=True, cbar=False)

    def animate(i):
        sns.heatmap(data[i], square=True, cbar=False)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=20, repeat = False)