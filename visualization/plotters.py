import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial as P
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm
from matplotlib import animation
from IPython import display

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

class Arrow3D(FancyArrowPatch):
    """
    Class for drawing customized arrows
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
def draw_3d_sensor_heatmap(locations, data, compass = True, compassPos = np.array([18, -20, -12.0]), arrowLength = 8):
    """
    Convenience function for plotting a heatmap using sensor positions and data for heatmap.
    Tip: use ""%matplotlib qt" before calling this function to move the 3d plot around
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.array([v[0] for v in locations])
    y = np.array([v[2] for v in locations])
    z = np.array([v[1] for v in locations])
    
    cageRadius = 25
    distanceFromEdge = np.array([np.linalg.norm(d) - 25 for d in [a[[0,2]] for a in locations]]) # Distance from the edges of the cage
    
    # Plot points
    cube = ax.scatter(x, y, z, zdir='z', c=data, cmap="winter", s = 40, edgecolors = ['k' if d > 0 else 'w' for d in distanceFromEdge], linewidth = 1.0, alpha = 1)  # Plot the cube
    cbar = fig.colorbar(cube, shrink=0.6, aspect=5)

    # Cylinder for the lice skirt
    center = 0
    us = np.linspace(0, 2 * np.pi, 32)
    zs = np.linspace(0, -10, 2)

    us, zs = np.meshgrid(us, zs)

    xs = cageRadius * np.cos(us)
    ys = cageRadius * np.sin(us)
    
    ax.plot_surface(xs, ys, zs, linewidth = 0, alpha=0.15)
    
    if compass:
        # Draw a compass with needles and directions
        eastArrow = Arrow3D([compassPos[0],compassPos[0] + arrowLength],[compassPos[1],compassPos[1]],[compassPos[2],compassPos[2]], mutation_scale=20, lw=2, arrowstyle="->", color="k")
        northArrow = Arrow3D([compassPos[0],compassPos[0]], [compassPos[1],compassPos[1] + arrowLength], [compassPos[2],compassPos[2]], mutation_scale=20, lw=2, arrowstyle="->", color="b")
        westArrow = Arrow3D([compassPos[0],compassPos[0] - arrowLength],[compassPos[1],compassPos[1]],[compassPos[2],compassPos[2]], mutation_scale=20, lw=2, arrowstyle="->", color="k")
        southArrow = Arrow3D([compassPos[0],compassPos[0]],[compassPos[1],compassPos[1] - arrowLength],[compassPos[2],compassPos[2]], mutation_scale=20, lw=2, arrowstyle="->", color="r")

        ax.add_artist(northArrow)
        ax.add_artist(southArrow)
        ax.add_artist(eastArrow)
        ax.add_artist(westArrow)

        ax.text(compassPos[0] + (arrowLength + 2), compassPos[1], compassPos[2], "E", va = "center", ha = "center")
        ax.text(compassPos[0] - (arrowLength + 2), compassPos[1], compassPos[2], "W", va = "center", ha = "center")
        ax.text(compassPos[0], compassPos[1] + (arrowLength + 2), compassPos[2], "N", va = "center", ha = "center")
        ax.text(compassPos[0], compassPos[1] - (arrowLength + 2), compassPos[2], "S", va = "center", ha = "center")

    # Top edge of the cage
    cage_circle=plt.Circle((center, center), cageRadius, color='black', fill=False, linewidth = 3, alpha = 0.5)
    ax.add_patch(cage_circle)
    art3d.pathpatch_2d_to_3d(cage_circle, z=0, zdir="z")

    ax.set_zlim(-12.5,-1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

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
        plt.plot(x_mesh, sample, lw=2, ls='--', label=f'Sample {i+1}')
        
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
    
def plot_residuals(y_pred, targets):
    plt.figure(figsize=(16,12))
    residuals = targets[~np.isnan(targets)] - y_pred[~np.isnan(targets)]
    plt.scatter(y_pred[~np.isnan(targets)], residuals, s=7, marker ='o', color='r', label='Residual')
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
    sns.heatmap(data[:,:,0], square=True, cmap = 'rocket')

    def init():
        sns.heatmap(np.zeros((data.shape[0], data.shape[1])), square=True, cbar=False, cmap='rocket')

    def animate(i):
        sns.heatmap(data[:,:,i], square=True, cbar=False, cmap='rocket')

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.arange(0,data.shape[2],10), repeat = False, interval = 100)
    video = anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()