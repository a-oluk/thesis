# This is a sample Python script.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display  #
from scipy.stats import norm
import statistics
from scipy.interpolate import *


def plot_3_dim():

    xdata = np.random.rand(1000) * 10
    ydata = np.random.rand(1000) * 10
    zdata = (xdata-5)**2 + (ydata-5)**2 -2 + np.random.normal(0, 1, len(xdata))
    X,Y = np.meshgrid(xdata, ydata)
    Z = 2*(X-2)**2 + 3*(Y-2)**2 + np.random.normal(0, 1, len(xdata))

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    #ax.plot_surface(xdata, ydata, zdata, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.scatter(xdata, ydata, zdata, c=xdata, cmap='viridis', linewidth=0.5);

    # Give labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Save figure
    #plt.savefig('3d_scatter.png', dpi=300);
    plt.show()

plot_3_dim()