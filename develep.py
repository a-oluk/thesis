# This is a sample Python script.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import LinearLocator
from IPython.display import display  #
from scipy.stats import norm
import statistics
from scipy.interpolate import *


def plot_3_dim():
    # ZUFÄLLIGE TRAININGSDATEN NICHT SO GUT
    xdata = np.random.rand(1000) * 10
    ydata = np.random.rand(1000) * 10

    # MIT LINSPACE FUNKTIONIERT SEHR GUT
    #xdata = np.linspace(1,10,10)
    #ydata = np.linspace(1, 10, 10)

    zdata = (xdata-5)**2 + (ydata-5)**2+ 0.2* np.random.normal(0, 1, len(xdata))

    d = {'X': xdata, 'Y': ydata}
    df = pd.DataFrame(data=d)
    final_df = df.sort_values(by=['X'], ascending=True)
    xdata = df['X'].to_numpy()
    ydata= df['Y'].to_numpy()

    X,Y = np.meshgrid(xdata, ydata)
    Z = 0.2*(X-5)**2 + 0.2*(Y-5)**2 #0.2*np.random.normal(0, 1, len(xdata))

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    #surf = ax.plot_surface(X, Y, Z,cmap='viridis')
    #ax.set_zlim(0, 5)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.scatter(xdata, ydata, zdata, c=zdata, cmap='viridis', linewidth=0.5);

    # Give labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Save figure
    #plt.savefig('3d_scatter.png', dpi=300);
    plt.show()

plot_3_dim()