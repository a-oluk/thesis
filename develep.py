# This is a sample Python script.
import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import LinearLocator
from IPython.display import display  #
from scipy.stats import norm
import statistics
from scipy.interpolate import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
######
import numpy, scipy, scipy.optimize
import matplotlib
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm # to colormap 3D surfaces from blue to red
import matplotlib.pyplot as plt

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


    xdata = np.random.rand(100) * 10
    ydata = np.random.rand(100) * 10
    zdata = (xdata-5)**2 + (ydata-5)**2 -2 #+ np.random.normal(0, 1, len(xdata))
    data = [xdata,ydata,zdata]
    X,Y = np.meshgrid(xdata, ydata)
    Z = 0.2*(X-5)**2 + 0.2*(Y-5)**2 #0.2*np.random.normal(0, 1, len(xdata))
    Z = 2*(X-2)**3 + 3*(Y-2)**3 + np.random.normal(0, 1, len(xdata))

    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')

    #surf = ax.plot_surface(X, Y, Z,cmap='viridis')
    #ax.set_zlim(0, 5)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.scatter(xdata, ydata, zdata, c=zdata, cmap='viridis', linewidth=0.5);

    #ax.plot_surface(xdata, ydata, zdata, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    #ax.scatter(xdata, ydata, zdata, c=xdata, cmap='viridis', linewidth=0.5);
    # Give labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Save figure
    #plt.savefig('3d_scatter.png', dpi=300);

    plt.figure(1)
    #ax = fig.add_subplot(111, projection="3d")
    #ax = Axes3D(fig)
    ax.plot_trisurf(xdata,ydata,zdata, cmap='viridis', edgecolor='none',linewidth=0,antialiased=False)
    plt.show()

def opti():
    xdata = np.random.rand(100) * 10
    ydata = np.random.rand(100) * 10
    zdata = (xdata - 5) ** 2 + (ydata - 5) ** 2 - 2  # + np.random.normal(0, 1, len(xdata))
    data = [xdata,ydata,zdata]
    initialParameters = [2,2,2,2,2]
    fittedParameters, pcov = scipy.optimize.curve_fit(func, [xdata, ydata], zdata, p0=initialParameters)

    ScatterPlot(data)
    SurfacePlot(func, data, fittedParameters)
    ContourPlot(func, data, fittedParameters)



    modelPredictions = func(data, *fittedParameters)
    absError = modelPredictions - zdata
    SE = numpy.square(absError)  # squared errors
    MSE = numpy.mean(SE)  # mean squared errors
    RMSE = numpy.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(zdata))
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)


def func(data, a, b, c,p,q):
    x = data[0]
    y = data[1]
    return a * (x)**p + b* (y) **q + c

def SurfacePlot(func, data, fittedParameters):
    f = plt.figure()

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = func(numpy.array([X, Y]), *fittedParameters)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter(x_data, y_data, z_data)  # show data along with plotted surface

    axes.set_title('Surface Plot (click-drag with mouse)')  # add a title for surface plot
    axes.set_xlabel('X Data')  # X axis data label
    axes.set_ylabel('Y Data')  # Y axis data label
    axes.set_zlabel('Z Data')  # Z axis data label

    plt.show()
    plt.close('all')  # clean up after using pyplot or else there can be memory and process problems

def ContourPlot(func, data, fittedParameters):
    f = plt.figure()
    axes = f.add_subplot(111)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = func(numpy.array([X, Y]), *fittedParameters)

    axes.plot(x_data, y_data, 'o')

    axes.set_title('Contour Plot')  # add a title for contour plot
    axes.set_xlabel('X Data')  # X axis data label
    axes.set_ylabel('Y Data')  # Y axis data label

    CS = matplotlib.pyplot.contour(X, Y, Z, colors='k')
    matplotlib.pyplot.clabel(CS, inline=1, fontsize=10)  # labels for contours

    plt.show()
    plt.close('all')  # clean up after using pyplot or else there can be memory and process problems

def ScatterPlot(data):
    f = plt.figure()

    plt.grid(True)
    axes = Axes3D(f)
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    axes.scatter(x_data, y_data, z_data)

    axes.set_title('Scatter Plot (click-drag with mouse)')
    axes.set_xlabel('X Data')
    axes.set_ylabel('Y Data')
    axes.set_zlabel('Z Data')

    plt.show()
    plt.close('all')  # clean up after using pyplot or else there can be memory and process problems


#plot_3_dim()
opti()