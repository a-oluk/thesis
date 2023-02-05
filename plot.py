import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import random


def start_plot():
    plt.figure("grid")
    plot_grid()

    plt.figure("random")
    plot_random()
    plt.show()

def plot_grid():
    x= np.array([0,1,2,3,4,5,6,7,8,9,10])
    y= np.array([0,1,2,3,4,5,6,7,8,9,10])

    X, Y = np.meshgrid(x, y)

    plt.title("Grid Sampling")
    plt.xlabel("First Parameter")
    plt.ylabel("Second Parameter")
    plt.scatter(X, Y)
    #plt.grid()



def plot_random():
    #x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x_random = []
    y_random = []
    for i in range(0, 100):
        # any random numbers from 0 to 1000
        x_random.append(random.randint(0, 100)/10)
        y_random.append(random.randint(0, 100)/10)

    #X, Y = np.meshgrid(x, y)

    plt.title("Grid Sampling")
    plt.xlabel("First Parameter")
    plt.ylabel("Second Parameter")
    plt.scatter(x_random, y_random)
    # plt.grid()

start_plot()