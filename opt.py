''' DAS IST DER CODE FÜR MEINE MASTERARBEIT - STRENG DICH AN!'''

from matplotlib import pyplot as plt
import numpy as np
import random
import sklearn




''' Generate Datapoints with the same size in both directions in the same area; 
if we want different sizes we need to import the function 2 xmin and 2 xmax'''
def generate_grid_data(xmin,xmax,Delta):
    def plot(x1, x2):
        plt.figure("Grid Sampling; Grid Search")
        plt.scatter(x1, x2, s=2)
        plt.show()

    # Calculate f=sin(x1)+cos(x2)
    x1 = np.arange(xmin,xmax+Delta,Delta)   # generate x1 values from xmin to xmax
    x2 = np.arange(xmin,xmax+Delta,Delta)   # generate x2 values from xmin to xmax
    x1, x2 = np.meshgrid(x1,x2)             # make x1,x2 grid of points

    f = np.sin(x1) + np.cos(x2)             # calculate for all (x1,x2) grid

    #random.seed(2020)                       # random seed for reproducibility of the randomness
    plot(x1,x2)
    return x1,x2

def generate_random_data(xmin, xmax, num):
    def plot(x1_random, x2_random):
        plt.figure("Random Sampling")
        plt.scatter(x1_random, x2_random, s=2)
        plt.show()

    random.seed(2020)
    x1_random = random.choices(range(xmin, xmax), k=num)
    x2_random = random.choices(range(xmin, xmax), k=num)
    plot(x1_random,x2_random)
    return x1,x2



# Create {x1,x2,f} dataset every 1.0 from -10 to 10
x1,x2 =generate_grid_data(-10,10,1.0)
# Create {x1,x2,f} dataset every 1.0 from -10 to 10
x1,x2 = generate_random_data(-10,10,1000)