# This is a sample Python script.
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import LinearLocator
from IPython.display import display  #
from scipy.stats import norm
import statistics
from scipy.interpolate import *


class Evaluator:
    def __init__(self, x , initial_data_size= 0,initial_data_points=[] ):
        self.x_init = x
        self.y_init = ...
        self.initial_data_size = 0
        self.initial_data_points = []
        self.number_of_iterations = 100
        self.query = [...,...]
        self.data_set =[]



    def startEval(self):
        'initial Data'
        self.x_init
        self.y_init
        # Gausian Prozess
        # gaussian_processes(self):
        # new_point():
        #   - acquisition functions: EI; EIGF; variance-based; dinstance-based
        # Get new Data
        # Check if

    'MUSS GEÄNDERT WERDEN, jedesmal das auswerten geht nicht -> durch String Function auswählen direkt!'
    def blackbox_functions(self, test = False, evaluate= True):
        if  len(self.x) == 1:
            def test_function():
                'One-dimensional non-stationary since-cosine function from GRAMACY(2005)'
                x1= self.x[0]
                if x1<10:
                    return np.sin(np.pi*x1/5)+ 1/5*np.cos(4*np.pi*x1/5)
                else:
                    return x1/10-0.8

        if len(self.x) == 2:
            x1 = self.x[0]
            x2 = self.x[1]
        def branin_function(x1,x2):
            'typical parameter values laut: https://statisticaloddsandends.wordpress.com/2019/06/24/test-functions-for-optimization-and-the-branin-hoo-function/'
            if True:
                a = 1
                b = 5.1 / (4 * np.pi**2)
                c = 5 / np.pi
                r = 6
                s = 10
                t = 1 / (8 * np.pi)

            return - a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

        def six_hump_camel_back_function(x1,x2):
            return (4 - 2.1 * (x1**2) + (x1*4) / 3.0) * (x1**2) + x1 * x2 + (-4 + 4 * (x2**2)) * (x2**2)

        def non_polynomial_surface_function(x1,x2):
            return ((30 + 5*x1*np.sin(5*x1))*(4+np.exp(-5*x2))-100)/6

        ' Hier kommt combination von 4 Gauß'
        def combined_gaussian_density_function(x1,x2):

            return None

        return None


    def gaussian_processes(self):

        return None

    def get_init_data(self):
        self.x
        self.y = self.blackbox_functions()
        return None




if __name__ == '__main__':
    eval = Evaluator(10)
    eval.startEval()
