import numpy as np

'''
In this file the functions which are used for regression are used are listed

'''
def function_1dim(X):
    return 2 * np.sin(2 * X) + X - 11

def function_1dim_1(X):
    if X < -5:
        return np.sin(3 * X) * X
    elif X >= -5 and X < 5.5:
        return (-X - 2)
    else:
        return 2 * np.sin(2 * X) + X - 11

def function_1dim_2(X):
    return 10*np.sin(2 * X) - X**2 - 10




def function_2dim(X):
    value = np.sin(X[:, 0]) + 3 * np.cos(X[:, 1])
    return value


# TODO: KERNEL PARAMETER ÄNDERN -> SONST SEHR SCHLECHTE ERGEBNISSE
def function_2dimm(X):
    # value= (X[:, 0] + 2 * X[:, 1] - 7)** 2 + (2 * X[:, 0] + X[:, 1] - 5)**2 #Booth function (watch Wikipedia) ABER ZU HOHE WERTE
    value = 100 * np.sqrt(np.abs(X[:, 1] - 0.01 * X[:, 0] ** 2)) + 0.01 * np.abs(X[:, 0] + 10)
    return value


def function_4dim(X):
    return np.sin(X[:, 0]) + 3 * np.cos(X[:, 1]) + np.sin(X[:, 2]) + 3 * np.cos(X[:, 3])
