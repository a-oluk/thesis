import numpy as np

'''
In this file the functions which are used for regression are used are listed

'''


def function_1dim(X):
    return 2 * np.sin(2 * X) + X - 11


@np.vectorize
def function_1dim_1(X):
    if X < -5:
        return np.sin(3 * X) * X
    elif X >= -5 and X < 5.5:
        return (-X - 2)
    else:
        return 2 * np.sin(2 * X) + X - 11


def function_1dim_2(X):
    return 10 * np.sin(2 * X) - X ** 2 - 10


def function_2dim(X):
    return np.sin(X[:, 0]) + 3 * np.cos(X[:, 1])


# TODO: KERNEL PARAMETER ÄNDERN -> SONST SEHR SCHLECHTE ERGEBNISSE
def function_2dimm(X):
    if False: # ANDERES BEISPIEL
        return (X[:, 0] + 2 * X[:, 1] - 7)** 2 + (2 * X[:, 0] + X[:, 1] - 5)**2 #Booth function (watch Wikipedia) ABER ZU HOHE WERTE
    #value = 100 * np.sqrt(np.abs(X[:, 1] - 0.01 * X[:, 0] ** 2)) + 0.01 * np.abs(X[:, 0] + 10)
    return 100 * np.sqrt(np.abs(X[:, 1] - 0.01 * X[:, 0] ** 2)) + 0.01 * np.abs(X[:, 0] + 10)


def function_3dim(X):
    return np.sin(X[:, 0]) + 3 * np.cos(X[:, 1]) + np.sin(X[:, 2])


def function_4dim(X):
    return np.sin(X[:, 0]) + 3 * np.cos(X[:, 1]) + np.sin(X[:, 2]) + 3 * np.cos(X[:, 3])


# KERNEL - FUNCTION
if True:
    def rbf_kernel(X1, X2, l=1.0, v=1.0):
        n_samples_1, n_features = X1.shape
        n_samples_2, _ = X2.shape
        K = np.zeros((n_samples_1, n_samples_2))
        for i in range(n_samples_1):
            for j in range(n_samples_2):
                diff = X1[i] - X2[j]
                K[i, j] = v * np.exp(-0.5 * np.dot(diff, diff) / (l ** 2))
        return K


    def periodic_kernel(x1, x2, ls=1, p=1):
        n1, d = x1.shape
        n2, d = x2.shape
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                diff = np.sin(np.pi * np.linalg.norm(x1[i] - x2[j]) / p)
                K[i, j] = np.exp(-2 * (diff ** 2) / (ls ** 2))
        return K


    def linear_kernel(x1, x2):
        return np.dot(x1, x2.T)
