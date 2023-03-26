import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def funct_3d(x1,x2):
    return np.sin(x1)+x2*np.cos(x2)

def exponential_cov(X, x_star, params):
    # Compute the pairwise distances between the inputs and the point x_star
    dist = np.sqrt(np.sum((X - x_star) ** 2, axis=1))
    # Compute the covariance using the radial basis function kernel
    cov = params[0] * np.exp(-0.5 * (dist / params[1]) ** 2)
    return cov


def conditional(x_new, X, y, params):
    B = exponential_cov(x_new, X, params)
    C = exponential_cov(X, X, params)
    A = exponential_cov(x_new, x_new, params)
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    return mu.squeeze(), sigma.squeeze()


##########################################################################################

# Example usage
dim = 2
X = np.zeros((1, dim))  # 100 data points in 5 dimensions AND # single point in 5 dimensions

params = [1, 1]  # variance and lengthscale

cov = exponential_cov(X, X, params)



x1 = np.arange(-3, 3, 0.5)
x2 = np.arange(-3, 3, 0.5)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')
Z = np.zeros_like(X1)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X1,X2,Z)
ax.errorbar(X1.flatten(), X2.flatten(), Z.flatten(),0.2, errorevery=2, linestyle='')
plt.show()


