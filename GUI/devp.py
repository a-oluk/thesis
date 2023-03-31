import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# import functions

# %%

# Just the test function
# f = lambda x: np.sin(0.9*x)*(x-2)
f_0 = lambda x: np.sin(5 * x) * x + (x - 2)


def f(x):
    if x < -5:
        return np.sin(3 * x) * x
    elif x >= -5 and x < 3:
        return (x - 2)
    else:
        return np.sin(2 * x) + x


# f = lambda x: np.sin(6*x)*0.4(x-2)

# %%

#### HIER KOMMT GAUS PROZESS REIN JETZT

'''Was gehört zu einem Gauss Prozess ? 
    -> 1.) Kernel-Function
    -> 2.) Gaus-Verteilung
            - Erwartungswert (mean)
            - Kovarianzfunktion (covariance) 
                - radial basis function
                - matern
                - kombination of 
    -> 3.) Neue Sampling Punkte bestimmen und Gaus Process neu anwenden
                
    Schritte: 
        1.) A priori Erwartungswertfunktion
        2.) A-priori Kovarianzfunktion
        3.) Feinabstimmung der parameter
        4.) Bedingte Verteilung
        5.) Interpretation
'''

# %% Not in Usage now

# not for my program yet
# from https://jcmwave.com/company/blog/item/1050-gaussian-process-regression
if False:
    X1 = ...
    X2 = ...


    def get_covariance_martices():
        # Covariance matrices
        Sigma11 = np.array([[rbf_kernel(x, y) for x in X1] for y in X1])
        Sigma12 = np.array([[rbf_kernel(x, y) for x in X2] for y in X1])
        Sigma21 = Sigma12.T
        Sigma22 = np.array([[rbf_kernel(x, y) for x in X2] for y in X2])
        return None


# %%  Kernel Function

def exponential_cov(x, y, params=[5, 10]):
    return params[0] * np.exp(-0.5 * params[1] * np.subtract.outer(x, y) ** 2)


def rbf_kernel(X, Y, var=1, gamma=1.0):
    """Radial basis function kernel"""
    # Calculate the squared Euclidean distance between all pairs of points
    sq_dists = np.sum(X * 2, 1).reshape(-1, 1) + np.sum(Y * 2, 1) - 2 * np.dot(X, Y.T)
    return var * np.exp(-gamma * sq_dists)


# %% Conditional Function  WARUM BENTUTZE ICH DAS NICHT ?!?!?!
def conditional(x_new, x, y, params):
    # B = exponential_cov(x_new, x, params)     # Kernel of oversavtion vs to-predict
    # C = exponential_cov(x, x, params)         # Kernel of observations
    # A = exponential_cov(x_new, x_new, params) # Kernel of to-predict

    B = rbf_kernel(x_new, x)  # Kernel of oversavtion vs to-predict
    C = rbf_kernel(x, x)  # Kernel of observations
    A = rbf_kernel(x_new, x_new)  # Kernel of to-predict

    mu = np.linalg.inv(C).dot(B.T).T.dot(y)  # mean
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))  # covariance
    return (mu.squeeze(), sigma.squeeze())


# %% Prediction
def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data]
    Sinv = np.linalg.inv(sigma)  # inverse of matrix -> use 2 times
    y_pred = np.dot(k, Sinv).dot(t)  # k * sigma^-1 * t
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)  # COV_xx - K*sigma^-1*k
    return y_pred, sigma_new


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]  # RETURN x WErt


def get_dist_normalized(f, x, x_pred):
    for i in np.arange(len(x_pred)):
        if i == 0:
            nn_first_value = find_nearest(x, x_pred[i])
            nn_value = [f(nn_first_value)]
        else:
            nn_value.append(f(find_nearest(x, x_pred[i])))

    dist = (nn_value - y_pred) ** 2
    dist_normalized = dist / dist.max()
    return dist_normalized


def get_sigmas_normalized(sigmas):
    sigmas_normalized = np.divide(sigmas, sigmas.max())
    return sigmas_normalized


###################################################################################################
def get_plots(x_pred, y_pred, sigmas, sigmas_normalized, dist_normalized, alpha):
    fig, axes = plt.subplots(4, 1, figsize=(6, 5))
    axes[0].errorbar(x_pred, y_pred, yerr=sigmas)
    axes[0].plot(x, y, "ro")
    # axes[0].plot(x_pred, y_pred,'black',linewidth=3) # PREDICTED FUNCTION
    # axes[0].plot(xpts,f(np.array(xpts))) # NUR FUNKTION

    axes[1].set_title("Gaussian Process")
    axes[1].plot(x_pred, sigmas_normalized)
    axes[1].set_title("Aquisition Function")

    axes[2].set_title("DIST ")
    axes[2].plot(x_pred, dist_normalized)  # +2*sigmas) Mit sigma angepasst
    axes[2].set_title("EIGF, y Term")

    axes[3].plot(x_pred, alpha * dist_normalized + (1 - alpha) * sigmas_normalized)  # +2*sigmas) Mit sigma angepasst
    axes[3].set_title("Combined with alpha = {}".format(alpha))


# %%###################################################################################################


# %%###################################################################################################
# HYPERPARAMETER
N = 10
N_0 = 3
θ = [5, 10]  # parameters for COV Variance / Lengthscale
a, b = -10, 10  # Interval to both sides
num_gaussian_points = (np.abs(a) + np.abs(b)) * 50  # for -10, 10 = 1000 points

alpha = 0.5 # weight for the eigf

######################################################################################################

σ_0 = exponential_cov(0, 0, θ)  # Cov = sigma = kernel

# linspace for the gaussian
x_pred = np.linspace(a, b, num_gaussian_points)
f_real_func = np.array([f(i) for i in x_pred])

# plt the a priori
plt.figure()
plt.errorbar(x_pred, np.zeros(len(x_pred)), yerr=σ_0)  # Errorbar (Gaussian Process)
plt.plot(x_pred, f_real_func)

# %% Random Dataset
np.random.seed(10)
x = np.random.uniform(a, b, N_0).tolist()
y = np.array([f(i) for i in x]).tolist()  # FOR INITIAL DATASET

# %%

for i in range(N + 1):

    if i == 0:  # FOR INITAL DATASET
        print("initial Data")
        σ_1 = exponential_cov(x, x, θ)  # get new Cov for x Data but same params

        predictions = [predict(i, x, exponential_cov, θ, σ_1, y) for i in x_pred]
        y_pred, sigmas = np.transpose(predictions)

        sigmas_normalized = get_sigmas_normalized(sigmas)
        dist_normalized = get_dist_normalized(f, x, x_pred)

        get_plots(x_pred, y_pred, sigmas, sigmas_normalized, dist_normalized, alpha)

    else:
        print("Iteration Number: {}".format(i))

        eigf_values = alpha * dist_normalized + (1 - alpha) * sigmas_normalized  # alpha increas -> more exploration
        # HIER MAX AUSWÄHLEN; ABER ANPASSEN !
        df = pd.DataFrame(data=(zip(x_pred, eigf_values)), columns=['x_pred', 'eigf'])

        x_new = df.loc[df['eigf'].idxmax()].x_pred
        x.append(x_new)
        y_new = f(x_new)
        y.append(y_new)

        σ_new = exponential_cov(x, x, θ)
        predictions = [predict(i, x, exponential_cov, θ, σ_new, y) for i in x_pred]
        y_pred, sigmas = np.transpose(predictions)

        sigmas_normalized = get_sigmas_normalized(sigmas)
        dist_normalized = get_dist_normalized(f, x, x_pred)

        get_plots(x_pred, y_pred, sigmas, sigmas_normalized, dist_normalized, alpha)

plt.show()
