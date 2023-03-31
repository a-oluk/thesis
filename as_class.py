import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

from functions import *


class GaussianProcessRegressor:
    def __init__(self, kernel, function, dim, X, SSN):
        self.N_gpr_iterations = 30
        self.kernel = kernel
        self.alpha = None  # TODO: WO GENAU ALPHA EINFÜHREN ?!
        self.function = function

        # INITIAL DATA
        self.X = X
        if dim == 1:
            self.Y = np.array([self.function(i) for i in self.X])
        else:
            self.Y = self.function(self.X)

        self.fstar, self.Sstar = [], []  # results in iteration
        self.fstar_, self.Sstar_ = [], []  # Save results from iterations in list
        self.fstar__, self.Sstar__ = [], []  # Save results from repetition
        ################################################
        self.linspaces = []
        start, stop, num = SSN
        # AREA OF INTEREST
        for i in range(dim):
            # self.linspaces.append(np.arange(start, stop + 1, steps))
            self.linspaces.append(np.linspace(start, stop, num))
        self.grids = np.meshgrid(*self.linspaces)
        self.Xstar = np.vstack([grid.flatten() for grid in self.grids]).T
        self.grid_shape = self.grids[0].shape
        self.K = None

    def GPR(self):
        self.K = self.kernel(self.X, self.X)
        # jittering is used here, to ensure the matrix is pos definit (numerical problem)
        try:
            L = np.linalg.cholesky(self.K + 1e-8 * np.eye(self.K.shape[0]))  # with jittering
            Lk = np.linalg.solve(L, self.kernel(self.X, self.Xstar))
            self.fstar = np.dot(Lk.T, np.linalg.solve(L, self.Y))

            Kstar = self.kernel(self.Xstar, self.Xstar)
            # Kstar += np.eye(len(self.Xstar))
            self.Sstar = Kstar - np.dot(Lk.T, Lk)

            return self.fstar, self.Sstar

        except np.linalg.LinAlgError:
            print("Matrix is not positive definite")
            try:
                Kinv = np.linalg.inv(self.K)
                self.fstar = np.dot(self.kernel(self.Xstar, self.X), np.dot(Kinv, self.Y))

                # calculate variance
                Kstar = self.kernel(self.Xstar, self.Xstar)
                self.Sstar = Kstar - np.dot(self.kernel(self.Xstar, self.X),
                                            np.dot(Kinv, self.kernel(self.X, self.Xstar)))

                return self.fstar, self.Sstar
            except np.lianlg.LinAlgError:
                print("Matrix inversion failed")
                return None
        return None

        ################################################

    def reset(self, X, dim):
        # TODO: RESET FOR REPETITION
        self.X = X
        if dim == 1:
            self.Y = np.array([self.function(i) for i in self.X])
        else:
            self.Y = self.function(self.X)
        self.fstar_, self.Sstar_ = [], []
        print("RESET WORKED")

    # UPDATE THE DATA
    def update_data(self, x_new, ):
        self.X = np.vstack([self.X, x_new])

        y_new = self.function(x_new)
        self.Y = np.append(self.Y, y_new)
        return self.X, self.Y

    # HIER IST IRGENDEIN FEHLER
    if False:
        def fit(self, X, y):
            K = self.kernel(self.X, self.X)
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(K.shape()))
            except np.linalg.LinAlgError:
                try:
                    # if Cholesky fails, try normal inversion
                    L = np.linalg.inv(K)
                except np.linalg.LinAlgError:
                    print(" INVERSION NOT POSSIBLE ")
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

        def predict(self, X_star):
            K = self.kernel(self.X, self.X)
            K_star = self.kernel(self.X, X_star)
            self.y_mean = np.dot(K_star.T, self.alpha)
            K_star_star = self.kernel(X_star, X_star)
            try:
                L = np.linalg.cholesky(
                    (K_star_star - np.dot(K_star.T, np.linalg.solve(K, K_star))) + 1e-8 * np.eye(K.shape()))
            except np.lianlg.LinAlgError:
                # if Cholesky fails, try normal inversion
                L = np.linalg.inv(K_star_star - np.dot(K_star.T, np.linalg.solve(K, K_star)))
            self.y_std = np.sqrt(
                np.diag(K_star_star - np.dot(K_star.T, np.linalg.solve(K, K_star))) - np.diag(np.dot(L.T, L)))
            return self.y_mean, self.y_std
            #################################################


''' HELPFUNCTIONS '''
# KERNEL - FUNCTION
if True:
    def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
        n_samples_1, n_features = X1.shape
        n_samples_2, _ = X2.shape
        K = np.zeros((n_samples_1, n_samples_2))
        for i in range(n_samples_1):
            for j in range(n_samples_2):
                diff = X1[i] - X2[j]
                K[i, j] = sigma_f * np.exp(-0.5 * np.dot(diff, diff) / (l ** 2))
        return K

# AQUISITION FUNCTION
if True:
    def get_ei_term(Sstar, shape_like):
        Sstar_diag = np.diag(Sstar)
        ei_term = Sstar_diag.reshape(shape_like)
        return ei_term
    def get_gf_term(meshgrid_vstacked, fstar, x_obs, y_obs, shape_like):
        meshgrid_arr = meshgrid_vstacked

        dist_to_nn = cdist(meshgrid_arr, x_obs, metric='euclidean')
        min_idx = np.argmin(dist_to_nn, axis=1)
        nn_values = y_obs[min_idx]

        # Reshape the arrays to match the shape of the meshgrid
        Z = fstar.reshape(shape_like)  # reshape the mean of it from GPR
        nn_values = nn_values.reshape(shape_like)

        # Calculate the absolute difference between each point and its nearest neighbor value
        diff_arr = (Z - nn_values) ** 2
        return diff_arr
    def norm_matrix(matrix):
        max_val = matrix.max()
        # A_norm = A / max_val slower than np.divide
        matrix_norm = np.divide(matrix, max_val)
        return matrix_norm
    def get_new_x_for_eigf(aquisition_function, Xstar):
        max_index = np.argmax(aquisition_function)
        # Convert the 1D index to 2D indices
        indices = np.unravel_index(max_index, np.shape(aquisition_function))
        idx = np.ravel_multi_index(indices, aquisition_function.shape)
        return np.array([Xstar[idx]])


# EVALUATION DATA
def data_for_evaluation(Xstar, fstar, function, dim, test_size=30):
    indices = np.random.choice(fstar.size, size=test_size, replace=False)
    y_pred = fstar.flatten()[indices]
    # y_true = function(Xstar[indices]) FUNKTIONIERT NICHT GUT
    if dim == 1:  # DIM 1
        y_true = np.array([function(i) for i in Xstar[indices]])[:, 0]  # WEIL DOPPELTE ARRAY TODO: ALTERNATIVE
    else:
        y_true = function(Xstar[indices])
    return y_pred, y_true


def get_rsme(y_pred, y_true):
    squared_error = ((y_true) - y_pred) ** 2
    rsme = np.sum(squared_error / len(squared_error))
    return rsme


def get_r2_score(y_pred, y_true):
    mean_true = sum(y_true) / len(y_true)

    ss_tot = sum((y_true - mean_true) ** 2)

    ss_res = sum((y_true - y_pred) ** 2)

    score = 1 - (ss_res / ss_tot)

    return score

def calculate_average_scores(r2_score__, rsme_score__):
    length_ = len(r2_score__[0])
    length__ = len(r2_score__)
    r2_score__avarage, rsme_score__avarage = [0] * length_, [0] * length_

    for i in range(length__):
        # Loop over each array
        for j in range(length_):
            # Add the value at the current position to the running total
            r2_score__avarage[j] += r2_score__[i][j]
            rsme_score__avarage[j] += rsme_score__[i][j]

    r2_score__avarage = [x / length__ for x in r2_score__avarage]
    rsme_score__avarage = [x / length__ for x in rsme_score__avarage]

    return r2_score__avarage, rsme_score__avarage


def plot_scores(x, r2, rsme, avarage=False, i=None):
    if avarage == False:
        fig, (ax2, ax1) = plt.subplots(1, 2)

        ax1.plot(x, r2)
        ax2.plot(x, rsme)

        fig.suptitle("Repetition number {}".format(i))

        # Set the plots
        ax1.set_title("R2 score over iteration")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("R2 Score")
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=1, ls="--", c="grey")

        ax2.set_title("RSME score over iteration")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("RSME Score")

    if avarage == True:
        fig, (ax2, ax1) = plt.subplots(1, 2)

        ax1.plot(x, r2)
        ax2.plot(x, rsme)

        fig.suptitle("Scores over {} repetition(s)".format(i))

        ax1.set_title("Avarage R2 score")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("R2 Score")
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=1, ls="--", c="grey")

        ax2.set_title(" Avarage RSME score")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("RSME Score")
    plt.show()
    return None


def start_gpr(kernel=rbf_kernel, function=function_2dim, SSN=(-5, 5, 100), dim=2, N_0=5, repetition=5, gp_iterations=20,
              test_size=30,
              alpha=0.5):
    if alpha < 0 or alpha > 1:
        user_input = input("Enter a value between 0 and 1: ")
        alpha = float(user_input)
        # raise TypeError("Wanted strawberry, got grape.")
    start, stop, num = SSN  # SSN
    X = np.random.uniform(start, stop, (N_0, dim))  # INITIAL DATA

    # generate a GPR Object, which use the kernels
    gpr = GaussianProcessRegressor(kernel, function, dim, X, (start, stop, num))

    fstar_i, Sstar_i = gpr.GPR()
    # List to save results for each iteration

    gpr.fstar_.append(fstar_i)
    gpr.Sstar_.append(Sstar_i)

    evaluation__ = []

    r2_score__, rsme_score__ = [], []
    for p in range(repetition):
        r2_score_, rsme_score_ = [], []  # RESET THE SCORES
        for q in range(gp_iterations):
            print("Start Iteration Number: ", q + 1)
            ei_term = get_ei_term(Sstar_i, shape_like=gpr.grid_shape)
            gf_term = get_gf_term(gpr.Xstar, fstar_i, x_obs=gpr.X, y_obs=gpr.Y, shape_like=gpr.grid_shape)

            ei_normalized = norm_matrix(ei_term)
            gf_normalized = norm_matrix(gf_term)

            aquisition_func = alpha * ei_normalized + (
                    1 - alpha) * gf_normalized  # Alpha 1 => AQUIDISTANT # Alpha 0 => JUST Y wert

            x_new = get_new_x_for_eigf(aquisition_func, gpr.Xstar)

            gpr.update_data(x_new)  # add the new data point to dataset

            fstar_i, Sstar_i = gpr.GPR()
            # ITERATION

            # SAVE fstar/Sstar
            gpr.fstar_.append(fstar_i)
            gpr.Sstar_.append(Sstar_i)

            y_pred, y_true = data_for_evaluation(gpr.Xstar, gpr.fstar, function, dim, test_size)
            r2_i = get_r2_score(y_pred, y_true)
            r2_score_.append(r2_i)

            rsme_i = get_rsme(y_pred, y_true)
            rsme_score_.append(rsme_i)
        ############
        # FINAL VALUE AT THE END
        evaluation__.append([rsme_i, r2_i])

        # SAVE THE LISTS OVER WHOLE ITERATIONS IN __
        r2_score__.append(r2_score_)
        rsme_score__.append(rsme_score_)
        ############
        plot_scores(np.arange(gp_iterations), r2_score__[p], rsme_score__[p], avarage=False, i=p + 1)

        # PLOT RESULTS FOR 1 and 2 DIM
        if dim == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            c1 = ax1.imshow(fstar_i.reshape(gpr.grid_shape), origin='lower', extent=[start, stop, start, stop],
                            cmap='coolwarm')
            ax1.scatter(gpr.X[:, 0], gpr.X[:, 1], c=gpr.Y, cmap='coolwarm', edgecolors='black', linewidths=1)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(c1, cax=cax, orientation='vertical')

            c2 = ax2.imshow(np.diag(Sstar_i).reshape(gpr.grid_shape), origin='lower', extent=[start, stop, start, stop],
                            cmap='coolwarm')
            ax2.scatter(gpr.X[:, 0], gpr.X[:, 1], c=gpr.Y, cmap='coolwarm', edgecolors='black', linewidths=1)

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(c2, cax=cax, orientation='vertical');

            ax1.set_xlabel('X1')
            ax2.set_xlabel('X1')
            ax1.set_ylabel('X2')

            fig.suptitle('GPR  Iteration: {}'.format(gp_iterations))
            for i, txt in enumerate(range(len(gpr.X[:, 0]))):
                if i > N_0 - 1:
                    ax1.annotate(txt + 1 - N_0, (gpr.X[i, 0], gpr.X[i, 1]))
                    ax2.annotate(txt + 1 - N_0, (gpr.X[i, 0], gpr.X[i, 1]))
            plt.show()

        elif dim == 1:
            x = gpr.grids[0]
            y = np.array([function(i) for i in gpr.grids[0]])

            plt.figure()
            plt.scatter(gpr.X, gpr.Y)
            plt.plot(x, gpr.fstar)
            plt.errorbar(x, fstar_i, yerr=np.diag(Sstar_i))

            plt.plot(x, y)

            for i, txt in enumerate(range(len(gpr.X))):
                if i > N_0 - 1:
                    plt.annotate(txt + 1, (gpr.X[i], gpr.Y[i]))
            plt.show()
        gpr.reset(X, dim)

    # calculate the average scores for every iteration and plot these
    r2_score__avarage, rsme_score__avarage = calculate_average_scores(r2_score__, rsme_score__)
    plot_scores(np.arange(gp_iterations), r2_score__avarage, rsme_score__avarage, avarage=True, i=p + 1)

    return gpr, evaluation__


random.seed(20)
# TWO DIMENSIONAL
# gpr_2dim, evaluation = start_gpr(kernel=rbf_kernel, function=function_2dim, dim=2, N_0=10, repetition=1,gp_iterations=30, test_size=30, alpha=0.5, SSN=(-5, 5, 40))

#########################################################
# TODO: HIER SEHR SCHLECHTE ERGEBNISSE -> DA STARK VERÄNDERLICH | KERNEL PARAMETER ÄNDERN BASED ON ABSTAND VON PUNKTEN !
#  !!      BASED ON OBSERVED POINTS ABSTAND (MAX oder MIN ABSTAND) EVENTUELL
#  !! ODER BASED --> GEHT IMMER ZU SEHR AUF NULL ZURÜCK !!!!!!!!!!!!!
#gpr_2dimm, evaluation = start_gpr(kernel=rbf_kernel, function=function_2dimm, SSN=(-10, 10, 40), dim=2, N_0=10, repetition=1,gp_iterations=30, test_size=30, alpha=0.5)
##########################################################

# ONE DIMENSIONAL
#gpr_1dim, evaluation = start_gpr(function=function_1dim, SSN=(-10, 10, 100), dim=1, N_0=5,repetition = 1, gp_iterations=30, test_size=30, alpha=0.5)
#gpr_1dim, evaluation = start_gpr(function=function_1dim_1, SSN=(-10, 10, 100), dim=1, N_0=5,repetition = 1, gp_iterations=30, test_size=30, alpha=0.5)
gpr_1dim, evaluation = start_gpr(function=function_1dim_2, SSN=(-10, 10, 100), dim=1, N_0=3, repetition = 1, gp_iterations=15, test_size=30, alpha=0.5)

# FOUR DIMENSIONAL #AUFLÖSUNG LEIDER NUR 5
# gpr_4dim, evaluation = start_gpr(function=function_4dim,SSN=(-5, 5, 3), dim=4, N_0=20, repetition=10, gp_iterations=10, test_size=10,alpha=0.5)


print(evaluation)
