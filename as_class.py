import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from mpl_toolkits.axes_grid1 import make_axes_locatable

from eval import *
import random

from functions import *


class GaussianProcessRegressor:
    def __init__(self, kernel, function, dim, X_init,Y_init, SSN):
        self.kernel = kernel
        self.function = function

        self.rep_nr = 1  # # iteration number
        self.iter_nr = 0  # iteration number

        # Initial Data
        self.X_init = X_init
        self.Y_init = Y_init

        # Observed Data
        self.X = X_init
        self.Y = Y_init

        self.fstar, self.Sstar = [], []  # results in iteration
        self.fstar_, self.Sstar_ = [], []  # Save results from iterations in list

        ################################################
        linspaces = []
        start, stop, num = SSN
        # AREA OF INTEREST
        for i in range(dim):
            linspaces.append(np.linspace(start, stop, num))
        self.grids = np.meshgrid(*linspaces)

        self.Xstar = np.vstack([grid.flatten() for grid in self.grids]).T

        self.grid_shape = self.grids[0].shape

        self.K = None

    def GPR(self):
        self.K = self.kernel(self.X, self.X)

        try:
            # jittering := adding a small value to guarantee pos. definit by: + 1e-8 * np.eye(self.K.shape[0])
            L = np.linalg.cholesky(self.K + 1e-8 * np.eye(self.K.shape[0]))
            Lk = np.linalg.solve(L, self.kernel(self.X, self.Xstar))

            self.fstar = np.dot(Lk.T, np.linalg.solve(L, self.Y))

            Kstar = self.kernel(self.Xstar, self.Xstar)

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

        ################################################

    # keine anwendung für Abgabe
    def reset(self, X):
        self.X = X
        self.Y = self.function(self.X)

        # self.fstar_, self.Sstar_ = [], []

    def reset_init_data(self,X_init_new,Y_init_new):
        self.X = X_init_new
        self.Y = Y_init_new

    # UPDATE THE DATA
    def update_data(self, x_new,y_new):
        self.X = np.vstack([self.X, x_new])
        self.Y = np.append(self.Y, y_new)


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


def plot_scores(x, r2, rsme, average=False, i=None):
    show_scores_plotted = True
    if average == False:
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

    if average == True:
        fig, (ax2, ax1) = plt.subplots(1, 2)

        ax1.plot(x, r2)
        ax2.plot(x, rsme)

        fig.suptitle("Scores over {} repetition(s)".format(i))

        ax1.set_title("average R2 score")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("R2 Score")
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=1, ls="--", c="grey")

        ax2.set_title(" average RSME score")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("RSME Score")
    if show_scores_plotted:
        plt.show()
    return None

def get_init_data(function,dim,SSN, N_0,):
    start, stop, num = SSN
    X = np.random.uniform(start, stop, (N_0, dim))  # INITIAL DATA
    Y = function(X)
    return X,Y


def start_gpr(kernel=rbf_kernel, function=function_2dim, SSN=(-5, 5, 100), dim=2, N_0=5, repetition=5, gp_iterations=20,
              test_size=30,
              alpha=0.5):
    start,stop,num = SSN # für den PLOT SPÄTER
    show_plots = False
    if alpha < 0 or alpha > 1:
        user_input = input("Enter a value between 0 and 1: ")
        alpha = float(user_input)

    X,Y = get_init_data(function,dim,SSN,N_0)

    # OBJEKTE ERZEUGEN
    data = Data(X, Y)
    # generate a GPR Object, which use the kernels
    gpr = GaussianProcessRegressor(kernel, function, dim, X, (start, stop, num))
    eval = Eval(ei=True, gf=True, normalize=True, alpha=0.5)

    fstar_i, Sstar_i = gpr.GPR()
    # List to save results for each iteration

    # gpr.fstar_.append(fstar_i)
    # gpr.Sstar_.append(Sstar_i)

    # r2_score__, rsme_score__ = [], []
    for p in range(repetition):
        data.reset_scores()
        gpr.iter_nr = 1
        for q in range(gp_iterations):
            print(gpr.iter_nr)
            print("Start Iteration Number: ", q + 1)
            aquisition_func = eval.get_aquisition_function(gpr.Xstar, fstar_i, x_obs=gpr.X, y_obs=gpr.Y, Sstar=Sstar_i,
                                                           shape_like=gpr.grid_shape)
            x_new = eval.get_new_x_for_eigf(aquisition_func, gpr.Xstar)

            fstar_i, Sstar_i = gpr.GPR()

            # GET SCORES
            y_pred, y_true, indices = data.data_for_evaluation(Xstar= gpr.Xstar, fstar= gpr.fstar, function= function, dim= param.dim, test_size=param.test_data_size)
            rsme_i, r2_i = eval.get_rsme_r2_scores(y_pred, y_true)

            # SAVE fstar/Sstar
            gpr.update_data(x_new)  # add the new data point to dataset
            gpr.fstar_.append(fstar_i)
            gpr.Sstar_.append(Sstar_i)

            data.update_data_(fstar_i, Sstar_i, x_new, function(x_new))
            data.save_scores_(rsme_i, r2_i)

            gpr.iter_nr += 1

        ############ SAVE SCORE AFTER ITERATION DONE
        data.save_final_score(rsme_i, r2_i)  # FINAL VALUE AT THE END OF A REPETITION
        data.save_scores__()  # SAVE SCORES OF ALL ITERATIONS
        ############
        plot_scores(np.arange(gp_iterations), data.r2_score__[-1], data.rsme_score__[-1], average=False, i=p + 1)

        data.reset_scores()

        # PLOT RESULTS FOR 1 and 2 DIM
        if dim == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2)

            c1 = ax1.imshow(data.fstar_[-1].reshape(gpr.grid_shape), origin='lower', extent=[start, stop, start, stop],
                            cmap='coolwarm')
            ax1.scatter(data.X_obs[:, 0], data.X_obs[:, 1], c=data.Y_obs, cmap='coolwarm', edgecolors='black',
                        linewidths=1)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(c1, cax=cax, orientation='vertical')

            c2 = ax2.imshow(np.diag(data.Sstar_[-1]).reshape(gpr.grid_shape), origin='lower',
                            extent=[start, stop, start, stop],
                            cmap='coolwarm')
            ax2.scatter(data.X_obs[:, 0], data.X_obs[:, 1], c=data.Y_obs, cmap='coolwarm', edgecolors='black',
                        linewidths=1)

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(c2, cax=cax, orientation='vertical');

            ax1.set_xlabel('X1')
            ax2.set_xlabel('X1')
            ax1.set_ylabel('X2')

            fig.suptitle('GPR  Iteration: {}'.format(gp_iterations))
            for i, txt in enumerate(range(len(data.X_obs[:, 0]))):
                if i > N_0 - 1:
                    ax1.annotate(txt + 1 - N_0, (data.X_obs[i, 0], data.X_obs[i, 1]))
                    ax2.annotate(txt + 1 - N_0, (data.X_obs[i, 0], data.X_obs[i, 1]))
            if show_plots:
                plt.show()

        elif dim == 1:
            x = gpr.grids[0]
            y = function(x)

            plt.figure()
            plt.scatter(data.X_obs, data.Y_obs)
            plt.plot(x, data.fstar_[-1])
            plt.errorbar(x, data.fstar_[-1], yerr=np.diag(data.Sstar_[-1]))

            plt.plot(x, y)

            for i, txt in enumerate(range(len(data.X_obs))):
                if i > N_0 - 1:
                    plt.annotate(txt + 1, (data.X_obs[i], data.Y_obs[i]))
            if not show_plots:
                plt.show()
        gpr.reset(X, dim)
        data.reset(dim)
        gpr.rep_nr += 1

    # calculate the average scores for every iteration and plot these
    r2_score__average, rsme_score__average = eval.calculate_average_scores(data.r2_score__, data.rsme_score__)
    data.save_average_scores__(r2_score__average, rsme_score__average)

    plot_scores(np.arange(gp_iterations), data.r2_score__average, data.rsme_score__average, average=True, i=p + 1)

    return gpr, data


random.seed(20)
# TWO DIMENSIONAL
# gpr_2dim, evaluation = start_gpr(kernel=rbf_kernel, function=function_2dim, dim=2, N_0=10, repetition=1,gp_iterations=30, test_size=30, alpha=0.5, SSN=(-5, 5, 40))

#########################################################
# TODO: HIER SEHR SCHLECHTE ERGEBNISSE -> DA STARK VERÄNDERLICH | KERNEL PARAMETER ÄNDERN BASED ON ABSTAND VON PUNKTEN !
#  !!      BASED ON OBSERVED POINTS ABSTAND (MAX oder MIN ABSTAND) EVENTUELL
#  !! ODER BASED --> GEHT IMMER ZU SEHR AUF NULL ZURÜCK !!!!!!!!!!!!!
# gpr_2dimm, evaluation = start_gpr(kernel=rbf_kernel, function=function_2dimm, SSN=(-10, 10, 40), dim=2, N_0=10, repetition=1,gp_iterations=30, test_size=30, alpha=0.5)
##########################################################

# ONE DIMENSIONAL
# gpr_1dim, evaluation = start_gpr(function=function_1dim, SSN=(-10, 10, 100), dim=1, N_0=5,repetition = 1, gp_iterations=30, test_size=30, alpha=0.5)
#gpr_1dim, data = start_gpr(function=function_1dim_1, SSN=(-10, 10, 500), dim=1, N_0=5, repetition=2, gp_iterations=30, test_size=30, alpha=0.5)
# gpr_1dim, evaluation = start_gpr(function=function_1dim_2, SSN=(-10, 10, 100), dim=1, N_0=3, repetition=1, gp_iterations=15, test_size=30, alpha=0.5)

# FOUR DIMENSIONAL #AUFLÖSUNG LEIDER NUR 5
# gpr_4dim, evaluation = start_gpr(function=function_4dim,SSN=(-5, 5, 3), dim=4, N_0=20, repetition=10, gp_iterations=10, test_size=10,alpha=0.5)


#for i, values in enumerate(data.final_score__):
#    print(f"Repetition {i} : RSME: {values[0]} , R2: {values[1]}")

#print(data)
