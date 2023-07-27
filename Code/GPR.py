import numpy as np


class GaussProcessRegression:
    def __init__(self, kernel, function, dim, X_init, Y_init, SSN):
        """
            Initialize the GaussianProcessRegressor.

            Inputs:
                - kernel:   Kernel function used for the Gaussian process regression
                - function: Target function to approximate using Gaussian process regression
                - dim:      Dimensionality of the input space
                - X_init:   Initial observed input data.
                - Y_init:   Initial observed output data.
                - SSN:      Tuple (start, stop, num)
                            specifying the range and number of points in each dimension for
                            creating a meshgrid := RoI (Region of Interest)
        """
        self.kernel = kernel  # set kernel function
        self.function = function  # set target function

        self.X_observed = X_init  # observed input data
        self.Y_observed = Y_init  # observed output data

        self.fstar, self.Sstar = None, None  # Initialize lists for storing results in each iteration
        linspaces = []
        start, stop, num = SSN  # Unpack the start, stop, and num values from SSN
        for i in range(dim):
            linspaces.append(np.linspace(start, stop, num))
        self.grids = np.meshgrid(*linspaces)  # Region of Interest

        # Create a array of all possible combinations of grid points
        self.Xstar = np.vstack([grid.flatten() for grid in self.grids]).T


    def gp(self):
        """
            Perform gaussian process regression.

            Returns:
                - fstar: Predicted (mean) function values
                - Sstar: Predictive covariance matrix.
        """
        K = self.kernel(self.X_observed, self.X_observed)
        K_starstar = self.kernel(self.Xstar, self.Xstar)
        K_star = self.kernel(self.X_observed, self.Xstar)
        K_star_ = self.kernel(self.Xstar, self.X_observed)
        try:
            # Do Cholesky decomposition of K with jittering for numeric stability
            # jittering := adding a small value to diagonal elements to guarantee pos. definit
            # done by: + 1e-8 * np.eye(self.K.shape[0])
            L = np.linalg.cholesky(K + 1e-8 * np.eye(K.shape[0]))
            Lk = np.linalg.solve(L, K_star)

            # Calculate the predicted function values and covariance matrix
            self.fstar = np.dot(Lk.T, np.linalg.solve(L, self.Y_observed))
            self.Sstar = K_starstar - np.dot(Lk.T, Lk)

            return self.fstar, self.Sstar

        except np.linalg.LinAlgError:
            print("Matrix inversion failed with Cholesky")
            try:
                K_inv = np.linalg.inv(K + 1e-8 * np.eye(K.shape[0]))  # Calculate the inverse of K with jittering
                # Calculate the predicted function values and covariance matrix
                self.fstar = np.dot(self.kernel(self.Xstar, self.X_observed), np.dot(K_inv, self.Y_observed))
                self.Sstar = K_starstar - np.dot(K_star_, np.dot(K_inv, K_star))

                return self.fstar, self.Sstar

            except np.lianlg.LinAlgError:
                print("Matrix inversion failed")
                return None, None

    def reset_init_data(self, X_init_new, Y_init_new):
        """
        Reset the known data.

        Inputs:
            - X_init_new: New observed input data.
            - Y_init_new: New observed output data.
        """
        self.X_observed = X_init_new
        self.Y_observed = Y_init_new

    # UPDATE THE DATA
    def update_data(self, x_new, y_new):
        """
            Update the known data with new data point.

            Inputs:
                - x_new: New input data point.
                - y_new: New output data point.
        """
        self.X_observed = np.vstack([self.X_observed, x_new])
        self.Y_observed = np.append(self.Y_observed, y_new)

    # def get_predicted_value_for_x_test_indices(self, indices_of_test_data):
    #     y_pred = self.fstar.flatten()[indices_of_test_data]  # prediction
    #     return y_pred
