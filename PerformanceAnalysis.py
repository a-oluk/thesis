
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist


class Metrics:
    def get_rsme_r2_scores(self, y_pred, y_true):
        """
        Calculates and returns the root mean squared error (RSME) and R-squared (R2) scores

        Inputs:
            - y_pred: Predicted values
            - y_true: True values

        Returns:
            - rsme: Root mean squared error
            - r2: R-squared score
        """
        rsme = self.get_rsme(y_pred, y_true)
        r2 = self.get_r2_score(y_pred, y_true)
        return rsme, r2

    def get_rsme(self, y_pred, y_true):
        """
        Calculates and returns the RSME score

        Inputs:
            - y_pred: Predicted values
            - y_true: True values

        Returns:
            - rsme_score: Root mean squared error score
        """
        squared_error = (y_true - y_pred) ** 2
        rsme_score = np.sum(squared_error / len(squared_error))
        return rsme_score

    def get_r2_score(self, y_pred, y_true):
        """
        Calculates and returns the R-squared (R2) score

        Inputs:
            - y_pred: Predicted values
            - y_true: True values

        Returns:
            - r2_score: R-squared score
        """
        mean_true = sum(y_true) / len(y_true)
        ss_tot = sum((y_true - mean_true) ** 2)
        ss_res = sum((y_true - y_pred) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    def get_r2_difference_to_before(self, i, x_obs, x_test, y_pred_new, y_pred_old):
        """
        Calculates the difference in R-squared (R2) scores between two iterations and visualizes it.

        -- not in use --

        Inputs:
            - i: Current iteration number.
            - x_obs: Observed input data.
            - x_test: Test input data.
            - y_pred_new: Predicted values in the current iteration.
            - y_pred_old: Predicted values in the previous iteration.
        """
        plt.figure()
        plt.scatter(x_obs[:, 0], x_obs[:, 1], c="black")
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred_new - y_pred_old, cmap='coolwarm')
        plt.title("Iteration {}".format(i + 1))
        plt.colorbar()


    # Calculate the average score of iteration over all repetitions
    def calculate_average_scores(self, r2_score__, rsme_score__):
        """
        Calculates the average r2 and RSME scores over all repetitions.

        Inputs:
            - r2_score__: List of R2 scores for each repetition
            - rsme_score__: List of RSME scores for each repetition

        Returns:
            - r2__average: Average R2 scores of iterations
            - rsme__average: Average RSME scores of iterations
        """
        length_ = len(r2_score__[0])
        length__ = len(r2_score__)
        r2__average, rsme__average = [0] * length_, [0] * length_

        for i in range(length__):
            # Loop over each array
            for j in range(length_):
                # Add the value at the current position to the running total
                r2__average[j] += r2_score__[i][j]
                rsme__average[j] += rsme_score__[i][j]

        r2__average = [x / length__ for x in r2__average]
        rsme__average = [x / length__ for x in rsme__average]

        return r2__average, rsme__average


class Acquisition:
    def __init__(self, ei=True, gf=True, normalize=True, alpha=0.5):
        """
        Initializes the Acquisition object.

        Attributes:
           - ei: Boolean value indicating whether to include the variance based (EI) term
           - gf: Boolean value indicating whether to include the global fit (GF) term
           - normalize: Boolean value indicating whether to normalize ei/gf
           - alpha: Weight parameter for balancing the EIGF (Expected Improvement for Global Fit).
        """
        self.ei = ei
        self.gf = gf
        self.normalize = normalize
        self.alpha = alpha

    def get_acquisition_function(self, Xstar, fstar, x_obs, y_obs, Sstar, shape_like):
        """
        Calculates and returns the acquisition function.

        Inputs:
            - Xstar: Input matrix for prediction.
            - fstar: Mean prediction vector.
            - x_obs: Observed input data.
            - y_obs: Observed output data.
            - Sstar: Covariance matrix for prediction.
            - shape_like: Shape of the output acquisition function.

        Returns:
            - aquisition_function: Acquisition function.
        """
        def get_ei_term(Sstar, shape_like):
            """
            Calculates the Expected Improvement (EI) term of the acquisition function.

            Inputs:
                - Sstar: Covariance matrix for prediction.
                - shape_like: Shape of the output EI term.

            Returns:
                - ei_term: Expected Improvement (EI) term.
            """
            Sstar_diag = np.diag(Sstar)
            ei_term = Sstar_diag.reshape(shape_like)
            return ei_term

        def get_gf_term(X_star, fstar, x_obs, y_obs, shape_like):
            """
            Calculates the global fit term of the acquisition function.

            Inputs:
                - X_star: Input matrix for prediction.
                - fstar: Mean prediction vector.
                - x_obs: Observed input data.
                - y_obs: Observed output data.
                - shape_like: Shape of the output GF term.

            Returns:
                - gf_term: global fit (GF) term.
            """

            dist_to_nn = cdist(X_star, x_obs, metric='euclidean')
            min_idx = np.argmin(dist_to_nn, axis=1)
            nn_values = y_obs[min_idx]

            Z = fstar.reshape(shape_like)
            nn_values = nn_values.reshape(shape_like)

            gf_term = (Z - nn_values) ** 2
            return gf_term

        def norm_matrix(matrix):
            """
            Normalizes a matrix by dividing it by its maximum value.

            Inputs:
                - matrix: Input matrix.

            Returns:
                - matrix_norm: Normalized matrix.
            """
            max_val = matrix.max()
            matrix_norm = np.divide(matrix, max_val)
            return matrix_norm

        aquisition_function = np.zeros(shape_like)

        if self.ei:
            ei_term = get_ei_term(Sstar, shape_like)
            if self.normalize:
                ei_term = norm_matrix(ei_term)
            aquisition_function += self.alpha * ei_term

        if self.gf:
            gf_term = get_gf_term(Xstar, fstar, x_obs, y_obs, shape_like)
            if self.normalize:
                gf_term = norm_matrix(gf_term)
            aquisition_function += (1 - self.alpha) * gf_term

        return aquisition_function

    def get_new_x_for_eigf(self, aquisition_function, Xstar):
        """
        Determines the new sample point based on the acquisition function.
        Instead of optimization algorithm, get max value.

        Inputs:
            - aquisition_function: Acquisition function
            - Xstar: Input matrix for prediction

        Returns:
            - new_x: New sample point
        """
        max_index = np.argmax(aquisition_function)
        indices = np.unravel_index(max_index, np.shape(aquisition_function))
        idx = np.ravel_multi_index(indices, aquisition_function.shape)
        return np.array([Xstar[idx]])
