import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

class dataset:
    def __init__(self):
        self.X_init = 0
        self.Y_init = 0

        self.x_obs = self.X_init
        self.y_obs = self.Y_init




    def data_for_evaluation(self, Xstar, fstar, function, dim, test_size=30):
        indices = np.random.choice(fstar.size, size=test_size, replace=False)
        y_pred = fstar.flatten()[indices]
        # y_true = function(Xstar[indices]) FUNKTIONIERT NICHT GUT
        if dim == 1:  # DIM 1
            y_true = np.array([function(i) for i in Xstar[indices]])[:, 0]  # WEIL DOPPELTE ARRAY TODO: ALTERNATIVE
        else:
            y_true = function(Xstar[indices])
        return y_pred, y_true




class eval:
    def __init__(self, ei=True, gf=True, normalize=True, alpha = 0.5):
        self.ei= ei
        self.gf= gf
        self.normalize = normalize
        self.alpha = alpha

    # Alpha 1 => AQUIDISTANT # Alpha 0 => JUST Y wert
    def get_aquisition_function(self,meshgrid_vstacked, fstar, x_obs, y_obs, Sstar, shape_like):
        if True: # HELPFUNCTIONS
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

        aquisition_function = np.zeros(shape_like)
        if self.ei:
            ei_term = get_ei_term(Sstar, shape_like)
            if self.normalize:
                ei_term = norm_matrix(ei_term)
            aquisition_function += self.alpha * ei_term

        if self.gf:
            gf_term= get_gf_term(meshgrid_vstacked, fstar, x_obs, y_obs, shape_like)
            if self.normlize:
                gf_term = norm_matrix(gf_term)
            aquisition_function += (1-self.alpha) * gf_term

        return aquisition_function

    def get_new_x_for_eigf(self, aquisition_function, Xstar):
        max_index = np.argmax(aquisition_function)
        # Convert the 1D index to 2D indices
        indices = np.unravel_index(max_index, np.shape(aquisition_function))
        idx = np.ravel_multi_index(indices, aquisition_function.shape)
        return np.array([Xstar[idx]])

    def get_rsme(self, y_pred, y_true):
        squared_error = ((y_true) - y_pred) ** 2
        rsme = np.sum(squared_error / len(squared_error))
        return rsme


    def get_r2_score(self, y_pred, y_true):
        mean_true = sum(y_true) / len(y_true)

        ss_tot = sum((y_true - mean_true) ** 2)

        ss_res = sum((y_true - y_pred) ** 2)

        score = 1 - (ss_res / ss_tot)

        return score


    def calculate_average_scores(self, r2_score__, rsme_score__):
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






