import numpy as np


class Data:
    def __init__(self, X_init, Y_init):
        """
        Initializes the Data object.

        Inputs:
            - X_init: Initial input data.
            - Y_init: Initial output data.
        """

        self.X_init = X_init
        self.Y_init = Y_init

        self.X_test = None
        self.Y_test = None

        # self.test_data_indices = None

        self.X_obs = self.X_init
        self.Y_obs = self.Y_init

        self.X__ = []
        self.Y__ = []

        self.fstar_, self.Sstar_ = [], []
        self.fstar__, self.Sstar__ = [], []

        self.r2_score_, self.rsme_score_ = [], []
        self.r2_score__, self.rsme_score__ = [], []

        self.r2_score__average, self.rsme_score__average = [], []

        self.final_score__ = []

        self.repetition = 1
        self.iteration = 0


    def reset(self):
        """
        Resets the observed data to the initial data.
        """
        self.X_obs = self.X_init
        self.Y_obs = self.Y_init

        self.fstar_, self.Sstar_ = [], []
        #print("RESET WORKED")

    # save results and add the new X and Y to observed points
    def save_data_(self, fstar_i, Sstar_i, x_new, y_new):
        """
        Saves the observed data and stores intermediate results.

        Inputs:
            - fstar_i: Mean prediction vector.
            - Sstar_i: Covariance matrix.
            - x_new: New input data point.
            - y_new: New output data point.

        Returns:
            - X_obs: Updated observed input data.
            - Y_obs: Updated observed output data.
        """
        self.fstar_.append(fstar_i)
        self.Sstar_.append(Sstar_i)
        self.X_obs = np.vstack([self.X_obs, x_new])
        self.Y_obs = np.append(self.Y_obs, y_new)

    def get_prediction(self, fstar, indices):
        """
           Retrieves the predicted values at the given indices
           -- not in use --

           Inputs:
           - fstar: Mean prediction vector.
           - indices: Indices of the desired predictions.

           Returns:
           - y_pred: Predicted values.
        """
        y_pred = fstar.flatten()[indices]
        return y_pred

    def create_test_data(self, test_data_size, dim , function, Xstar):
        indices = np.random.choice(Xstar.shape[0], size= test_data_size, replace=False)
        if dim == 1:  # DIM 1
            y_true = np.array([function(i) for i in Xstar[indices]])[:,0]  # respect double array structure
        else:
            y_true = function(Xstar[indices])
        self.X_test = Xstar[indices]
        self.Y_test = y_true
        return indices


    def save_scores_(self, rsme_i, r2_i):
        """
        Saves scores for the current iteration.

        Inputs:
            - rsme_i: RSME score.
            - r2_i: R-squared (R2) score.
        """
        self.rsme_score_.append(rsme_i)
        self.r2_score_.append(max(0,r2_i))

    def save_scores__(self):
        """
        Saves the RSME and R-squared (R2) scores for the current repetition.

        Returns:
            - None
        """
        self.r2_score__.append(self.r2_score_)
        self.rsme_score__.append(self.rsme_score_)

    # Update final score of the repetition
    def save_final_scores(self, rsme_i, r2_i):
        """
        Saves the final RSME and R-squared (R2) scores for the current repetition

        Inputs:
            - rsme_i: Final RSME score.
            - r2_i: Final R-squared (R2) score.

        Returns:
            - final_score__: Final scores
        """
        self.final_score__.append([rsme_i, r2_i])
        return self.final_score__

    def save_data__(self):
        """
        Saves the observed data and results of repetition

        Returns:
            - None
        """
        self.X__.append(self.X_obs)
        self.Y__.append(self.Y_obs)

        self.fstar__.append(self.fstar_)
        self.Sstar__.append(self.Sstar_)

    def save_average_scores__(self, r2__average, rsme__average):
        """
        Saves the average R2 and RSME scores over all repetitions

        Inputs:
            - r2__average: Average R-squared (R2) scores
            - rsme__average: Average RSME scores

        Returns:
            - None
        """
        self.r2_score__average = r2__average
        self.rsme_score__average = rsme__average

    def reset_scores(self):
        """
        Resets score lists for the next repetition

        Returns:
            - None
        """
        self.r2_score_, self.rsme_score_ = [], []

    def reset_init_data(self, X_init_new, Y_init_new):
        """
        Resets the initial data.

        Inputs:
            - X_init_new: New initial input data.
            - Y_init_new: New initial output data.

        Returns:
            - None.
        """
        self.X_init = X_init_new
        self.Y_init = Y_init_new

        self.X_obs = self.X_init
        self.Y_obs = self.Y_init

    def new_init_data(self, SSN, size, dim, function):
        """
        Generate new initial data.

        Inputs:
            - SSN:      RoI from tulpe (start,stop,num)
            - size:     Dize of initial dataset
            - dim:      Dimension
            - function: Target function

        Returns:
            - X_init_new: New initial input data
            - Y_init_new: New initial output data
        """
        start, stop, num = SSN
        X_init_new = np.random.uniform(start, stop, (size, dim))
        Y_init_new = function(X_init_new)
        return X_init_new, Y_init_new
