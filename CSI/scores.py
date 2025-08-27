from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone
from tqdm import tqdm


# defining score basic class
class Scores(ABC):
    """
    Base class to build any conformity score of choosing.
    In this class, one can define any conformity score for any base model of interest, already fitted or not.
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, is_fitted, **kwargs):
        self.is_fitted = is_fitted
        if self.is_fitted:
            self.base_model = base_model
        elif base_model is not None:
            self.base_model = base_model(**kwargs)

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the base model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   y: Training label vector.

        Output: Scores object
        """
        pass

    @abstractmethod
    def compute(self, X_calib, y_calib):
        """
        Compute the conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   y_calib: Calibration label vector

        Output: Conformity score vector
        """
        pass

    @abstractmethod
    def predict(self, X_test, cutoff):
        """
        Compute prediction intervals specified cutoff(s).
        --------------------------------------------------------
        Input: (i)    X_test: Test feature matrix
               (ii)   cutoff: Cutoff vector

        Output: Prediction intervals for test sample.
        """
        pass


# basic lambda score that does not need to be estimated
class LambdaScore(Scores):
    def fit(self, X, y):
        return self

    def compute(self, thetas, lambdas):
        return lambdas

    def predict(self, thetas, cutoff):
        pred = np.vstack((thetas - cutoff, thetas + cutoff)).T
        return pred


class RegressionScore(Scores):
    """
    Conformity score for regression (residuals)
    --------------------------------------------------------
    Input: (i)    base_model: Point prediction model object with fit and predict methods.
           (ii)   is_fitted: Boolean indicating if the regression model is already fitted.
    """

    def fit(self, X, y):
        """
        Fit the regression base model to training data
        --------------------------------------------------------
        Input: (i)    X: Training feature matrix.
               (ii)   y: Training label vector.

        Output: RegressionScore object
        """
        if self.is_fitted:
            return self
        elif self.base_model is not None:
            self.base_model.fit(X, y)
        else:
            return self

    def compute(self, X_calib, y_calib):
        """
        Compute the regression conformity score in the calibration set
        --------------------------------------------------------
        Input: (i)    X_calib: Calibration feature matrix
               (ii)   y_calib: Calibration label vector

        Output: Conformity score vector
        """
        if self.base_model is not None:
            pred = self.base_model.predict(X_calib)
            res = np.abs(pred - y_calib)
            return res
        else:
            return np.abs(y_calib)

    def predict(self, X_test, cutoff):
        """
        Compute prediction intervals using each observation cutoffs.
        --------------------------------------------------------
        Input: (i)    X_test: Test feature matrix
               (ii)   cutoff: Cutoff vector

        Output: Prediction intervals for test sample.
        """
        pred_mu = self.base_model.predict(X_test)
        pred = np.vstack((pred_mu - cutoff, pred_mu + cutoff)).T
        return pred


# main LFI statistics scores
# waldo score
class WaldoScore(Scores):
    def fit(self, X=None, thetas=None):
        # setting up model for normalizing flows
        if self.is_fitted:
            return self
        elif self.base_model is not None:
            self.base_model.fit(X, thetas)
        else:
            return self

    def compute(self, thetas, X, N=10**3, one_sample=False, disable_tqdm=False):
        # simulating from each theta and dataset to compute waldo statistic
        i = 0
        lambdas = np.zeros(thetas.shape[0])

        for theta in tqdm(
            thetas,
            desc="Computing waldo statistics using posterior model",
            disable=disable_tqdm,
        ):
            if not one_sample:
                # simulating from the model
                s = self.base_model.sample(X=X[i, :], num_samples=N)

                # computing E[theta|X]
                mean_theta_X = np.mean(s, axis=0)
                var_theta_x = np.cov(s, rowvar=False)

                # computing waldo statistic
                if mean_theta_X.ndim == 1:
                    if mean_theta_X.shape[0] > 1:
                        lambdas[i] = (
                            (mean_theta_X - theta).transpose()
                            @ np.linalg.inv(var_theta_x)
                            @ (mean_theta_X - theta)
                        )
                    else:
                        lambdas[i] = (mean_theta_X - theta) ** 2 / (var_theta_x)
                else:
                    lambdas[i] = (mean_theta_X - theta) ** 2 / (var_theta_x)
                i += 1
            else:
                # simulating from the model
                s = self.base_model.sample(X=X, num_samples=N)

                # computing E[theta|X]
                mean_theta_X = np.mean(s, axis=0)
                var_theta_x = np.cov(s, rowvar=False)

                # computing waldo statistic
                if mean_theta_X.ndim == 1:
                    if mean_theta_X.shape[0] > 1:
                        lambdas[i] = (
                            (mean_theta_X - theta).transpose()
                            @ np.linalg.inv(var_theta_x)
                            @ (mean_theta_X - theta)
                        )
                    else:
                        lambdas[i] = (mean_theta_X - theta) ** 2 / (var_theta_x)

                else:
                    lambdas[i] = (mean_theta_X - theta) ** 2 / (var_theta_x)
                i += 1

        return lambdas

    def predict(self, thetas_grid, X, cutoffs, theta_1d=True):
        # predicting lambdas for all thetas
        lambdas = self.compute(thetas_grid, X, one_sample=True)
        idxs = list(np.where(lambdas <= cutoffs)[0])

        if theta_1d:
            return thetas_grid.squeeze()[idxs]
        else:
            return thetas_grid[idxs, :]


# BFF score
class BFFScore(Scores):
    def fit(self, X=None, thetas=None):
        # setting up model for normalizing flows
        if self.is_fitted:
            return self
        elif self.base_model is not None:
            self.base_model.fit(X, thetas)
        else:
            return self

    def compute(self, thetas, X, one_sample=False, disable_tqdm=False):
        # simulating from each theta and dataset to compute waldo statistic
        # predicting posterior density for X and theta
        if not one_sample:
            return -self.base_model.predict(thetas, X)
        else:
            X_tile = np.tile(X, (thetas.shape[0], 1))
            return -self.base_model.predict(thetas, X_tile)

    # function to compute nuissance
    def compute_nuissance(
        self,
        thetas,
        X,
        nuissance_idx,
        disable_tqdm=False,
        MC_samples=500,
        posterior_marginalized=False,
    ):
        idx_array = np.arange(0, thetas.shape[1])
        parameter_idx = np.setdiff1d(idx_array, nuissance_idx)
        par_reorder = np.concatenate((parameter_idx, nuissance_idx), axis=None)
        thetas_par = thetas[:, parameter_idx]

        i = 0
        if disable_tqdm:
            if not posterior_marginalized:
                nuis_prob = np.zeros(thetas.shape[0])
                for theta in thetas_par:
                    theta_sample = self.base_model.sample(
                        X=X[i, :], num_samples=MC_samples
                    )

                    # computing mean across several nuissance parameters
                    theta_repeat = np.tile(theta, (MC_samples, 1))
                    new_theta_sample = np.column_stack(
                        (theta_repeat, theta_sample[:, nuissance_idx])
                    )
                    # rearranging new theta sample columns
                    new_theta_sample = new_theta_sample[:, par_reorder]

                    # computing probabilities
                    X_tile = np.tile(X[i, :], (new_theta_sample.shape[0], 1))

                    # computing the mean
                    nuis_prob[i] = -np.mean(
                        self.base_model.predict(new_theta_sample, X_tile)
                    )
                    i += 1
            else:
                nuis_prob = -self.base_model.predict(thetas_par, X)

        else:
            if not posterior_marginalized:
                nuis_prob = np.zeros(thetas.shape[0])
                for theta in tqdm(
                    thetas_par,
                    desc="Computing marginal posterior probability for theta",
                ):
                    theta_sample = self.base_model.sample(
                        X=X[i, :], num_samples=MC_samples
                    )

                    # computing mean across several nuissance parameters
                    theta_repeat = np.tile(theta, (MC_samples, 1))
                    new_theta_sample = np.column_stack(
                        (theta_repeat, theta_sample[:, nuissance_idx])
                    )
                    # rearranging new theta sample columns
                    new_theta_sample = new_theta_sample[:, par_reorder]

                    # computing probabilities
                    X_tile = np.tile(X[i, :], (new_theta_sample.shape[0], 1))

                    # computing the mean
                    nuis_prob[i] = -np.mean(
                        self.base_model.predict(new_theta_sample, X_tile)
                    )
                    i += 1
            else:
                nuis_prob = -self.base_model.predict(thetas_par, X)
        return nuis_prob

    def predict(self, thetas_grid, X, cutoffs, theta_1d=True):
        # predicting lambdas for all thetas
        lambdas = self.compute(thetas_grid, X, one_sample=True)
        idxs = list(np.where(lambdas <= cutoffs)[0])

        if theta_1d:
            return thetas_grid.squeeze()[idxs]
        else:
            return thetas_grid[idxs, :]


# E-value score
class E_valueScore(Scores):
    def fit(self, X=None, thetas=None):
        # setting up model for normalizing flows
        if self.is_fitted:
            return self
        elif self.base_model is not None:
            self.base_model.fit(X, thetas)
        else:
            return self

    def compute(self, thetas, X, N=10**3, one_sample=False, disable_tqdm=False):
        # simulating from each theta and dataset to compute waldo statistic
        i = 0
        lambdas = np.zeros(thetas.shape[0])
        theta_shape = thetas.shape[1]

        for theta in tqdm(
            thetas,
            desc="Computing e-value statistics using posterior model",
            disable=disable_tqdm,
        ):
            if not one_sample:
                # simulating from the posterior for each X
                s = self.base_model.sample(X=X[i, :], num_samples=N)

                # computing probability for each sample
                prob_s = self.base_model.predict(s, np.tile(X[i, :], (N, 1)))

                # compute probability for theta
                if theta_shape > 1:
                    prob_theta = self.base_model.predict(
                        theta.reshape(1, -1), X[i, :].reshape(1, -1)
                    )
                else:
                    prob_theta = self.base_model.predict(theta, X[i, :].reshape(1, -1))
            else:
                # simulating from the posterior for each X
                s = self.base_model.sample(X=X, num_samples=N)

                # computing probability for each sample
                prob_s = self.base_model.predict(s, np.tile(X, (N, 1)))

                # compute probability for theta
                if theta_shape > 1:
                    prob_theta = self.base_model.predict(
                        theta.reshape(1, -1), X.reshape(1, -1)
                    )
                else:
                    prob_theta = self.base_model.predict(theta, X.reshape(1, -1))

            # computing e-value
            lambdas[i] = np.mean(prob_s >= prob_theta)
            i += 1

        return lambdas

    def predict(self, thetas_grid, X, cutoffs, theta_1d=True):
        # predicting lambdas for all thetas
        lambdas = self.compute(thetas_grid, X, one_sample=True)
        idxs = list(np.where(lambdas <= cutoffs)[0])

        if theta_1d:
            return thetas_grid.squeeze()[idxs]
        else:
            return thetas_grid[idxs, :]
