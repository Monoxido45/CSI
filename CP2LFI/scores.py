from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import clone


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
