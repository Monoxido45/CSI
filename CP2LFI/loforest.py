import numpy as np
from sklearn.base import BaseEstimator, clone
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import scipy.stats as st


class ConformalLoforest(BaseEstimator):
    """
    Local Regression Tree and Local Forests class.
    Fit the conformal version of LOFOREST local calibration methods for any conformity score and base model of interest. The specification of the conformity
    score can be made through the usage of the basic class "Scores". Through the "split_calib" parameter we can decide whether to use all calibration set to
    obtain both the partition and cutoffs or split it into two sets, one specific for partitioning and other for obtaining the local cutoffs. Also, if
    desired, we can fit the augmented version of both our methods (A-LOCART and A-LOFOREST) by the "weighting" parameter, which if True, adds difficulty
    estimates to our feature matrix in the calibration and prediction step.
    ----------------------------------------------------------------
    """

    def __init__(
        self,
        nc_score,
        base_model,
        alpha,
        is_fitted=False,
        base_model_type=None,
        split_calib=True,
        weighting=False,
        tune_K=False,
        **kwargs
    ):
        """
        Initialize the ConformalLoforest object.

        Parameters:
        - nc_score: Conformity score of choosing. It can be specified by instantiating a conformal score class based on the Scores basic class.
        - base_model: Base model with fit and predict methods to be embedded in the conformity score class.
        - alpha: Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
        - is_fitted: Boolean indicating whether the base model is already fitted. Default is False.
        - base_model_type: Boolean indicating whether the base model outputs quantiles or not. Default is None.
        - split_calib: Boolean designating if we should split the calibration set into partitioning and cutoff set. Default is True.
        - weighting: Set whether we should augment the feature space with conditional variance (difficulty) estimates. Default is False.
        - tune_K: Boolean indicating whether to tune the parameter K. Default is False.
        - **kwargs: Additional keyword arguments passed to fit base_model.
        """
        self.base_model_type = base_model_type
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(
                base_model, is_fitted=is_fitted, alpha=alpha, **kwargs
            )
        else:
            self.nc_score = nc_score(base_model, is_fitted=is_fitted, **kwargs)

        # checking if base model is fitted
        self.base_model = self.nc_score.base_model
        self.alpha = alpha
        self.split_calib = split_calib
        self.weighting = weighting
        self.tune = tune_K

    def fit(self, X, y, random_seed_tree=1250, **kwargs):
        """
        Fit the base model embedded in the conformal score class to the training set.

        Parameters:
        - X: Training numpy feature matrix.
        - y: Training label array.
        - random_seed_tree: Random Forest random seed for variance estimation (if weighting parameter is True).
        - **kwargs: Keyword arguments passed to fit the random forest used for variance estimation.

        Returns:
        - self: The fitted ConformalLoforest object.
        """
        self.nc_score.fit(X, y)
        if self.weighting == True:
            if not isinstance(self.nc_score.base_model, RandomForestRegressor):
                self.dif_model = (
                    RandomForestRegressor(random_state=random_seed_tree)
                    .set_params(**kwargs)
                    .fit(X, y)
                )
            else:
                self.dif_model = deepcopy(self.nc_score.base_model)
        return self

    def calibrate(
        self,
        X_calib,
        y_calib,
        random_seed=1250,
        train_size=0.5,
        objective="mse_based",
        K=None,
        fit_nuisance_tree=False,
        n_estimators=200,
        min_samples_leaf=100,
        min_impurity_decrease=0,
        max_features=0.9,
        colsample_bynode=0.9,
        **kwargs
    ):
        """
        Calibrate the conformity score using conformal Random Forest.

        Parameters:
        - X_calib: Calibration numpy feature matrix.
        - y_calib: Calibration label array.
        - random_seed: Random seed for CART or Random Forest fitted to the conformity scores.
        - train_size: Proportion of calibration data used in partitioning.
        - objective: Objective function for calibration. Options are "mse_based" and "quantile". Default is "mse_based".
        - K: Number of trees to use in the local forest. If None, K is set to half the number of estimators in the random forest. Default is None.
        - n_estimators: Number of trees in the random forest. Default is 200.
        - min_samples_leaf: Minimum number of samples required to be at a leaf node. Default is 100.
        - min_impurity_decrease: Minimum impurity decrease required for a split. Default is 0.
        - max_features: Maximum number of features to consider when looking for the best split. Default is 0.9.
        - colsample_bynode: Subsample ratio of columns for each split. Default is 0.9.
        - **kwargs: Keyword arguments to be passed to Random Forest or XGBoost Random Forest.

        Returns:
        - None
        """
        res = self.nc_score.compute(X_calib, y_calib)

        if self.weighting:
            w = self.compute_difficulty(X_calib)
            X_calib = np.concatenate((X_calib, w.reshape(-1, 1)), axis=1)

        if self.split_calib:
            (
                X_calib_train,
                X_calib_test,
                res_calib_train,
                res_calib_test,
            ) = train_test_split(
                X_calib, res, test_size=1 - train_size, random_state=random_seed
            )
        else:
            X_calib_train, X_calib_test, res_calib_train, res_calib_test = (
                X_calib,
                X_calib,
                res,
                res,
            )

        if objective == "mse_based":
            self.RF = RandomForestRegressor(
                random_state=random_seed,
                min_samples_leaf=min_samples_leaf,
                n_estimators=n_estimators,
                min_impurity_decrease=min_impurity_decrease,
                max_features=max_features,
            ).set_params(**kwargs)
        elif objective == "quantile":
            base_model = XGBRegressor(
                n_estimators=1,
                objective="reg:quantileerror",
                grow_policy="depthwise",
                verbosity=0,
                max_leaves=0,
                quantile_alpha=1 - self.alpha,
                colsample_bynode=colsample_bynode,
                min_child_weight=min_samples_leaf,
                booster="gbtree",
                tree_method="approx",
                gamma=min_impurity_decrease,
            ).set_params(**kwargs)

            self.RF = BaggingRegressor(
                base_model,
                n_estimators=n_estimators,
                max_features=max_features,
                random_state=random_seed,
                n_jobs=-1,
            )

        self.RF.fit(X_calib_train, res_calib_train)

        if fit_nuisance_tree:
            self.nuisance_tree = DecisionTreeRegressor(
                random_state=random_seed,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease,
            )
            self.nuisance_tree.fit(X_calib_train, res_calib_train)

        if K is None:
            self.K = len(self.RF.estimators_) / 2
        else:
            self.K = K

        self.res_vector = res_calib_test
        if objective == "mse_based" or objective == None:
            self.objective = "mse_based"
            self.res_leaves = self.RF.apply(X_calib_test)
        elif objective == "quantile":
            self.objective = "quantile"
            self.res_leaves = self.get_bagging_leaves(X_calib_test)
        return None

    def get_bagging_leaves(self, X_calib):
        """
        Get the leaves of the trees in the bagging ensemble.

        Parameters:
        - X_calib: Calibration numpy feature matrix.

        Returns:
        - leaves_matrix: Matrix of leaves for each sample in X_calib.
        """
        estimators_list = self.RF.estimators_
        leaves_matrix = np.zeros((X_calib.shape[0], len(estimators_list)))
        j = 0
        for estimator in estimators_list:
            leaves_matrix[:, j] = estimator.apply(X_calib)
            j += 1
        return leaves_matrix

    def bagging_apply(self, X):
        estimators_list = self.RF.estimators_
        leaves_matrix = np.zeros((X.shape[0], len(estimators_list)))
        j = 0
        for estimator in estimators_list:
            leaves_matrix[:, j] = estimator.apply(X)
            j += 1
        return leaves_matrix

    def compute_difficulty(self, X):
        """
        Auxiliary function to compute difficulty for each sample.
        --------------------------------------------------------
        input: (i)    X: specified numpy feature matrix

        output: Vector of variance estimates for each sample.
        """
        cart_pred = np.zeros((X.shape[0], len(self.dif_model.estimators_)))
        i = 0
        # computing the difficulty score for each X_score
        for cart in self.dif_model.estimators_:
            cart_pred[:, i] = cart.predict(X)
            i += 1
        # computing variance for each dataset row
        return cart_pred.var(1)

    def compute_breiman_matrix(self, X_test, leaves_obs_comp=False):
        test_size = X_test.shape[0]
        if leaves_obs_comp:
            leaves_obs = self.leaves_obs
        else:
            leaves_obs = self.RF.apply(X_test)
        calib_size = self.res_leaves.shape[0]

        # Define a function to compute a single row of the Breiman matrix
        def compute_breiman_row(i):
            leaves_obs_sel = self.leaves_obs[i, :]
            matches = self.res_leaves == leaves_obs_sel
            return matches.sum(axis=1)

        breiman_matrix = np.zeros((test_size, calib_size))
        for i in range(test_size):
            leaves_obs_sel = leaves_obs[i, :]
            matches = self.res_leaves == leaves_obs_sel
            breiman_matrix[i, :] = matches.sum(axis=1)

        # Vectorize the function
        # vectorized_compute_breiman_row = np.vectorize(compute_breiman_row)

        # Use the vectorized function to compute the Breiman matrix
        # breiman_matrix = vectorized_compute_breiman_row(np.arange(test_size))

        return breiman_matrix

    def compute_cutoffs(self, X, K=None, breiman_mat=None):
        # if weighting is enabled
        if self.weighting:
            w = self.compute_difficulty(X)
            X_tree = np.concatenate((X, w.reshape(-1, 1)), axis=1)
        else:
            X_tree = X
        # obtaining cutoffs for each X on the fly

        # obtaining all leaves for all X's
        if self.objective == "mse_based":
            self.leaves_obs = self.RF.apply(X_tree)
        else:
            self.leaves_obs = self.bagging_apply(X_tree)

        test_size = X_tree.shape[0]

        # cutoffs array
        cutoffs = np.zeros(test_size)

        # computing breiman matrix
        if breiman_mat is None:
            breiman_matrix = self.compute_breiman_matrix(X_tree)
        else:
            breiman_matrix = breiman_mat

        # choosing K again
        if K is None:
            K = self.K

        for i in range(0, test_size):
            obs_idx = np.where(breiman_matrix[i, :] >= K)[0]

            # obtaining cutoff based on found residuals
            local_res = self.res_vector[obs_idx]
            # checking if local_res is empty
            # if it is, use global cutoff
            if local_res.shape[0] == 0:
                n = self.res_vector.shape[0]
                cutoffs[i] = np.quantile(
                    self.res_vector, q=np.ceil((n + 1) * (1 - self.alpha)) / n
                )
            # else, use local cutoff
            else:
                n = local_res.shape[0]
                # quantile correction only when n is large
                # if correction is larger than 1, than use 1-alpha
                correction = np.ceil((n + 1) * (1 - self.alpha)) / n

                if correction > 1:
                    cutoffs[i] = np.quantile(local_res, q=1 - self.alpha)
                else:
                    cutoffs[i] = np.quantile(local_res, q=correction)
        return cutoffs

    def predict(self, X):
        cutoffs = self.compute_cutoffs(X)
        return self.nc_score.predict(X, cutoffs)


def tune_loforest_LFI(
    loforest_model, theta_data, lambda_data, K_grid=np.arange(30, 85, 5)
):
    """
    Tune K parameter of the loforest model using the a test data set and generated lambda

    Parameters:
    - loforest_model: Already fitted loforest model to be tuned.
    - theta_data: Theta dataset for validation.
    - lambda_data: Lambda dataset for validation, each row gives.
    - K_grid: Grid of K to tune K.

    Returns:
    Tuned K
    """
    # computing breiman matrix of validation data
    breiman_mat = loforest_model.compute_breiman_matrix(theta_data)

    # cutoffs dict
    cutoffs = {}

    # compute cutoffs for K_grid
    for K in K_grid:
        cutoffs[K] = loforest_model.compute_cutoffs(
            theta_data, breiman_mat=breiman_mat, K=K
        )

    # computing loss through lambda_data
    err_data = np.zeros((lambda_data.shape[0], K_grid.shape[0]))
    i = 0
    for lambdas in lambda_data:
        j = 0
        for K in K_grid:
            coverage = np.mean(lambdas <= cutoffs[K][i])
            err_data[i, j] = np.abs(coverage - (1 - loforest_model.alpha))
            j += 1
        i += 1

    mae_array = np.mean(err_data, axis=0)
    # selecting K that minimizes validation error
    return K_grid[np.argmin(mae_array)]
