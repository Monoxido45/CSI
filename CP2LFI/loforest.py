import numpy as np
from sklearn.base import BaseEstimator, clone
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

# using xgboost random forest regressor to build random forests with quantile loss
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
        Input: (i)    nc_score: Conformity score of choosing. It can be specified by instantiating a conformal score class based on the Scores basic class.
               (ii)   base_model: Base model with fit and predict methods to be embedded in the conformity score class.
               (iii)  alpha: Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
               (iv)   base_model_type: Boolean indicating whether the base model ouputs quantiles or not. Default is False.
               (v)    cart_type: Set "CART" to obtain LOCART prediction intervals and "RF" to obtain LOFOREST prediction intervals. Default is CART.
               (vi)   split_calib: Boolean designating if we should split the calibration set into partitioning and cutoff set. Default is True.
               (vii)  weighting: Set whether we should augment the feature space with conditional variance (difficulty) estimates. Default is False.
               (viii) **kwargs: Additional keyword arguments passed to fit base_model.
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
        Fit base model embeded in the conformal score class to the training set.
        If "weigthing" is True, we fit a Random Forest model to obtain variance estimations as done in Bostrom et.al.(2021).
        --------------------------------------------------------

        Input: (i)    X: Training numpy feature matrix
               (ii)   y: Training label array
               (iii)  random_seed_tree: Random Forest random seed for variance estimation (if weighting parameter is True).
               (iv)   **kwargs: Keyword arguments passed to fit the random forest used for variance estimation.

        Output: LocartSplit object
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
        n_estimators=200,
        min_samples_leaf=100,
        min_impurity_decrease=0,
        max_features=0.9,
        colsample_bynode=0.9,
        **kwargs
    ):
        """
        Calibrate conformity score using conformal Random Forest
        As default, we fix "min_samples_leaf" as 100 for both the CART and RF algorithms,meaning that each partition element will have at least
        100 samples each, and use the sklearn default for the remaining parameters. To generate other partitioning schemes, all RF and CART parameters
        can be changed through keyword arguments, but we recommend changing only the "min_samples_leaf" argument if needed.
        --------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix
               (ii)   y_calib: Calibration label array
               (iii)  random_seed: Random seed for CART or Random Forest fitted to the confomity scores.
               (iv)   train_size: Proportion of calibration data used in partitioning.
               (v)    **kwargs: Keyword arguments to be passed to Random Forest or XGBoost Random Forest.

        Ouput: Vector of cutoffs.
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

        # pinball loss still not working
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
            # obtaining leaves by iterating over estimator
            self.res_leaves = self.get_bagging_leaves(X_calib_test)
        return None

    def get_bagging_leaves(self, X_calib):
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

    def compute_breiman_matrix(self, X_test):
        test_size = X_test.shape[0]

        # Define a function to compute a single row of the Breiman matrix
        def compute_breiman_row(i):
            leaves_obs_sel = self.leaves_obs[i, :]
            matches = self.res_leaves == leaves_obs_sel
            return matches.sum(axis=1)

        # Vectorize the function
        vectorized_compute_breiman_row = np.vectorize(compute_breiman_row)

        # Use the vectorized function to compute the Breiman matrix
        breiman_matrix = vectorized_compute_breiman_row(np.arange(test_size))

        return breiman_matrix

    def check_k(self, breiman_matrix, K, min_samples):
        percentage_coverage = np.mean(np.sum(breiman_matrix > K, axis=1) > min_samples)
        if percentage_coverage >= 0.9:
            return [True, percentage_coverage]
        else:
            return [False, percentage_coverage]

    def tune_k(self, breiman_matrix, K_init=80, min_samples=300, step=5):
        K_trial = K_init
        k_list = self.check_k(breiman_matrix, K_init, min_samples)
        k_bool, percentage_coverage = k_list[0], k_list[1]

        if k_bool:
            while percentage_coverage > 0.9:
                new_K = K_trial + step
                k_list = self.check_k(breiman_matrix, new_K, min_samples)
                if not k_list[0]:
                    break
                else:
                    K_trial = new_K
                    percentage_coverage = k_list[1]
                percentage_coverage = np.mean(
                    np.sum(breiman_matrix > new_K, axis=1) > min_samples
                )
                K_trial = new_K
        else:
            while percentage_coverage < 0.9:
                new_K = K_trial - step
                k_list = self.check_k(breiman_matrix, new_K, min_samples)
                K_trial = new_K
                if k_list[0]:
                    break
                else:
                    percentage_coverage = k_list[1]

        return K_trial

    def compute_cutoffs(self, X, K_init=50):
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
        breiman_matrix = self.compute_breiman_matrix(X_tree)

        if self.tune:
            K = self.tune_k(breiman_matrix=breiman_matrix, K_init=K_init)
        else:
            K = self.K

        for i in range(0, test_size):
            obs_idx = np.where(breiman_matrix >= K)[0]

            # obtaining cutoff based on found residuals
            local_res = self.res_vector[obs_idx]
            n = local_res.shape[0]

            cutoffs[i] = np.quantile(
                local_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
            )
        return cutoffs

    def predict(self, X):
        cutoffs = self.compute_cutoffs(X)
        return self.nc_score.predict(X, cutoffs)
