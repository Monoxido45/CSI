import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import itertools
from tqdm import tqdm

# package to fit glm
import statsmodels.api as sm


class GLM_stat:
    def __init__(
        self,
        prior_func,
        X_mat,
        rng=None,
        dist="gamma",
        link_func="log",
        random_state=45,
    ):
        # initializing variable and attributes
        # distribution from exponential family and link function
        self.dist = dist
        self.link_func = link_func
        # random seed
        if rng is None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = rng

        # prior and simulator functions
        self.prior = prior_func

        # dimension of betas
        self.dim = X_mat.shape[1]

        # fixed covariate matrix
        self.X_mat = sm.add_constant(X_mat)

    def loglikelihood(self, beta, phi, Y_data):
        if self.dist == "gamma":
            if self.link_func == "log":
                shape_par = 1 / phi
                if self.X_mat.ndim == 1:
                    X_mat_used = self.X_mat.reshape(-1, 1)
                else:
                    X_mat_used = self.X_mat
                # computing sum of log likelihood
                log_l = np.sum(
                    np.log(
                        stats.gamma.pdf(
                            Y_data,
                            a=shape_par,
                            scale=(1 / shape_par)
                            * np.exp(beta @ np.transpose(X_mat_used)),
                        )
                    )
                )

        return log_l

    def change_seed(self, new_rng):
        self.rng = np.random.default_rng(new_rng)
        return None

    def simulator(self, beta_vec, phi_vec):
        B, n = beta_vec.shape[0], self.X_mat.shape[0]
        # computing linear component
        linear_comp = beta_vec @ np.transpose(self.X_mat)
        if self.dist == "gamma":
            if self.link_func == "log":
                shape_par = 1 / phi_vec
                shape_par_dist = np.tile(shape_par, (n, 1)).transpose()
                Y_sim = self.rng.gamma(
                    shape=shape_par_dist,
                    scale=(1 / shape_par_dist) * np.exp(linear_comp),
                    size=(B, n),
                )

        return Y_sim

    # computing likelihood ratio for glm
    def LR_sim_lambda(
        self,
        beta_value,
        phi_value,
        B,
        idx_1,
        fit_intercept=True,
    ):
        lambda_results = []
    
        # Pre-calculating indices and offsets
        if not fit_intercept:
            X_glm = self.X_mat[:, 1:]
            intercept_val = beta_value[0]
            offset_array = np.repeat(intercept_val, X_glm.shape[0])
        else:
            X_glm = self.X_mat
            offset_array = None

        # Construct R_mat for the constrained/partial model
        R_mat = np.zeros((idx_1.shape[0], beta_value.shape[0]))
        R_mat[np.arange(0, idx_1.shape[0]), idx_1] = 1.0
        beta_1 = beta_value[idx_1]

        attempts = 0
        max_attempts = B * 5  # Safety break to prevent infinite loops if theta is truly degenerate
        
        while len(lambda_results) < B and attempts < max_attempts:
            attempts += 1
            try:
                # 1. Simulate one realization of Y
                Y_sim = self.simulator(beta_value.reshape(1, -1), np.array([phi_value]))[0]
                
                # 2. Fit complete and partial models
                glm_model = sm.GLM(
                    Y_sim, X_glm, 
                    family=sm.families.Gamma(link=sm.families.links.Log()),
                    offset=offset_array
                )
                
                complete_model = glm_model.fit()
                partial_model = glm_model.fit_constrained((R_mat, beta_1))
                
                # 3. Compute likelihoods
                params_complete = complete_model.params
                params_test = partial_model.params
                
                if not fit_intercept:
                    params_complete = np.insert(params_complete, 0, intercept_val)
                    params_test = np.insert(params_test, 0, intercept_val)
                    
                mle_l = self.loglikelihood(params_complete, phi_value, Y_sim)
                test_l = self.loglikelihood(params_test, phi_value, Y_sim)
                
                lrt_val = 2 * (mle_l - test_l)
                
                # 4. Filter for numerical sanity
                if not (np.isnan(lrt_val) or np.isinf(lrt_val) or lrt_val < 0):
                    lambda_results.append(lrt_val)
                    
            except (ValueError, RuntimeWarning, Exception):
                # Skip the iteration if statsmodels encounters NaN weights or LinAlg errors
                continue

        # Return result or handle total failure
        if len(lambda_results) == 0:
            return np.array([np.nan] * B)

        # If we couldn't reach B but have some results, we return what we have
        return np.array(lambda_results)
    
    def LR_sample(self, B, idx_1, fit_intercept=True, intercept_value=None):
        lambda_array = np.zeros(B)
        # We need to keep track of the final beta/phi values used
        final_betas = []
        final_phis = []

        for i in tqdm(range(B), desc="Simulating parameters and lambdas"):
            success = False
            while not success:
                try:
                    # 1. Simulate a single parameter set from the prior
                    # Note: We simulate 1 at a time to handle specific failures
                    beta_vals, phi_vals = self.prior(
                        1, 
                        rng=self.rng, 
                        intercept_value=intercept_value, 
                        dim=self.dim - 1 if not fit_intercept else self.dim
                    )
                    curr_beta = beta_vals[0]
                    curr_phi = phi_vals[0]

                    # 2. Simulate Y data for this parameter set
                    Y_sim = self.simulator(curr_beta.reshape(1, -1), curr_phi.reshape(1))[0]
                    
                    if idx_1.ndim == 1:
                        beta_1 = curr_beta[idx_1]
                    else:
                        beta_1 = curr_beta[idx_1]

                    # 3. Setup the GLM models
                    if not fit_intercept:
                        X_glm = self.X_mat[:, 1:]
                        offset_array = np.repeat(intercept_value, X_glm.shape[0])
                    else:
                        X_glm = self.X_mat
                        offset_array = None

                    glm_model = sm.GLM(
                        Y_sim,
                        X_glm,
                        family=sm.families.Gamma(link=sm.families.links.Log()), # Use Log class
                        offset=offset_array,
                    )

                    # 4. Attempt to fit. This is where the ValueError usually triggers.
                    complete_model = glm_model.fit()

                    # Construct R_mat and q_vec for constrained fit
                    R_mat = np.zeros((idx_1.shape[0], curr_beta.shape[0]))
                    R_mat[np.arange(0, idx_1.shape[0]), idx_1] = 1.0
                    q_vec = beta_1

                    partial_model = glm_model.fit_constrained((R_mat, q_vec))

                    # 5. Compute Log-Likelihoods
                    params_complete = complete_model.params
                    params_test = partial_model.params

                    if not fit_intercept:
                        params_complete = np.insert(params_complete, 0, intercept_value)
                        params_test = np.insert(params_test, 0, intercept_value)

                    mle_l = self.loglikelihood(params_complete, curr_phi, Y_sim)
                    test_l = self.loglikelihood(params_test, curr_phi, Y_sim)

                    lambda_array[i] = 2 * (mle_l - test_l)
                    
                    # If we reached here without error, store parameters and break while loop
                    final_betas.append(curr_beta)
                    final_phis.append(curr_phi)
                    success = True

                except (ValueError, RuntimeWarning, Exception) as e:
                    # If statsmodels fails, the loop continues and resimulates
                    continue

        par_values = np.concatenate(
            (np.array(final_betas), np.array(final_phis).reshape(-1, 1)), 
            axis=1
        )

        return par_values, lambda_array


class Simulations:
    def __init__(
        self,
        rng=None,
        kind_model="1d_normal",
        random_state=45,
    ):
        # Initialize any necessary variables or attributes here
        self.kind_model = kind_model
        if rng is None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = rng

    def LRT_sim_lambda(self, theta, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                lambda_array[i] = self.compute_lrt_statistic(theta, X)

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # function to compute gmm likelihood
            for i in range(0, B):
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = self.compute_lrt_statistic(theta, X)

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            # function to compute LRT for 1d normal
            for i in range(0, B):
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)
                lambda_array[i] = self.compute_lrt_statistic(theta, X)
        return lambda_array

    def LRT_sample(self, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)
        i = 0
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # sampling theta
            thetas = self.rng.uniform(-5, 5, size=B)

            for theta in thetas:
                X = self.rng.normal(theta, 1, N)
                lambda_array[i] = self.compute_lrt_statistic(theta, X)
                i += 1

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # sampling theta
            thetas = self.rng.uniform(0, 5, size=B)

            for theta in thetas:
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = self.compute_lrt_statistic(theta, X)
                i += 1

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            thetas = np.c_[
                self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.15, 1.25, B)
            ]

            for theta in thetas:
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)
                lambda_array[i] = self.compute_lrt_statistic(theta_0=theta, X=X)
                i += 1
        return thetas, lambda_array

    def KS_sim_lambda(self, theta, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":

            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                lambda_array[i] = self.compute_ks_statistic(theta, X)
                i += 1

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # function to compute KS statistic for gmm
            for i in range(0, B):
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = self.compute_ks_statistic(theta, X)
                i += 1

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            # function to compute KS statistic for lognormal
            for i in range(0, B):
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)
                lambda_array[i] = self.compute_ks_statistic(theta, X)
                i += 1

        return lambda_array

    def KS_sample(self, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # sampling theta
            thetas = self.rng.uniform(-5, 5, size=B)

            i = 0
            for theta in thetas:
                X = self.rng.normal(theta, 1, N)
                lambda_array[i] = self.compute_ks_statistic(theta, X)
                i += 1

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # sampling theta
            thetas = self.rng.uniform(0, 5, size=B)

            i = 0
            for theta in thetas:
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = self.compute_ks_statistic(theta, X)
                i += 1

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            thetas = np.c_[self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.15, 1, B)]

            i = 0
            for theta in thetas:
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)
                lambda_array[i] = self.compute_ks_statistic(theta, X)
                i += 1

        return thetas, lambda_array

    def BFF_sim_lambda(self, theta, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # function to compute posterior parameters for 1d normal
            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = self.compute_posterior_par(X)
                lambda_array[i] = -stats.norm.pdf(
                    theta, loc=mu_pos, scale=np.sqrt(sigma_pos)
                )

        elif self.kind_model == "gmm":
            # function to compute posterior parameters for gmm
            for i in range(0, B):
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = -self.posterior_pdf(theta, X, BF=True)

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            for i in range(0, B):
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)
                lambda_array[i] = -self.posterior_pdf(theta, X)

        return lambda_array

    def BFF_sample(self, B, N):

        # testing kind of model
        lambda_array = np.zeros(B)
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            thetas = self.rng.uniform(-5, 5, size=B)

            # function to compute posterior parameters for 1d normal
            i = 0
            for theta in thetas:
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = self.compute_posterior_par(X)
                lambda_array[i] = -stats.norm.pdf(
                    theta, loc=mu_pos, scale=np.sqrt(sigma_pos)
                )
                i += 1

        elif self.kind_model == "gmm":
            thetas = self.rng.uniform(0, 5, size=B)

            # function to compute posterior parameters for gmm
            i = 0
            for theta in thetas:
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = -self.posterior_pdf(theta, X, BF=True)
                i += 1

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            thetas = np.c_[
                self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.15, 1.25, B)
            ]

            i = 0
            for theta in thetas:
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)
                lambda_array[i] = -self.posterior_pdf(theta, X)
                i += 1

        return thetas, lambda_array

    def FBST_sim_lambda(self, theta, B, N, MC_samples=10**3):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # obtaining evidence using monte carlo integration from posterior samples
            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = self.compute_posterior_par(X)

                # posterior samples
                theta_sample = self.rng.normal(
                    loc=mu_pos, scale=np.sqrt(sigma_pos), size=MC_samples
                )

                # posterior densities of theta_sample
                theta_dens = stats.norm.pdf(
                    theta_sample, loc=mu_pos, scale=np.sqrt(sigma_pos)
                )

                # density value at H0
                f_h0 = stats.norm.pdf(theta, loc=mu_pos, scale=np.sqrt(sigma_pos))

                lambda_array[i] = np.mean(theta_dens >= f_h0)

        elif self.kind_model == "gmm":
            # function to compute posterior parameters for gmm
            for i in range(0, B):
                # simulating sample
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )

                # simulating from posterior
                theta_sample = self.posterior_sim(MC_samples, X)

                # posterior densities of theta_sample
                theta_dens = self.posterior_pdf(theta_sample, X, BF=False)

                # density value at H0
                f_h0 = self.posterior_pdf(theta, X, BF=False)

                lambda_array[i] = np.mean(theta_dens >= f_h0)

        elif self.kind_model == "lognormal":
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            for i in range(0, B):
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)

                mu_pos, nu_pos, alpha_pos, beta_pos = self.compute_posterior_par(X)

                # posterior samples
                sigmas = 1 / self.rng.gamma(
                    shape=alpha_pos, scale=1 / beta_pos, size=MC_samples
                )
                mus = self.rng.normal(
                    loc=mu_pos, scale=np.sqrt(sigmas / nu_pos), size=MC_samples
                )

                # posterior densities
                theta_dens = np.log(
                    stats.norm.pdf(mus, loc=mu_pos, scale=np.sqrt(sigmas / nu_pos))
                ) + np.log(stats.invgamma.pdf(sigmas, a=alpha_pos, scale=beta_pos))

                # density value at H0
                f_h0 = np.log(
                    stats.norm.pdf(
                        theta[0], loc=mu_pos, scale=np.sqrt(theta[1] / nu_pos)
                    )
                ) + np.log(stats.invgamma.pdf(theta[1], a=alpha_pos, scale=beta_pos))

                lambda_array[i] = np.mean(theta_dens >= f_h0)

        return lambda_array

    def FBST_sample(self, B, N, MC_samples=10**4):

        # testing kind of model
        lambda_array = np.zeros(B)
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # sampling theta
            thetas = self.rng.uniform(-5, 5, size=B)

            # obtaining evidence using monte carlo integration from posterior samples
            i = 0
            for theta in thetas:
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = self.compute_posterior_par(X)

                # posterior samples
                theta_sample = self.rng.normal(
                    loc=mu_pos, scale=np.sqrt(sigma_pos), size=MC_samples
                )

                # posterior densities of theta_sample
                theta_dens = stats.norm.pdf(
                    theta_sample, loc=mu_pos, scale=np.sqrt(sigma_pos)
                )

                # density value at H0
                f_h0 = stats.norm.pdf(theta, loc=mu_pos, scale=np.sqrt(sigma_pos))

                lambda_array[i] = np.mean(theta_dens >= f_h0)
                i += 1

        elif self.kind_model == "gmm":
            thetas = self.rng.uniform(0, 5, size=B)

            i = 0
            for theta in thetas:
                # simulating sample
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )

                # simulating from posterior
                theta_sample = self.posterior_sim(MC_samples, X)

                # posterior densities of theta_sample
                theta_dens = self.posterior_pdf(theta_sample, X, BF=False)

                # density value at H0
                f_h0 = self.posterior_pdf(theta, X, BF=False)

                lambda_array[i] = np.mean(theta_dens >= f_h0)
                i += 1

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            thetas = np.c_[self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.15, 1, B)]

            i = 0
            for theta in thetas:
                X = self.rng.lognormal(mean=theta[0], sigma=np.sqrt(theta[1]), size=N)

                mu_pos, nu_pos, alpha_pos, beta_pos = self.compute_posterior_par(X)

                # posterior samples
                sigmas = 1 / self.rng.gamma(
                    shape=alpha_pos, scale=1 / beta_pos, size=MC_samples
                )
                mus = self.rng.normal(
                    loc=mu_pos, scale=np.sqrt(sigmas / nu_pos), size=MC_samples
                )

                # posterior densities
                theta_dens = np.log(
                    stats.norm.pdf(mus, loc=mu_pos, scale=np.sqrt(sigmas / nu_pos))
                ) + np.log(stats.invgamma.pdf(sigmas, a=alpha_pos, scale=beta_pos))

                # density value at H0
                f_h0 = np.log(
                    stats.norm.pdf(
                        theta[0], loc=mu_pos, scale=np.sqrt(theta[1] / nu_pos)
                    )
                ) + np.log(stats.invgamma.pdf(theta[1], a=alpha_pos, scale=beta_pos))

                lambda_array[i] = np.mean(theta_dens >= f_h0)
                i += 1

        return thetas, lambda_array

    # auxiliary functions to compute posterior and so on
    def compute_lrt_statistic(self, theta_0, X, lower=0, upper=5):
        if self.kind_model == "1d_normal":
            mle_theta = np.mean(X)
            lrt_stat = -2 * (
                np.sum(np.log(stats.norm.pdf(X, loc=theta_0, scale=1)))
                - np.sum(np.log(stats.norm.pdf(X, loc=mle_theta, scale=1)))
            )
        elif self.kind_model == "gmm":

            def l_func(theta, x):
                # prob from X
                p_x = np.log(
                    (0.5 * stats.norm.pdf(x, loc=theta, scale=1))
                    + (0.5 * stats.norm.pdf(x, loc=-theta, scale=1))
                )
                return -(np.sum(p_x))

            # computing MLE by grid
            res = minimize_scalar(
                l_func,
                args=(X),
                bounds=(lower, upper),
                tol=0.01,
                options={"maxiter": 100},
            )
            mle_theta = res.x
            lrt_stat = -2 * ((-l_func(theta_0, X)) - (-l_func(mle_theta, X)))
        elif self.kind_model == "lognormal":
            mle_mu, mle_sigma = np.mean(np.log(X)), np.var(np.log(X))
            lrt_stat = -2 * (
                np.sum(
                    np.log(
                        stats.lognorm.pdf(
                            X, scale=np.exp(theta_0[0]), s=np.sqrt(theta_0[1])
                        )
                    )
                )
                - np.sum(
                    np.log(
                        stats.lognorm.pdf(X, scale=np.exp(mle_mu), s=np.sqrt(mle_sigma))
                    )
                )
            )
            return lrt_stat

        return lrt_stat

    def compute_ks_statistic(self, theta_0, X):
        if self.kind_model == "1d_normal":
            empirical = stats.ecdf(X)
            theoretical = np.sort(stats.norm.cdf(X, loc=theta_0, scale=1))
            ks_stat = np.sqrt(X.shape[0]) * np.max(
                np.abs(theoretical - empirical.cdf.probabilities)
            )

        elif self.kind_model == "gmm":
            empirical = stats.ecdf(X)
            theoretical = np.sort(
                0.5 * stats.norm.cdf(X, loc=theta_0, scale=1)
                + 0.5 * stats.norm.cdf(X, loc=-theta_0, scale=1)
            )
            ks_stat = np.sqrt(X.shape[0]) * np.max(
                np.abs(theoretical - empirical.cdf.probabilities)
            )

        elif self.kind_model == "lognormal":
            empirical = stats.ecdf(X)
            theoretical = np.sort(
                stats.lognorm.cdf(X, s=np.sqrt(theta_0[1]), scale=np.exp(theta_0[0]))
            )
            ks_stat = np.sqrt(X.shape[0]) * np.max(
                np.abs(theoretical - empirical.cdf.probabilities)
            )

        return ks_stat

    def compute_posterior_par(self, X):
        if self.kind_model == "1d_normal":
            sigma = 0.25
            n = X.shape[0]
            mu_value = (1 / ((1 / sigma) + n)) * (np.sum(X))
            sigma_value = ((1 / sigma) + n) ** (-1)
            return mu_value, sigma_value

        elif self.kind_model == "lognormal":
            nu, mu_0 = 2, 0
            alpha, beta = 2, 1
            n = X.shape[0]
            mu_pos = ((nu * mu_0) + (n * np.mean(np.log(X)))) / (nu + n)
            nu_pos = nu + n
            alpha_pos = alpha + (n / 2)
            beta_pos = (
                beta
                + (1 / 2 * n * np.var(np.log(X)))
                + (((n * nu) / (nu + n)) * ((np.mean(np.log(X)) - mu_0) ** 2 / 2))
            )
            return mu_pos, nu_pos, alpha_pos, beta_pos

    def posterior_pdf(self, theta, X, MC_samples=10**3, BF=False):
        if self.kind_model == "gmm":
            # computing p_x only for BF
            if BF:
                l_X_theta = np.log(
                    (0.5 * stats.norm.pdf(X, loc=theta))
                    + (0.5 * stats.norm.pdf(X, loc=-theta))
                )

                thetas_sim = self.rng.normal(loc=1, scale=1, size=MC_samples)
                p_X = (
                    0.5
                    * stats.norm.pdf(
                        np.tile(X, (MC_samples, 1)),
                        loc=np.repeat(thetas_sim, X.shape[0]).reshape(
                            MC_samples, X.shape[0]
                        ),
                    )
                ) + (
                    0.5
                    * stats.norm.pdf(
                        np.tile(X, (MC_samples, 1)),
                        loc=np.repeat(-thetas_sim, X.shape[0]).reshape(
                            MC_samples, X.shape[0]
                        ),
                    )
                )

                p_X = np.mean(np.prod(p_X, axis=1))

                p_theta = np.log(stats.norm.pdf(theta, loc=1, scale=1)) + np.sum(
                    l_X_theta
                )
                p_theta_x = p_theta - np.log(p_X)

                return p_theta_x
            else:
                if isinstance(theta, float):
                    l_X_theta = np.log(
                        (0.5 * stats.norm.pdf(X, loc=theta))
                        + (0.5 * stats.norm.pdf(X, loc=-theta))
                    )

                    p_theta = np.log(stats.norm.pdf(theta, loc=1, scale=1)) + np.sum(
                        l_X_theta
                    )
                else:
                    l_X_theta = np.log(
                        (
                            0.5
                            * stats.norm.pdf(
                                np.tile(X, (theta.shape[0], 1)),
                                loc=np.repeat(theta, X.shape[0]).reshape(
                                    theta.shape[0], X.shape[0]
                                ),
                            )
                        )
                        + (
                            0.5
                            * stats.norm.pdf(
                                np.tile(X, (theta.shape[0], 1)),
                                loc=np.repeat(-theta, X.shape[0]).reshape(
                                    theta.shape[0], X.shape[0]
                                ),
                            )
                        )
                    )

                    p_theta = np.log(stats.norm.pdf(theta, loc=1, scale=1)) + np.sum(
                        l_X_theta, axis=1
                    )

                return p_theta

        elif self.kind_model == "lognormal":
            mu_pos, nu_pos, alpha_pos, beta_pos = self.compute_posterior_par(X)
            p_theta = np.log(
                stats.norm.pdf(theta[0], loc=mu_pos, scale=np.sqrt(theta[1] / nu_pos))
            ) + np.log(stats.invgamma.pdf(theta[1], a=alpha_pos, scale=beta_pos))
            return p_theta

    def posterior_sim(self, B, X):
        # for each simulation: simulate from Z, then simulate for theta
        p_x = np.exp(-1 / 4 * (X - 1) ** 2) / (
            np.exp(-1 / 4 * (X - 1) ** 2) + np.exp(-1 / 4 * (X + 1) ** 2)
        )

        z = self.rng.binomial(n=1, p=np.tile(p_x, (B, 1)), size=(B, X.shape[0]))
        X_0, X_1 = np.sum((z == 0) * np.tile(X, (B, 1)), axis=1), np.sum(
            (z == 1) * np.tile(X, (B, 1)), axis=1
        )

        mu_value, sigma_value = (X_1 - X_0 + 1) / (X.shape[0] + 1), 1 / (X.shape[0] + 1)

        thetas = self.rng.normal(mu_value, np.repeat(np.sqrt(sigma_value), B), size=B)
        return thetas


# implementing also the naive approach to fit each case:
def naive(stat, kind_model, alpha, rng, B=1000, N=100, naive_n=500):
    n_grid = int(B / naive_n)
    sim_obj = Simulations(rng=rng, kind_model=kind_model)
    sim_lambda = getattr(sim_obj, stat + "_sim_lambda")
    quantiles = {}

    if kind_model == "1d_normal":
        thetas = np.linspace(-4.9999, 4.9999, n_grid)
        for theta in thetas:
            lambdas = sim_lambda(B=naive_n, N=N, theta=theta)
            quantiles[theta] = np.quantile(lambdas, q=1 - alpha)

    elif kind_model == "gmm":
        thetas = np.linspace(0.0001, 4.9999, n_grid)
        for theta in thetas:
            lambdas = sim_lambda(B=naive_n, N=N, theta=theta)
            quantiles[theta] = np.quantile(lambdas, q=1 - alpha)

    elif kind_model == "lognormal":
        n_grid = round(B / naive_n)
        a_s = np.linspace(-2.4999, 2.4999, int(np.ceil(n_grid ** (1 / 2))))
        b_s = np.linspace(0.1501, 1.2499, int(np.ceil(n_grid ** (1 / 2))))
        for mu, sigma in itertools.product(a_s, b_s):
            theta = np.array([mu, sigma])
            lambdas = sim_lambda(B=naive_n, N=N, theta=theta)
            quantiles[(mu, sigma)] = np.quantile(lambdas, q=1 - alpha)
    return quantiles


def predict_naive_quantile(kind_model, theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for theta in theta_grid:
        if kind_model != "lognormal":
            idx = thetas_values[int(np.argmin(np.abs(theta - thetas_values)))]
            quantiles_list.append(quantiles_dict[idx])
        else:
            distances = np.linalg.norm(thetas_values - theta, axis=1)
            idx = thetas_values[np.argmin(distances)]
            quantiles_list.append(quantiles_dict[tuple(idx)])
    return quantiles_list
