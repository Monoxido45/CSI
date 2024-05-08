import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import itertools


class Simulations:
    def __init__(
        self,
        rng=None,
        kind_model="1d_normal",
        random_state=45,
    ):
        # Initialize any necessary variables or attributes here
        self.model = kind_model
        if rng is None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = rng

    def LRT_sim_lambda(self, theta, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # function to compute LRT for 1d normal
            def compute_lrt_statistic(theta_0, X):
                mle_theta = np.mean(X)
                lrt_stat = -2 * (
                    np.log(stats.norm.pdf(X, loc=theta_0, scale=1))
                    - np.log(stats.norm.pdf(X, loc=mle_theta, scale=1))
                )
                return lrt_stat

            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                lambda_array[i] = compute_lrt_statistic(theta, X)

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # function to compute gmm likelihood
            def l_func(theta, x):
                # prob from X
                p_x = np.log(
                    (0.5 * stats.norm.pdf(x, loc=theta, scale=1))
                    + (0.5 * stats.norm.pdf(x, loc=-theta, scale=1))
                )
                return -(np.sum(p_x))

            def compute_lrt_statistic(theta_0, X, lower=0, upper=5):
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
                return lrt_stat

            for i in range(0, B):
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = compute_lrt_statistic(theta, X)

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            # function to compute LRT for 1d normal
            def compute_lrt_statistic(theta_0, X):
                mle_mu, mle_sigma = np.mean(X), np.sqrt(np.var(X))
                lrt_stat = -2 * (
                    np.log(stats.lognorm.pdf(X, loc=theta_0[0], s=theta_0[1]))
                    - np.log(stats.norm.pdf(X, loc=mle_mu, s=mle_sigma))
                )
                return lrt_stat

            for i in range(0, B):
                X = self.rng.lognormal(mu=theta[0], sigma=theta[1], size=N)
                lambda_array[i] = compute_lrt_statistic(theta, X)
        return lambda_array

    def LRT_sample(self, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # sampling theta
            thetas = self.rng.uniform(-5, 5, size=B)

            # function to compute LRT for 1d normal
            def compute_lrt_statistic(theta_0, X):
                mle_theta = np.mean(X)
                lrt_stat = -2 * (
                    np.log(stats.norm.pdf(X, loc=theta_0, scale=1))
                    - np.log(stats.norm.pdf(X, loc=mle_theta, scale=1))
                )
                return lrt_stat

            i = 0
            for theta in thetas:
                X = self.rng.normal(theta, 1, N)
                lambda_array[i] = compute_lrt_statistic(theta, X)
                i += 1

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # sampling theta
            thetas = self.rng.uniform(0, 5, size=B)

            # function to compute gmm likelihood
            def l_func(theta, x):
                # prob from X
                p_x = np.log(
                    (0.5 * stats.norm.pdf(x, loc=theta, scale=1))
                    + (0.5 * stats.norm.pdf(x, loc=-theta, scale=1))
                )
                return -(np.sum(p_x))

            def compute_lrt_statistic(theta_0, X, lower=0, upper=5):
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
                return lrt_stat

            for theta in thetas:
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = compute_lrt_statistic(theta, X)
                i += 1

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            thetas = np.c_[
                self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.25, 6.25, B)
            ]

            # function to compute LRT for 1d normal
            def compute_lrt_statistic(theta_0, X):
                mle_mu, mle_sigma = np.mean(X), np.sqrt(np.var(X))
                lrt_stat = -2 * (
                    np.log(stats.lognorm.pdf(X, loc=theta_0[0], s=theta_0[1]))
                    - np.log(stats.norm.pdf(X, loc=mle_mu, s=mle_sigma))
                )
                return lrt_stat

            for theta in thetas:
                X = self.rng.lognormal(mu=theta[0], sigma=theta[1], size=N)
                lambda_array[i] = compute_lrt_statistic(theta, X)
                i += 1
        return thetas, lambda_array

    def KS_sim_lambda(self, theta, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # function to compute KS statistic for 1d normal
            def compute_ks_statistic(theta_0, X):
                empirical = stats.ecdf(X)
                theoretical = np.sort(stats.norm.cdf(X, loc=theta_0, scale=1))
                ks_stat = np.sqrt(X.shape[0]) * np.max(
                    np.abs(theoretical - empirical.cdf.probabilities)
                )
                return ks_stat

            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                lambda_array[i] = compute_ks_statistic(theta, X)
                i += 1

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # function to compute KS statistic for gmm
            def compute_ks_statistic(theta_0, X):
                empirical = stats.ecdf(X)
                theoretical = np.sort(
                    0.5 * stats.norm.cdf(X, loc=theta_0, scale=1)
                    + 0.5 * stats.norm.cdf(X, loc=-theta_0, scale=1)
                )
                ks_stat = np.sqrt(X.shape[0]) * np.max(
                    np.abs(theoretical - empirical.cdf.probabilities)
                )
                return ks_stat

            for i in range(0, B):
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = compute_ks_statistic(theta, X)
                i += 1

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            # function to compute KS statistic for lognormal
            def compute_ks_statistic(theta_0, X):
                empirical = stats.ecdf(X)
                theoretical = np.sort(
                    stats.lognorm.cdf(X, s=theta_0[1], loc=theta_0[0])
                )
                ks_stat = np.sqrt(X.shape[0]) * np.max(
                    np.abs(theoretical - empirical.cdf.probabilities)
                )
                return ks_stat

            for i in range(0, B):
                X = self.rng.lognormal(mu=theta[0], sigma=theta[1], size=N)
                lambda_array[i] = compute_ks_statistic(theta, X)
                i += 1

        return lambda_array

    def KS_sample(self, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # sampling theta
            thetas = self.rng.uniform(-5, 5, size=B)

            # function to compute KS statistic for 1d normal
            def compute_ks_statistic(theta_0, X):
                empirical = stats.ecdf(X)
                theoretical = np.sort(stats.norm.cdf(X, loc=theta_0, scale=1))
                ks_stat = np.sqrt(X.shape[0]) * np.max(
                    np.abs(theoretical - empirical.cdf.probabilities)
                )
                return ks_stat

            i = 0
            for theta in thetas:
                X = self.rng.normal(theta, 1, N)
                lambda_array[i] = compute_ks_statistic(theta, X)
                i += 1

        # second model: gaussian mixture model. In this case, the MLE is computed through optim
        elif self.kind_model == "gmm":
            # sampling theta
            thetas = self.rng.uniform(0, 5, size=B)

            # function to compute KS statistic for gmm
            def compute_ks_statistic(theta_0, X):
                empirical = stats.ecdf(X)
                theoretical = np.sort(
                    0.5 * stats.norm.cdf(X, loc=theta_0, scale=1)
                    + 0.5 * stats.norm.cdf(X, loc=-theta_0, scale=1)
                )
                ks_stat = np.sqrt(X.shape[0]) * np.max(
                    np.abs(theoretical - empirical.cdf.probabilities)
                )
                return ks_stat

            for theta in thetas:
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = compute_ks_statistic(theta, X)
                i += 1

        # third model: lognormal distribution.
        elif self.kind_model == "lognormal":
            thetas = np.c_[
                self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.25, 6.25, B)
            ]

            # function to compute KS statistic for lognormal
            def compute_ks_statistic(theta_0, X):
                empirical = stats.ecdf(X)
                theoretical = np.sort(
                    stats.lognorm.cdf(X, s=theta_0[1], loc=theta_0[0])
                )
                ks_stat = np.sqrt(X.shape[0]) * np.max(
                    np.abs(theoretical - empirical.cdf.probabilities)
                )
                return ks_stat

            for theta in thetas:
                X = self.rng.lognormal(mu=theta[0], sigma=theta[1], size=N)
                lambda_array[i] = compute_ks_statistic(theta, X)
                i += 1

        return lambda_array

    def BFF_sim_lambda(self, theta, B, N):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # function to compute posterior parameters for 1d normal
            def compute_posterior_par(X):
                sigma = 0.25
                n = X.shape[0]
                mu_value = (1 / ((1 / sigma) + n)) * (np.sum(X))
                sigma_value = ((1 / sigma) + n) ** (-1)
                return mu_value, sigma_value

            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = compute_posterior_par(theta, X)
                lambda_array[i] = -stats.norm.pdf(
                    theta, loc=mu_pos, scale=np.sqrt(sigma_pos)
                )

        elif self.kind_model == "gmm":
            # function to compute posterior parameters for gmm
            def compute_posterior_par(X):
                n = X.shape[0]
                mu_value = (n / (n + 4)) * (np.mean(X))
                sigma_value = 1 / (n + 4)
                return mu_value, sigma_value

            def posterior_pdf(theta, x):
                mu_value, sigma_value = compute_posterior_par(x)
                # prob from X
                p_theta = (
                    0.5
                    * stats.norm.pdf(theta, loc=mu_value, scale=np.sqrt(sigma_value))
                ) + (
                    0.5
                    * stats.norm.pdf(theta, loc=-mu_value, scale=np.sqrt(sigma_value))
                )
                return p_theta

            for i in range(0, B):
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = -posterior_pdf(theta, X)

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            def compute_posterior_par(X):
                nu, mu_0 = 2, 0
                alpha, beta = 2, 2
                n = X.shape[0]
                mu_pos = ((nu * mu_0) + (n * np.mean(X))) / (nu + n)
                nu_pos = nu + n
                alpha_pos = alpha + (n / 2)
                beta_pos = (
                    beta
                    + (1 / 2 * n * np.var(X))
                    + (((n * nu) / (nu + n)) * ((np.mean(X) - mu_0) ** 2 / 2))
                )
                return mu_pos, nu_pos, alpha_pos, beta_pos

            def posterior_pdf(theta, X):
                mu_pos, nu_pos, alpha_pos, beta_pos = compute_posterior_par(X)
                return stats.norm.pdf(
                    theta[0], loc=mu_pos, scale=theta[1] / np.sqrt(nu_pos)
                ) * stats.invgamma.pdf(theta[1] ** 2, a=alpha_pos, scale=beta_pos)

            for i in range(0, B):
                X = self.rng.lognormal(mean=theta[0], sigma=theta[1], size=N)
                lambda_array[i] = -posterior_pdf(theta, X)

        return lambda_array

    def BFF_sample(self, B, N):

        # testing kind of model
        lambda_array = np.zeros(B)
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            thetas = self.rng.uniform(-5, 5, size=B)

            # function to compute posterior parameters for 1d normal
            def compute_posterior_par(X):
                sigma = 0.25
                n = X.shape[0]
                mu_value = (1 / ((1 / sigma) + n)) * (np.sum(X))
                sigma_value = ((1 / sigma) + n) ** (-1)
                return mu_value, sigma_value

            i = 0
            for theta in thetas:
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = compute_posterior_par(X)
                lambda_array[i] = -stats.norm.pdf(
                    theta, loc=mu_pos, scale=np.sqrt(sigma_pos)
                )
                i += 1

        elif self.kind_model == "gmm":
            thetas = self.rng.uniform(0, 5, size=B)

            # function to compute posterior parameters for gmm
            def compute_posterior_par(X):
                n = X.shape[0]
                mu_value = (n / (n + 4)) * (np.mean(X))
                sigma_value = 1 / (n + 4)
                return mu_value, sigma_value

            def posterior_pdf(theta, x):
                mu_value, sigma_value = compute_posterior_par(x)
                # prob from X
                p_theta = (
                    0.5
                    * stats.norm.pdf(theta, loc=mu_value, scale=np.sqrt(sigma_value))
                ) + (
                    0.5
                    * stats.norm.pdf(theta, loc=-mu_value, scale=np.sqrt(sigma_value))
                )
                return p_theta

            i = 0
            for theta in thetas:
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )
                lambda_array[i] = -posterior_pdf(theta, X)
                i += 1

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            thetas = np.c_[
                self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.25, 6.25, B)
            ]

            def compute_posterior_par(X):
                nu, mu_0 = 2, 0
                alpha, beta = 2, 2
                n = X.shape[0]
                mu_pos = ((nu * mu_0) + (n * np.mean(X))) / (nu + n)
                nu_pos = nu + n
                alpha_pos = alpha + (n / 2)
                beta_pos = (
                    beta
                    + (1 / 2 * n * np.var(X))
                    + (((n * nu) / (nu + n)) * ((np.mean(X) - mu_0) ** 2 / 2))
                )
                return mu_pos, nu_pos, alpha_pos, beta_pos

            def posterior_pdf(theta, X):
                mu_pos, nu_pos, alpha_pos, beta_pos = compute_posterior_par(X)
                return stats.norm.pdf(
                    theta[0], loc=mu_pos, scale=theta[1] / np.sqrt(nu_pos)
                ) * stats.invgamma.pdf(theta[1] ** 2, a=alpha_pos, scale=beta_pos)

            i = 0
            for theta in thetas:
                X = self.rng.lognormal(mean=theta[0], sigma=theta[1], size=N)
                lambda_array[i] = -posterior_pdf(theta, X)
                i += 1

        return lambda_array

    def FBST_sim_lambda(self, theta, B, N, MC_samples=10**4):
        # testing kind of model
        lambda_array = np.zeros(B)

        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # function to compute posterior parameters for 1d normal
            def compute_posterior_par(X):
                sigma = 0.25
                n = X.shape[0]
                mu_value = (1 / ((1 / sigma) + n)) * (np.sum(X))
                sigma_value = ((1 / sigma) + n) ** (-1)
                return mu_value, sigma_value

            # obtaining evidence using monte carlo integration from posterior samples
            for i in range(0, B):
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = compute_posterior_par(theta, X)

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

                lambda_array[i] = np.mean(theta_dens > f_h0)

        elif self.kind_model == "gmm":
            # function to compute posterior parameters for gmm
            def compute_posterior_par(X):
                n = X.shape[0]
                mu_value = (n / (n + 4)) * (np.mean(X))
                sigma_value = 1 / (n + 4)
                return mu_value, sigma_value

            def posterior_pdf(theta, mu_value, sigma_value):
                # prob from X
                p_theta = (
                    0.5
                    * stats.norm.pdf(theta, loc=mu_value, scale=np.sqrt(sigma_value))
                ) + (
                    0.5
                    * stats.norm.pdf(theta, loc=-mu_value, scale=np.sqrt(sigma_value))
                )
                return p_theta

            def posterior_sim(B, mu_value, sigma_value):
                group = self.rng.binomial(n=1, p=0.5, size=B)
                thetas = (
                    (group == 0)
                    * (self.rng.normal(mu_value, np.sqrt(sigma_value), size=B))
                ) + (
                    (group == 1)
                    * (self.rng.normal(-mu_value, np.sqrt(sigma_value), size=B))
                )
                return thetas

            for i in range(0, B):
                # simulating sample
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )

                # computing posterior parameters
                mu_value, sigma_value = compute_posterior_par(X)

                # simulating from posterior
                theta_sample = posterior_sim(MC_samples, mu_value, sigma_value)

                # posterior densities of theta_sample
                theta_dens = posterior_pdf(theta_sample, mu_value, sigma_value)

                # density value at H0
                f_h0 = posterior_pdf(theta, mu_value, sigma_value)

                lambda_array[i] = np.mean(theta_dens > f_h0)

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            def compute_posterior_par(X):
                nu, mu_0 = 2, 0
                alpha, beta = 2, 2
                n = X.shape[0]
                mu_pos = ((nu * mu_0) + (n * np.mean(X))) / (nu + n)
                nu_pos = nu + n
                alpha_pos = alpha + (n / 2)
                beta_pos = (
                    beta
                    + (1 / 2 * n * np.var(X))
                    + (((n * nu) / (nu + n)) * ((np.mean(X) - mu_0) ** 2 / 2))
                )
                return mu_pos, nu_pos, alpha_pos, beta_pos

            def posterior_pdf(theta, mu_pos, nu_pos, alpha_pos, beta_pos):
                return stats.norm.pdf(
                    theta[0], loc=mu_pos, scale=theta[1] / np.sqrt(nu_pos)
                ) * stats.invgamma.pdf(theta[1] ** 2, a=alpha_pos, scale=beta_pos)

            for i in range(0, B):
                X = self.rng.lognormal(mean=theta[0], sigma=theta[1], size=N)

                mu_pos, nu_pos, alpha_pos, beta_pos = compute_posterior_par(X)

                # posterior samples
                sigmas = 1 / self.rng.gamma(
                    shape=alpha_pos, scale=1 / beta_pos, size=MC_samples
                )
                mus = self.rng.normal(
                    loc=mu_pos, scale=np.sqrt(sigmas / nu_pos), size=MC_samples
                )

                # posterior densities
                theta_dens = stats.norm.pdf(
                    mus, loc=mu_pos, scale=np.sqrt(sigmas / nu_pos)
                ) * stats.invgamma.pdf(sigmas, a=alpha_pos, scale=beta_pos)

                # density value at H0
                f_h0 = stats.norm.pdf(
                    theta[0], loc=mu_pos, scale=theta[1] / np.sqrt(nu_pos)
                ) * stats.invgamma.pdf(theta[1] ** 2, a=alpha_pos, scale=beta_pos)

                lambda_array[i] = np.mean(theta_dens > f_h0)

        return lambda_array

    def FBST_sample(self, B, N, MC_samples=10**4):
        # testing kind of model
        lambda_array = np.zeros(B)
        # first model: normal 1d. In this case, the MLE is given by the sample mean
        if self.kind_model == "1d_normal":
            # sampling theta
            thetas = self.rng.uniform(-5, 5, size=B)

            # function to compute posterior parameters for 1d normal
            def compute_posterior_par(X):
                sigma = 0.25
                n = X.shape[0]
                mu_value = (1 / ((1 / sigma) + n)) * (np.sum(X))
                sigma_value = ((1 / sigma) + n) ** (-1)
                return mu_value, sigma_value

            # obtaining evidence using monte carlo integration from posterior samples
            i = 0
            for theta in thetas:
                X = self.rng.normal(loc=theta, scale=1, size=N)
                mu_pos, sigma_pos = compute_posterior_par(X)

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

                lambda_array[i] = np.mean(theta_dens > f_h0)
                i += 1

        elif self.kind_model == "gmm":
            thetas = self.rng.uniform(0, 5, size=B)

            # function to compute posterior parameters for gmm
            def compute_posterior_par(X):
                n = X.shape[0]
                mu_value = (n / (n + 4)) * (np.mean(X))
                sigma_value = 1 / (n + 4)
                return mu_value, sigma_value

            def posterior_pdf(theta, mu_value, sigma_value):
                # prob from X
                p_theta = (
                    0.5
                    * stats.norm.pdf(theta, loc=mu_value, scale=np.sqrt(sigma_value))
                ) + (
                    0.5
                    * stats.norm.pdf(theta, loc=-mu_value, scale=np.sqrt(sigma_value))
                )
                return p_theta

            def posterior_sim(B, mu_value, sigma_value):
                group = self.rng.binomial(n=1, p=0.5, size=B)
                thetas = (
                    (group == 0)
                    * (self.rng.normal(mu_value, np.sqrt(sigma_value), size=B))
                ) + (
                    (group == 1)
                    * (self.rng.normal(-mu_value, np.sqrt(sigma_value), size=B))
                )
                return thetas

            i = 0
            for theta in thetas:
                # simulating sample
                group = self.rng.binomial(n=1, p=0.5, size=N)
                X = ((group == 0) * (self.rng.normal(theta, 1, size=N))) + (
                    (group == 1) * (self.rng.normal(-theta, 1, size=N))
                )

                # computing posterior parameters
                mu_value, sigma_value = compute_posterior_par(X)

                # simulating from posterior
                theta_sample = posterior_sim(MC_samples, mu_value, sigma_value)

                # posterior densities of theta_sample
                theta_dens = posterior_pdf(theta_sample, mu_value, sigma_value)

                # density value at H0
                f_h0 = posterior_pdf(theta, mu_value, sigma_value)

                lambda_array[i] = np.mean(theta_dens > f_h0)
                i += 1

        elif self.kind_model == "lognormal":
            # function to compute posterior parameters for lognormal
            # we consider: alpha = 2, beta = 2, mu_0 = 0, nu = 0.5
            thetas = np.c_[
                self.rng.uniform(-2.5, 2.5, B), self.rng.uniform(0.25, 6.25, B)
            ]

            def compute_posterior_par(X):
                nu, mu_0 = 2, 0
                alpha, beta = 2, 2
                n = X.shape[0]
                mu_pos = ((nu * mu_0) + (n * np.mean(X))) / (nu + n)
                nu_pos = nu + n
                alpha_pos = alpha + (n / 2)
                beta_pos = (
                    beta
                    + (1 / 2 * n * np.var(X))
                    + (((n * nu) / (nu + n)) * ((np.mean(X) - mu_0) ** 2 / 2))
                )
                return mu_pos, nu_pos, alpha_pos, beta_pos

            i = 0
            for theta in thetas:
                X = self.rng.lognormal(mean=theta[0], sigma=theta[1], size=N)

                mu_pos, nu_pos, alpha_pos, beta_pos = compute_posterior_par(X)

                # posterior samples
                sigmas = 1 / self.rng.gamma(
                    shape=alpha_pos, scale=1 / beta_pos, size=MC_samples
                )
                mus = self.rng.normal(
                    loc=mu_pos, scale=np.sqrt(sigmas / nu_pos), size=MC_samples
                )

                # posterior densities
                theta_dens = stats.norm.pdf(
                    mus, loc=mu_pos, scale=np.sqrt(sigmas / nu_pos)
                ) * stats.invgamma.pdf(sigmas, a=alpha_pos, scale=beta_pos)

                # density value at H0
                f_h0 = stats.norm.pdf(
                    theta[0], loc=mu_pos, scale=theta[1] / np.sqrt(nu_pos)
                ) * stats.invgamma.pdf(theta[1] ** 2, a=alpha_pos, scale=beta_pos)

                lambda_array[i] = np.mean(theta_dens > f_h0)
                i += 1

        return lambda_array


# implementing also the naive approach to fit each case:
def naive(stat, kind_model, alpha, rng, B=1000, N=100, seed=250, naive_n=500):
    n_grid = int(B / naive_n)
    sim_obj = Simulations(rng=rng, kind_model=kind_model)
    sim_lambda = getattr(sim_obj, stat + "_sim_lambda")
    quantiles = {}

    if kind_model == "1d_normal":
        thetas = np.linspace(-5, 5, n_grid)
        for theta in thetas:
            lambdas = sim_lambda(B=naive_n, N=N, theta=theta)
            quantiles[theta] = np.quantile(lambdas, q=1 - alpha)

    elif kind_model == "gmm":
        thetas = np.linspace(0, 5, n_grid)
        for theta in thetas:
            lambdas = sim_lambda(B=naive_n, N=N, theta=theta)
            quantiles[theta] = np.quantile(lambdas, q=1 - alpha)

    elif kind_model == "lognormal":
        n_grid = round(np.sqrt(B / naive_n))
        a_s = np.linspace(-2.4999, 2.4999, n_grid)
        b_s = np.linspace(0.25001, 2.4999, n_grid)
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
