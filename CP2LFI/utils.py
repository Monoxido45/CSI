# functions used to perform LFI experiments
import numpy as np
from tqdm import tqdm
import itertools
import torch
from torch.distributions.beta import Beta
from CP2LFI.loforest import ConformalLoforest, tune_loforest_LFI
from CP2LFI.scores import LambdaScore
from CP2LFI.posterior_models import normflow_posterior
from clover import LocartSplit

import pickle
import io
import pandas as pd

# quantile regression
from sklearn.ensemble import HistGradientBoostingRegressor


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def beta_prior(type, sample_size):
    # simulating from beta(1/2, 1/2) for high-dimensional problems
    new_prior = Beta(0.5, 0.5)
    if type == "mg1":
        prior_tensor = new_prior.sample((sample_size, 3))
        # re-scaling to (0,10)
        prior_tensor[:, :2] = prior_tensor[:, :2] * (10)
        # re-scaling to (0,1/3)
        prior_tensor[:, 2] = prior_tensor[:, 2] * (1 / 3)
    elif type == "tractable":
        prior_tensor = (new_prior.sample((sample_size, 5)) * 6) - 3
    elif type == "sir":
        prior_tensor = new_prior.sample((sample_size, 2)) * 0.5
    elif type == "weinberg":
        prior_tensor = (new_prior.sample((sample_size,))) + 0.5
    elif type == "two moons":
        prior_tensor = (new_prior.sample((sample_size, 2)) * 2) - 1
    # TODO: add gravitational waves priors
    return prior_tensor


# defining naive function
def naive(
    kind,
    simulator,
    score,
    alpha,
    B=1000,
    N=100,
    naive_n=500,
    disable_tqdm=True,
    disable_tqdm_naive=True,
    log_transf=False,
):
    n_grid = int(B / naive_n)
    quantiles = {}

    if not disable_tqdm_naive:
        if kind == "weinberg":
            thetas = np.linspace(0.5001, 1.4999, n_grid)
            for theta in tqdm(thetas, desc="fitting monte carlo cutoffs"):
                theta_fixed = torch.tensor([theta])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[theta] = np.quantile(lambdas, q=1 - alpha)

        elif kind == "tractable":
            # given the complexity, reducing to only 10 grid if B > 5000
            par_grid = np.linspace(-2.9999, 2.9999, int(np.ceil(n_grid ** (1 / 5))))
            par_grid[par_grid == 0] = 0.01

            for theta_1, theta_2, theta_3, theta_4, theta_5 in tqdm(
                itertools.product(par_grid, par_grid, par_grid, par_grid, par_grid),
                desc="fitting monte carlo cutoffs",
            ):
                theta_fixed = torch.tensor(
                    [theta_1, theta_2, theta_3, theta_4, theta_5]
                )
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(theta_1, theta_2, theta_3, theta_4, theta_5)] = np.quantile(
                    lambdas, q=1 - alpha
                )

        elif kind == "sir":
            par_grid = np.linspace(0, 0.5, int(np.ceil(np.sqrt(n_grid))))

            for theta_1, theta_2 in tqdm(
                itertools.product(par_grid, par_grid),
                desc="fitting monte carlo cutoffs",
            ):
                theta_fixed = torch.tensor([theta_1, theta_2])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(theta_1, theta_2)] = np.quantile(lambdas, q=1 - alpha)

        elif kind == "two moons":
            pars_1 = np.linspace(-0.9999, 0.9999, int(np.ceil(n_grid ** (1 / 2))))

            for par1, par2 in tqdm(
                itertools.product(pars_1, pars_1),
                desc="fitting monte carlo cutoffs",
            ):
                theta_fixed = torch.tensor([par1, par2])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(par1, par2)] = np.quantile(lambdas, q=1 - alpha)

        elif kind == "mg1":
            pars_1 = np.linspace(0.0001, 9.9999, int(np.ceil(n_grid ** (1 / 3))))
            pars_2 = np.linspace(
                0.0001, 1 / 3 - 0.0001, int(np.ceil((n_grid) ** (1 / 3)))
            )

            for par1, par2, par3 in tqdm(
                itertools.product(pars_1, pars_1, pars_2),
                desc="fitting monte carlo cutoffs",
            ):
                theta_fixed = torch.tensor([par1, par2, par3])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(par1, par2, par3)] = np.quantile(lambdas, q=1 - alpha)
    else:
        if kind == "weinberg":
            thetas = np.linspace(0.5001, 1.4999, n_grid)
            for theta in thetas:
                theta_fixed = torch.tensor([theta])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[theta] = np.quantile(lambdas, q=1 - alpha)

        elif kind == "tractable":
            # given the complexity, reducing to only 10 grid if B > 5000
            par_grid = np.linspace(-2.9999, 2.9999, int(np.ceil(n_grid ** (1 / 5))))
            par_grid[par_grid == 0] = 0.01

            for theta_1, theta_2, theta_3, theta_4, theta_5 in itertools.product(
                par_grid, par_grid, par_grid, par_grid, par_grid
            ):
                theta_fixed = torch.tensor(
                    [theta_1, theta_2, theta_3, theta_4, theta_5]
                )
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(theta_1, theta_2, theta_3, theta_4, theta_5)] = np.quantile(
                    lambdas, q=1 - alpha
                )

        elif kind == "sir":
            par_grid = np.linspace(0, 0.5, int(np.ceil(np.sqrt(n_grid))))

            for theta_1, theta_2 in itertools.product(par_grid, par_grid):
                theta_fixed = torch.tensor([theta_1, theta_2])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(theta_1, theta_2)] = np.quantile(lambdas, q=1 - alpha)

        elif kind == "two moons":
            pars_1 = np.linspace(-0.9999, 0.9999, int(np.ceil(n_grid ** (1 / 2))))

            for par1, par2 in itertools.product(pars_1, pars_1):
                theta_fixed = torch.tensor([par1, par2])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(par1, par2)] = np.quantile(lambdas, q=1 - alpha)

        elif kind == "mg1":
            pars_1 = np.linspace(0.0001, 9.9999, int(np.ceil(n_grid ** (1 / 3))))
            pars_2 = np.linspace(
                0.0001, 1 / 3 - 0.0001, int(np.ceil((n_grid) ** (1 / 3)))
            )

            for par1, par2, par3 in itertools.product(pars_1, pars_1, pars_2):
                theta_fixed = torch.tensor([par1, par2, par3])
                repeated_thetas = theta_fixed.reshape(1, -1).repeat_interleave(
                    repeats=naive_n * N, dim=0
                )
                X_samples = simulator(repeated_thetas)

                if log_transf:
                    X_samples = torch.log(X_samples)

                X_dim = X_samples.shape[1]
                X_samples = X_samples.reshape(naive_n, N * X_dim)

                lambdas = score.compute(
                    thetas=repeated_thetas.numpy()[0:naive_n, :],
                    X=X_samples.numpy(),
                    disable_tqdm=disable_tqdm,
                )

                quantiles[(par1, par2, par3)] = np.quantile(lambdas, q=1 - alpha)
    return quantiles


# prediction function for naive
def predict_naive_quantile(kind, theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for theta in theta_grid:
        if kind == "weinberg":
            idx = thetas_values[int(np.argmin(np.abs(theta - thetas_values)))]
            quantiles_list.append(quantiles_dict[idx])
        else:
            distances = np.linalg.norm(thetas_values - theta, axis=1)
            idx = thetas_values[np.argmin(distances)]
            quantiles_list.append(quantiles_dict[tuple(idx)])
    return quantiles_list


def obtain_quantiles_saved_tune(
    kind,
    score,
    score_name,
    theta_grid_eval,
    simulator,
    prior,
    N,
    original_path,
    tune_path,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=100,
    K=50,
    K_grid=np.concatenate((np.array([0]), np.arange(15, 105, 5))),
    naive_n=500,
    disable_tqdm=True,
    log_transf=False,
    split_calib=False,
    using_beta=False,
    two_moons=False,
):
    # fitting and predicting naive (monte-carlo
    print("Running naive method")
    naive_quantiles = naive(
        kind=kind,
        simulator=simulator,
        score=score,
        alpha=alpha,
        B=B,
        N=N,
        naive_n=naive_n,
        log_transf=log_transf,
        disable_tqdm=disable_tqdm,
        disable_tqdm_naive=True,
    )
    naive_list = predict_naive_quantile(kind, theta_grid_eval, naive_quantiles)

    # simulating to fit models
    if not using_beta:
        if two_moons:
            thetas_sim = prior(num_samples=B)
        else:
            thetas_sim = prior.sample((B,))
    else:
        thetas_sim = beta_prior(type=kind, sample_size=B)

    if thetas_sim.ndim == 1:
        model_thetas = thetas_sim.reshape(-1, 1)
    else:
        model_thetas = thetas_sim

    repeated_thetas = thetas_sim.repeat_interleave(repeats=N, dim=0)
    X_net = simulator(repeated_thetas)
    if log_transf:
        X_net = torch.log(X_net)
    X_dim = X_net.shape[1]
    X_net = X_net.reshape(B, N * X_dim)

    model_lambdas = score.compute(
        model_thetas.numpy(), X_net.numpy(), disable_tqdm=disable_tqdm
    )

    # checking if there are any NaNs in training and printing and message
    # if True, remove elements with nan
    nan_lambda = np.isnan(model_lambdas)
    sum_nan = np.sum(nan_lambda)
    if sum_nan > 0:
        print(f"Warning: simulated data has {sum_nan} nan values")
        model_lambdas = model_lambdas[~nan_lambda]
        model_thetas = model_thetas[~nan_lambda, :]

    locart_object = LocartSplit(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=split_calib
    )
    locart_quantiles = locart_object.calib(
        model_thetas.numpy(), model_lambdas, min_samples_leaf=min_samples_leaf
    )

    # loforest quantiles
    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=split_calib
    )
    loforest_object.calibrate(
        model_thetas.numpy(),
        model_lambdas,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        K=K,
    )

    # boosting quantiles
    model = HistGradientBoostingRegressor(
        loss="quantile",
        max_iter=100,
        max_depth=3,
        quantile=1 - alpha,
        random_state=105,
        n_iter_no_change=15,
        early_stopping=True,
    )
    model.fit(model_thetas.numpy(), model_lambdas)

    # importing tuning samples
    # first, importing theta_tune object
    theta_tune = pd.read_pickle(
        original_path + tune_path + f"{kind}_theta_tune_{N}.pickle"
    )

    # then, importing lambda_tune object
    lambda_tune = pd.read_pickle(
        original_path + tune_path + f"{kind}_{score_name}_tune_{N}.pickle"
    )

    if theta_tune.ndim == 1:
        K_valid_thetas = theta_tune.reshape(-1, 1)
    else:
        K_valid_thetas = theta_tune

    # fitting tuned loforest
    K_loforest = tune_loforest_LFI(
        loforest_object,
        theta_data=K_valid_thetas.numpy(),
        lambda_data=lambda_tune,
        K_grid=K_grid,
    )

    # locart quantiles
    if theta_grid_eval.ndim == 1:
        model_eval = theta_grid_eval.reshape(-1, 1)
    else:
        model_eval = theta_grid_eval

    idxs = locart_object.cart.apply(model_eval)
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # loforest
    loforest_cutoffs = loforest_object.compute_cutoffs(model_eval)

    # tuned loforest
    loforest_tuned_cutoffs = loforest_object.compute_cutoffs(model_eval, K=K_loforest)

    # boosting
    boosting_quantiles = model.predict(model_eval)

    # dictionary of quantiles
    quantile_dict = {
        "naive": naive_list,
        "locart": list_locart_quantiles,
        "loforest_fixed": loforest_cutoffs,
        "loforest_tuned": loforest_tuned_cutoffs,
        "boosting": boosting_quantiles,
    }

    return quantile_dict, K_loforest


def obtain_quantiles(
    kind,
    score,
    theta_grid_eval,
    simulator,
    prior,
    N,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=100,
    K=50,
    B_valid=1000,
    N_lambda=150,
    K_grid=np.concatenate((np.array([0]), np.arange(15, 90, 5))),
    naive_n=500,
    disable_tqdm=True,
    log_transf=False,
    split_calib=False,
    using_beta=False,
    two_moons=False,
):
    # fitting and predicting naive (monte-carlo
    print("Running naive method")
    naive_quantiles = naive(
        kind=kind,
        simulator=simulator,
        score=score,
        alpha=alpha,
        B=B,
        N=N,
        naive_n=naive_n,
        log_transf=log_transf,
    )
    naive_list = predict_naive_quantile(kind, theta_grid_eval, naive_quantiles)

    # simulating to fit models
    if not using_beta:
        if two_moons:
            thetas_sim = prior(num_samples=B)
        else:
            thetas_sim = prior.sample((B,))
    else:
        thetas_sim = beta_prior(type=kind, sample_size=B)

    if thetas_sim.ndim == 1:
        model_thetas = thetas_sim.reshape(-1, 1)
    else:
        model_thetas = thetas_sim

    repeated_thetas = thetas_sim.repeat_interleave(repeats=N, dim=0)
    X_net = simulator(repeated_thetas)
    if log_transf:
        X_net = torch.log(X_net)
    X_dim = X_net.shape[1]
    X_net = X_net.reshape(B, N * X_dim)

    model_lambdas = score.compute(
        model_thetas.numpy(), X_net.numpy(), disable_tqdm=disable_tqdm
    )
    print(model_lambdas)

    print("Running all the other methods")
    locart_object = LocartSplit(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=split_calib
    )
    locart_quantiles = locart_object.calib(
        model_thetas.numpy(), model_lambdas, min_samples_leaf=min_samples_leaf
    )

    # loforest quantiles
    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=split_calib
    )
    loforest_object.calibrate(
        model_thetas.numpy(),
        model_lambdas,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        K=K,
    )

    # boosting quantiles
    model = HistGradientBoostingRegressor(
        loss="quantile",
        max_iter=100,
        max_depth=3,
        quantile=1 - alpha,
        random_state=105,
        n_iter_no_change=15,
        early_stopping=True,
    )
    model.fit(model_thetas.numpy(), model_lambdas)

    print("Obtaining tuning samples")
    # tuned loforest samples
    if not using_beta:
        if two_moons:
            theta_tune = prior(num_samples=B_valid)
        else:
            theta_tune = prior.sample((B_valid,))
    else:
        theta_tune = beta_prior(type=kind, sample_size=B_valid)

    # finding tuned K
    lambda_tune = np.zeros((theta_tune.shape[0], N_lambda))
    if theta_tune.ndim == 1:
        K_valid_thetas = theta_tune.reshape(-1, 1)
    else:
        K_valid_thetas = theta_tune

    i = 0
    for theta in tqdm(theta_tune, desc="Simulating all tuning samples"):
        if theta_tune.ndim == 1:
            theta_repeated = (
                torch.tensor([theta])
                .reshape(1, -1)
                .repeat_interleave(repeats=N_lambda * N, dim=0)
            )
        else:
            theta_repeated = theta.reshape(1, -1).repeat_interleave(
                repeats=N_lambda * N, dim=0
            )

        X_net = simulator(theta_repeated)
        if log_transf:
            X_net = torch.log(X_net)
        X_dim = X_net.shape[1]
        X_net = X_net.reshape(N_lambda, N * X_dim)
        lambda_tune[i, :] = score.compute(
            thetas=theta_repeated.numpy()[0:N_lambda, :],
            X=X_net.numpy(),
            disable_tqdm=disable_tqdm,
        )
        i += 1

    print("Fitting tuned loforest")
    K_loforest = tune_loforest_LFI(
        loforest_object,
        theta_data=K_valid_thetas.numpy(),
        lambda_data=lambda_tune,
        K_grid=K_grid,
    )

    # locart quantiles
    if theta_grid_eval.ndim == 1:
        model_eval = theta_grid_eval.reshape(-1, 1)
    else:
        model_eval = theta_grid_eval

    idxs = locart_object.cart.apply(model_eval)
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # loforest
    loforest_cutoffs = loforest_object.compute_cutoffs(model_eval)

    # tuned loforest
    loforest_tuned_cutoffs = loforest_object.compute_cutoffs(model_eval, K=K_loforest)

    # boosting
    boosting_quantiles = model.predict(model_eval)

    # dictionary of quantiles
    quantile_dict = {
        "naive": naive_list,
        "locart": list_locart_quantiles,
        "loforest_fixed": loforest_cutoffs,
        "loforest_tuned": loforest_tuned_cutoffs,
        "boosting": boosting_quantiles,
    }

    return quantile_dict, K_loforest


# fitting posterior model
def fit_post_model(
    simulator,
    prior,
    nuisance_idx=None,
    log_transf=False,
    B_model=20000,
    n=1,
    seed=45,
    split_seed=0,
    n_flows=8,
    hidden_units=128,
    hidden_layers=2,
    enable_cuda=True,
    patience=50,
    n_epochs=2000,
    batch_size=250,
    type_flow="Quadratic Spline",
    plot_history=True,
    two_moons=False,
    poisson=False,
    glm=False,
    alpha=0.5,
    X_mat=None,
):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # simulating thetas
    if two_moons:
        thetas = prior(num_samples=B_model)
    elif poisson:
        thetas = prior(n=(B_model,))
    elif glm:
        thetas = prior(n=B_model)
    else:
        thetas = prior.sample((B_model,))

    if not glm:
        repeated_thetas = thetas.repeat_interleave(repeats=n, dim=0)

        # simulating X's
        X_sample = simulator(repeated_thetas)
        # applying log if needed
        if log_transf:
            X_sample = torch.log(X_sample)
        X_dim = X_sample.shape[1]
        X_net = X_sample.reshape(B_model, n * X_dim)
    else:
        X_net = simulator(thetas, X_mat=X_mat, alpha=alpha)

    if nuisance_idx is not None:
        size = thetas.shape[1]
        # idx list
        idx_array = np.arange(0, size)
        par_idx = idx_array[np.where(idx_array != nuisance_idx)]
        thetas = thetas[:, par_idx]

    if thetas.ndim == 1:
        thetas = thetas.reshape(-1, 1)

    nflow_post = normflow_posterior(
        latent_size=thetas.shape[1],
        sample_size=X_net.shape[1],
        n_flows=n_flows,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers,
        enable_cuda=enable_cuda,
    )

    nflow_post.fit(
        X_net.numpy(),
        thetas.numpy(),
        patience=patience,
        n_epochs=n_epochs,
        batch_size=batch_size,
        split_seed=split_seed,
        type=type_flow,
    )
    if plot_history:
        nflow_post.plot_history()

    return nflow_post
