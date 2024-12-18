from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest
from CP2LFI.scores import LambdaScore

from clover import Scores
from clover import LocartSplit

from copy import deepcopy

from tqdm import tqdm

import os
from os import path

original_path = os.getcwd()


def sim_lambda(theta, rng, B=1000, N=100):
    lambdas = np.zeros(B)
    theoretical = np.e ** (-theta)
    for k in range(0, B):
        exp = rng.exponential(1 / theta, N)
        empirical = len([i for i in exp if i > 1]) / len(exp)
        lambdas[k] = np.abs(theoretical - empirical)
    return lambdas


def train_naive(alpha, rng, B=1000, N=100, naive_n=500, lower=0.0001, upper=6.9999):
    # simulating by a fixed theta_grid with size compatible with the amount of samples
    # we want to simulate
    n_grid = int(B / naive_n)
    if n_grid > 1:
        step = (upper - lower) / n_grid
        thetas_fixed = np.arange(lower, upper, step)
    else:
        step = (upper - lower) / 2
        thetas_fixed = np.array([np.arange(lower, upper, step)[1]])

    thetas_fixed = np.linspace(lower, upper, n_grid)

    quantiles = {}
    for theta in thetas_fixed:
        diff = sim_lambda(theta, B=n_grid, N=N, rng=rng)
        quantiles[theta] = np.quantile(diff, q=1 - alpha)
    return quantiles


def predict_naive_quantile(theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for theta in theta_grid:
        idx = thetas_values[int(np.argmin(np.abs(theta - thetas_values)))]
        quantiles_list.append(quantiles_dict[idx])
    return quantiles_list


def generate_parameters_random(rng, B=5000, N=1000):
    random_theta_grid = rng.uniform(0, 7, B)
    lambdas = np.zeros(B)
    i = 0
    for theta in random_theta_grid:
        theoretical = np.e ** (-theta)
        exp = rng.exponential(1 / theta, N)
        empirical = len([i for i in exp if i > 1]) / len(exp)
        lambdas[i] = np.abs(theoretical - empirical)
        i += 1
    return random_theta_grid, lambdas


def obtain_quantiles(
    thetas,
    N,
    rng,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    naive_n=500,
):
    # fitting and predicting naive
    naive_quantiles = train_naive(alpha=alpha, B=B, N=N, naive_n=naive_n, rng=rng)
    naive_list = predict_naive_quantile(thetas, naive_quantiles)

    # simulating to fit models
    theta_sim, model_lambdas = generate_parameters_random(B=B, rng=rng, N=N)
    model_thetas = theta_sim.reshape(-1, 1)

    locart_object = LocartSplit(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    locart_quantiles = locart_object.calib(
        model_thetas, model_lambdas, min_samples_leaf=min_samples_leaf
    )

    # loforest quantiles
    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    loforest_object.calibrate(
        model_thetas,
        model_lambdas,
        min_samples_leaf=min_samples_leaf,
        K=50,
        n_estimators=200,
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
    model.fit(model_thetas, model_lambdas)

    # locart quantiles
    idxs = locart_object.cart.apply(thetas.reshape(-1, 1))
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # loforest
    loforest_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 1))

    # boosting
    boosting_quantiles = model.predict(thetas.reshape(-1, 1))

    # dictionary of quantiles
    quantile_dict = {
        "naive": naive_list,
        "locart": list_locart_quantiles,
        "loforest": loforest_cutoffs,
        "boosting": boosting_quantiles,
    }

    return quantile_dict


def compute_MAE_N(
    thetas,
    n_it=100,
    N=np.array([5, 10, 20, 50]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    n=1000,
    seed=45,
    min_samples_leaf=300,
    naive_n=500,
):
    folder_path = "/experiments/results_data"

    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)
    N_list = []
    methods_list = []
    B_list = []
    mae_list = []
    se_list = []

    rng = np.random.default_rng(seed)
    for N_fixed in tqdm(N, desc="Computing coverage for each N"):
        for B_fixed in B:
            print("Running example simulating {} samples".format(B_fixed))
            mae_vector = np.zeros((n_it, 4))
            for it in range(0, n_it):
                # computing all quantiles for fixed N
                quantiles_dict = obtain_quantiles(
                    thetas,
                    N=N_fixed,
                    B=B_fixed,
                    alpha=alpha,
                    min_samples_leaf=min_samples_leaf,
                    naive_n=naive_n,
                    rng=rng,
                )
                err_data = np.zeros((thetas.shape[0], 4))
                l = 0
                for theta in thetas:
                    # generating several lambdas
                    lambda_stat = sim_lambda(
                        B=n,
                        N=N_fixed,
                        theta=theta,
                        rng=rng,
                    )

                    # comparing coverage of methods
                    locart_cover = np.mean(lambda_stat <= quantiles_dict["locart"][l])
                    loforest_cover = np.mean(
                        lambda_stat <= quantiles_dict["loforest"][l]
                    )
                    boosting_cover = np.mean(
                        lambda_stat <= quantiles_dict["boosting"][l]
                    )
                    naive_cover = np.mean(lambda_stat <= quantiles_dict["naive"][l])

                    # appending the errors
                    err_locart = np.abs(locart_cover - (1 - alpha))
                    err_loforest = np.abs(loforest_cover - (1 - alpha))
                    err_boosting = np.abs(boosting_cover - (1 - alpha))
                    err_naive = np.abs(naive_cover - (1 - alpha))

                    # saving in numpy array
                    err_data[l, :] = np.array(
                        [err_locart, err_loforest, err_boosting, err_naive]
                    )

                    l += 1
                mae_vector[it, :] = np.mean(err_data, axis=0)

            mae_list.extend(np.mean(mae_vector, axis=0).tolist())
            se_list.extend((np.std(mae_vector, axis=0) / np.sqrt(n_it)).tolist())
            methods_list.extend(["LOCART", "LOFOREST", "boosting", "naive"])
            N_list.extend([N_fixed] * 4)
            B_list.extend([B_fixed] * 4)

        # obtaining MAE and standard error for each method
        stats_data = pd.DataFrame(
            {
                "methods": methods_list,
                "N": N_list,
                "B": B_list,
                "MAE": mae_list,
                "se": se_list,
            }
        )
        # saving data
        stats_data.to_csv(original_path + folder_path + "/exp_data.csv")


if __name__ == "__main__":
    print("We will now compute all MAE statistics for the exponential example")
    n_out = 500
    thetas = np.linspace(0.0001, 6.9999, n_out)
    n_it = int(input("Input the desired numper of experiment repetition to be made: "))
    compute_MAE_N(
        thetas, B=np.array([1000, 5000, 10000, 15000]), n_it=n_it, naive_n=100
    )
