from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest, tune_loforest_LFI
from CP2LFI.scores import LambdaScore

from clover import LocartSplit

import time
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
    min_samples_leaf=300,
    naive_n=500,
    K=50,
    B_valid=1000,
    N_lambda=500,
    K_grid=np.arange(30, 85, 5),
):
    # naive quantiles
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

    idxs = locart_object.cart.apply(thetas.reshape(-1, 1))
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

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
    boosting_quantiles = model.predict(thetas.reshape(-1, 1))

    # loforest tuned quantiles
    start_time = time.time()
    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False, tune_K=False
    )
    loforest_object.calibrate(
        model_thetas,
        model_lambdas,
        min_samples_leaf=min_samples_leaf,
        K=K,
        n_estimators=200,
    )

    theta_valid, _ = generate_parameters_random(
        B=B_valid,
        N=N,
        rng=rng,
    )
    model_thetas_valid = theta_valid.reshape(-1, 1)
    lambda_valid = np.zeros((theta_valid.shape[0], N_lambda))
    i = 0
    for theta in theta_valid:
        lambda_valid[i, :] = sim_lambda(B=N_lambda, N=N, theta=theta, rng=rng)
        i += 1

    start_tune = time.time()
    # tuning K using validation set
    K_loforest = tune_loforest_LFI(
        loforest_object, model_thetas_valid, lambda_valid, K_grid=K_grid
    )
    end_tune = time.time()

    running_tune = end_tune - start_tune
    # print(f"Tuning took {running_tune} seconds to run for N = {N} and B = {B}.")

    loforest_tuned_cutoffs = loforest_object.compute_cutoffs(
        thetas.reshape(-1, 1), K=K_loforest
    )
    end_time = time.time()

    # loforest fixed quantiles
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
    loforest_fixed_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 1))

    running_time = end_time - start_time
    # print(f"Loforest took {running_time} seconds to run.")

    quantile_dict = {
        "loforest_fixed": loforest_fixed_cutoffs,
        "loforest_tuned": loforest_tuned_cutoffs,
        "naive": naive_list,
        "locart": list_locart_quantiles,
        "boosting": boosting_quantiles,
    }

    return quantile_dict, K_loforest


def evaluate_coverage_N_tuned_loforest(
    seed=45,
    n_out=500,
    n=1000,
    n_it=100,
    N=np.array([5, 10, 20, 50]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    min_samples_leaf=300,
    K=50,
    naive_n=100,
    K_grid=np.arange(30, 85, 5),
    B_valid=1000,
    N_lambda=750,
):
    rng = np.random.default_rng(seed)
    # generate testing grid
    thetas = np.linspace(0.0001, 6.9999, n_out)
    N_list = []
    mae_list = []
    se_list = []
    B_list = []
    methods_list = []

    for N_fixed in tqdm(N, desc="N"):
        for B_fixed in tqdm(B, desc="B"):
            mae_vector = np.zeros((n_it, 5))
            for it in tqdm(range(0, n_it), desc="iteracoes"):
                # Obtain the quantiles
                quantiles_dict, K_loforest = obtain_quantiles(
                    thetas,
                    N_fixed,
                    rng,
                    B=B_fixed,
                    alpha=alpha,
                    min_samples_leaf=min_samples_leaf,
                    K=K,
                    K_grid=K_grid,
                    B_valid=B_valid,
                    N_lambda=N_lambda,
                    naive_n=naive_n,
                )

                err_data = np.zeros((thetas.shape[0], 5))

                # print("tuned K = {}".format(K_loforest))

                # Check if the true lambda values fall within the predicted quantiles
                l = 0
                for theta in thetas:
                    lambda_stat = sim_lambda(
                        B=n,
                        N=N_fixed,
                        theta=theta,
                        rng=rng,
                    )

                    locart_coverage = np.mean(
                        lambda_stat <= quantiles_dict["locart"][l]
                    )
                    loforest_tuned_coverage = np.mean(
                        lambda_stat <= quantiles_dict["loforest_tuned"][l]
                    )
                    loforest_fixed_coverage = np.mean(
                        lambda_stat <= quantiles_dict["loforest_fixed"][l]
                    )
                    boosting_coverage = np.mean(
                        lambda_stat <= quantiles_dict["boosting"][l]
                    )
                    naive_coverage = np.mean(lambda_stat <= quantiles_dict["naive"][l])

                    err_locart = np.abs(locart_coverage - (1 - alpha))
                    err_loforest_tuned = np.abs(loforest_tuned_coverage - (1 - alpha))
                    err_loforest_fixed = np.abs(loforest_fixed_coverage - (1 - alpha))
                    err_boosting = np.abs(boosting_coverage - (1 - alpha))
                    err_naive = np.abs(naive_coverage - (1 - alpha))

                    err_data[l, :] = np.array(
                        [
                            err_locart,
                            err_loforest_tuned,
                            err_loforest_fixed,
                            err_boosting,
                            err_naive,
                        ]
                    )
                    l += 1
                mae_vector[it, :] = np.mean(err_data, axis=0)

            # K_list.extend([K_loforest])
            mae_list.extend(np.mean(mae_vector, axis=0).tolist())
            methods_list.extend(
                ["LOCART", "LOFOREST TUNED", "LOFOREST FIXED", "BOOSTING", "NAIVE"]
            )
            se_list.extend(
                (np.std(err_data, axis=0) / (np.sqrt(thetas.shape[0]))).tolist()
            )
            N_list.extend([N_fixed] * 5)
            B_list.extend([B_fixed] * 5)

    stats_data = pd.DataFrame(
        {
            "methods": methods_list,
            "N": N_list,
            "B": B_list,
            "MAE": mae_list,
            "se": se_list,
        }
    )

    folder_path = "/experiments/results_data"

    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)

    stats_data.to_csv(original_path + folder_path + "/exponential_data_tuned.csv")


if __name__ == "__main__":
    print(
        "We will now compute all MAE statistics for the exponential example with tuned loforest"
    )
    n_it = int(input("Input the desired number of experiment repetition to be made: "))

    start_time = time.time()
    stats_df = evaluate_coverage_N_tuned_loforest(
        B=np.array([1000, 5000, 10000, 15000]),
        N=np.array([5, 10, 20, 50]),
        n_out=500,
        n=1000,
        min_samples_leaf=300,
        K=50,
        K_grid=np.arange(30, 90, 5),
        B_valid=1000,
        naive_n=100,
        n_it=n_it,
    )
    end_time = time.time()

    running_time = end_time - start_time
    print(f"The simulation took {running_time} seconds to run.")
