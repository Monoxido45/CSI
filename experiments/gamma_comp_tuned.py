from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest, tune_loforest_LFI
from CP2LFI.scores import LambdaScore

from clover import LocartSplit

from tqdm import tqdm

from scipy import stats

import itertools
import time

import os
from os import path

original_path = os.getcwd()


def sim_gamma(gamma_shape, gamma_scale, n, rng, threshold=None):
    random_x = rng.gamma(gamma_shape, gamma_scale, n)

    if not threshold:
        threshold = rng.choice(random_x)

    emp_prob = np.mean(random_x > threshold)
    t_prob = 1 - stats.gamma.cdf(threshold, a=gamma_shape, scale=gamma_scale)

    lambda_stat = np.abs(t_prob - emp_prob)

    return lambda_stat


# randomly sampling from gamma
def sample_gamma(n, N, rng, threshold=None):
    thetas = np.c_[rng.uniform(2, 8, n), rng.uniform(4, 10, n)]
    lambdas = np.zeros(n)
    i = 0
    for shape, scale in thetas:
        lambdas[i] = sim_gamma(
            gamma_shape=shape, gamma_scale=scale, n=N, rng=rng, threshold=threshold
        )
        i += 1
    return thetas, lambdas


def naive(alpha, rng, B=1000, N=100, seed=250, naive_n=100, threshold=None):
    np.random.seed(seed)
    n_grid = round(np.sqrt(B / naive_n))
    a_s = np.linspace(2.0001, 7.9999, n_grid)
    b_s = np.linspace(4.0001, 9.9999, n_grid)

    quantiles = {}
    for shape, scale in itertools.product(a_s, b_s):
        lambda_stat = np.zeros(naive_n)
        for i in range(naive_n):
            lambda_stat[i] = sim_gamma(
                gamma_shape=shape,
                gamma_scale=scale,
                n=N,
                rng=rng,
                threshold=threshold,
            )

        quantiles[(shape, scale)] = np.quantile(lambda_stat, q=1 - alpha)
    return quantiles


# naive predict function
def predict_naive_quantile(theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for x in theta_grid:
        distances = np.linalg.norm(thetas_values - x, axis=1)
        idx = thetas_values[np.argmin(distances)]
        quantiles_list.append(quantiles_dict[tuple(idx)])
    return quantiles_list


def obtain_quantiles(
    thetas,
    N,
    rng,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=200,
    K=50,
    threshold=None,
    B_valid=1000,
    N_lambda=500,
    naive_n=500,
    K_grid=np.arange(30, 85, 5),
):
    # fitting and predicting naive
    naive_quantiles = naive(
        alpha, rng=rng, B=B, N=N, naive_n=naive_n, threshold=threshold
    )
    naive_list = predict_naive_quantile(thetas, naive_quantiles)

    # state of rng
    rng_state = rng.bit_generator.state

    # simulating to fit models
    thetas_sim, model_lambdas = sample_gamma(n=B, N=N, rng=rng, threshold=threshold)

    model_thetas = thetas_sim.reshape(-1, 2)

    # locart
    locart_object = LocartSplit(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    locart_quantiles = locart_object.calib(
        model_thetas, model_lambdas, min_samples_leaf=100
    )

    idxs = locart_object.cart.apply(thetas.reshape(-1, 2))
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # loforest fixed quantiles
    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    loforest_object.calibrate(
        model_thetas,
        model_lambdas,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        K=K,
    )
    loforest_fixed_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 2))

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
    boosting_quantiles = model.predict(thetas.reshape(-1, 2))

    # loforest tuned quantiles
    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    loforest_object.calibrate(
        model_thetas,
        model_lambdas,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        K=K,
    )

    theta_valid, _ = sample_gamma(n=B_valid, N=N, rng=rng, threshold=threshold)
    model_thetas_valid = theta_valid.reshape(-1, 2)
    lambda_valid = np.zeros((theta_valid.shape[0], N_lambda))

    i = 0
    for theta in theta_valid:
        for j in range(0, N_lambda):
            lambda_valid[i, j] = sim_gamma(
                gamma_shape=theta[0],
                gamma_scale=theta[1],
                n=N,
                rng=rng,
                threshold=threshold,
            )
        i += 1

    # tuning K using validation set
    K_loforest = tune_loforest_LFI(
        loforest_object, model_thetas_valid, lambda_valid, K_grid=K_grid
    )

    loforest_cutoffs = loforest_object.compute_cutoffs(
        thetas.reshape(-1, 2), K=K_loforest
    )

    quantile_dict = {
        "naive": naive_list,
        "locart": list_locart_quantiles,
        "loforest_fixed": loforest_fixed_cutoffs,
        "loforest_tuned": loforest_cutoffs,
        "boosting": boosting_quantiles,
    }

    return quantile_dict, K_loforest, rng_state


def evaluate_coverage_N_tuned_loforest(
    seed=45,
    n_out=50,
    n=1000,
    N=np.array([5, 10, 20, 50]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    min_samples_leaf=300,
    K=50,
    K_grid=np.arange(30, 85, 5),
    B_valid=1000,
    N_lambda=750,
    threshold=30,
    n_it=100,
    naive_n=100,
):
    folder_path = "/experiments/results_data"
    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)

    rng = np.random.default_rng(seed)
    # generate testing grid
    a_s = np.linspace(2.0001, 7.9999, n_out)
    b_s = np.linspace(4.0001, 9.9999, n_out)
    thetas = np.c_[list(itertools.product(a_s, b_s))]
    N_list = []
    methods_list = []
    B_list = []
    mae_list = []
    se_list = []
    rng_simulate_list = []
    rng_test_list = []

    for N_fixed in tqdm(N, desc="N"):
        for B_fixed in tqdm(B, desc="B"):
            mae_vector = np.zeros((n_it, 5))
            for it in tqdm(range(0, n_it), desc="iteracoes"):
                # Obtain the quantiles
                quantiles_dict, K_loforest, rng_state = obtain_quantiles(
                    thetas=thetas,
                    N=N_fixed,
                    rng=rng,
                    B=B_fixed,
                    alpha=alpha,
                    min_samples_leaf=min_samples_leaf,
                    K=K,
                    K_grid=K_grid,
                    B_valid=B_valid,
                    N_lambda=N_lambda,
                    threshold=threshold,
                    naive_n=naive_n,
                )

                err_data = np.zeros((thetas.shape[0], 5))

                # print("tuned K = {}".format(K_loforest))
                rng_simulate_list.append(rng_state)

                # Check if the true lambda values fall within the predicted quantiles
                l = 0
                for theta in thetas:
                    sim_rng_state = rng.bit_generator.state
                    rng_test_list.append(sim_rng_state)

                    lambda_stat = np.zeros(n)
                    for j in range(0, n):
                        lambda_stat[j] = sim_gamma(
                            gamma_shape=theta[0],
                            gamma_scale=theta[1],
                            n=N_fixed,
                            rng=rng,
                            threshold=threshold,
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

            mae_list.extend(np.mean(mae_vector, axis=0).tolist())
            methods_list.extend(
                ["LOCART", "LOFOREST TUNED", "LOFOREST FIXED", "BOOSTING", "NAIVE"]
            )
            se_list.extend((np.std(mae_vector, axis=0) / (np.sqrt(n_it))).tolist())
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

            stats_data.to_csv(original_path + folder_path + "/gamma_data_tuned.csv")

            # saving rng state lists
            rng_simulate_array = np.array(rng_simulate_list)
            np.savez(
                original_path + folder_path + "/gamma_rng_simulate_list",
                rng_simulate_array,
                allow_pickle=True,
            )

            rng_test_array = np.array(rng_test_list)
            np.savez(
                original_path + folder_path + "/gamma_rng_test_list",
                rng_test_array,
                allow_pickle=True,
            )


if __name__ == "__main__":
    print(
        "We will now compute all MAE statistics for the gamma example with tuned loforest"
    )
    n_it = int(input("Input the desired number of experiment repetition to be made: "))
    start_time = time.time()

    stats_df = evaluate_coverage_N_tuned_loforest(
        B=np.array([1000, 5000, 10000, 15000]),
        N=np.array([5, 10, 20, 50]),
        n_out=50,
        n_it=n_it,
        n=1000,
        min_samples_leaf=300,
        K=50,
        K_grid=np.arange(30, 90, 5),
        B_valid=1000,
        N_lambda=750,
        threshold=30,
        naive_n=100,
    )

    end_time = time.time()

    running_time = end_time - start_time
    print(f"The simulation took {running_time} seconds to run.")
