from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest, tune_loforest_LFI
from CP2LFI.scores import LambdaScore

from clover import Scores
from clover import LocartSplit

from tqdm import tqdm

from scipy import stats
from scipy.optimize import minimize_scalar

import time

import os
from os import path

original_path = os.getcwd()


def sim_gmm(B, theta, rng):
    group = rng.binomial(n=1, p=0.5, size=B)
    X = ((group == 0) * (rng.normal(theta, 1, size=B))) + (
        (group == 1) * (rng.normal(-theta, 1, size=B))
    )
    return X


def sim_lambda(theta, rng, B=1000, N=100):
    lambdas = np.zeros(B)
    for i in range(0, B):
        X = sim_gmm(B=N, theta=theta, rng=rng)
        lambdas[i] = compute_lrt_statistic(theta, X)
    return lambdas


# randomly sampling from gmm
def sample_gmm(n, N, rng):
    thetas = rng.uniform(0, 5, size=n)
    lambdas = np.zeros(n)
    i = 0
    for theta in thetas:
        X = sim_gmm(B=N, theta=theta, rng=rng)
        lambdas[i] = compute_lrt_statistic(theta, X)
        i += 1
    return thetas, lambdas


# likelihood function
def l_func(theta, x):
    # prob from X
    p_x = np.log(
        (0.5 * stats.norm.pdf(x, loc=theta, scale=1))
        + (0.5 * stats.norm.pdf(x, loc=-theta, scale=1))
    )
    return -(np.sum(p_x))


# likelihood ratio statistic
def compute_lrt_statistic(theta_0, X, lower=0, upper=5):
    # computing MLE by grid
    res = minimize_scalar(
        l_func, args=(X), bounds=(lower, upper), tol=0.01, options={"maxiter": 100}
    )
    mle_theta = res.x
    lrt_stat = -2 * ((-l_func(theta_0, X)) - (-l_func(mle_theta, X)))
    return lrt_stat


def naive(alpha, rng, B=1000, N=100, lower=0, upper=5, naive_n=100):
    n_grid = int(B / naive_n)
    thetas = np.linspace(lower, upper, n_grid)
    quantiles = {}
    for theta in thetas:
        LRT_stat = sim_lambda(theta=theta, B=naive_n, N=N, rng=rng)
        quantiles[theta] = np.quantile(LRT_stat, q=1 - alpha)
    return quantiles


# naive predict function
def predict_naive_quantile(theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for theta in theta_grid:
        idx = thetas_values[int(np.argmin(np.abs(theta - thetas_values)))]
        quantiles_list.append(quantiles_dict[idx])
    return quantiles_list


# function to obtain all quantiles for a given B, sample size N and theta grid
def obtain_quantiles(
    thetas,
    N,
    rng,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    naive_n=500,
    K=50,
    B_valid=1000,
    N_lambda=500,
    K_grid=np.arange(30, 85, 5),
):
    # fitting and predicting naive
    naive_quantiles = naive(alpha=alpha, B=B, N=N, naive_n=naive_n, rng=rng)
    naive_list = predict_naive_quantile(thetas, naive_quantiles)

    # state of rng
    rng_state = rng.bit_generator.state

    # simulating to fit loforest
    theta_sim, model_lambdas = sample_gmm(n=B, N=N, rng=rng)

    model_thetas = theta_sim.reshape(-1, 1)

    # locart
    locart_object = LocartSplit(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    locart_quantiles = locart_object.calib(
        model_thetas, model_lambdas, min_samples_leaf=min_samples_leaf
    )
    idxs = locart_object.cart.apply(thetas.reshape(-1, 1))
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # loforest fixed quantiles
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
    loforest_fixed_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 1))

    # boosting
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

    theta_valid, _ = sample_gmm(
        n=B_valid,
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

    loforest_cutoffs = loforest_object.compute_cutoffs(
        thetas.reshape(-1, 1), K=K_loforest
    )
    end_time = time.time()

    running_time = end_time - start_time
    # print(f"Loforest took {running_time} seconds to run.")

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
    n_out=500,
    n=1000,
    N=np.array([5, 10, 20, 50]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    min_samples_leaf=300,
    K=50,
    K_grid=np.arange(30, 85, 5),
    B_valid=1000,
    N_lambda=750,
    n_it=100,
    naive_n=100,
):
    folder_path = "/experiments/results_data"
    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)

    rng = np.random.default_rng(seed)
    # generate testing grid
    thetas = np.linspace(0.001, 4.999, n_out)
    N_list = []
    mae_list = []
    se_list = []
    B_list = []
    methods_list = []
    rng_simulate_list = []
    rng_test_list = []

    for N_fixed in tqdm(N, desc="N"):
        for B_fixed in tqdm(B, desc="B"):
            mae_vector = np.zeros((n_it, 5))
            for it in tqdm(range(0, n_it), desc="iteracoes"):
                # Obtain the quantiles
                quantiles_dict, K_loforest, rng_state = obtain_quantiles(
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
                # saving rng_state
                rng_simulate_list.append(rng_state)

                # Check if the true lambda values fall within the predicted quantiles
                l = 0
                for theta in thetas:
                    # saving rng to simulate sim_lambda
                    sim_rng_state = rng.bit_generator.state
                    rng_test_list.append(sim_rng_state)

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

            stats_data.to_csv(original_path + folder_path + "/gmm_data_tuned.csv")

            # saving rng state lists
            rng_simulate_array = np.array(rng_simulate_list)
            np.savez(
                original_path + folder_path + "/gmm_rng_simulate_list",
                rng_simulate_array,
                allow_pickle=True,
            )

            rng_test_array = np.array(rng_test_list)
            np.savez(
                original_path + folder_path + "/gmm_rng_test_list",
                rng_test_array,
                allow_pickle=True,
            )


if __name__ == "__main__":
    print(
        "We will now compute all MAE statistics for the GMM example with tuned loforest"
    )
    n_it = int(input("Input the desired number of experiment repetition to be made: "))
    start_time = time.time()
    stats_df = evaluate_coverage_N_tuned_loforest(
        B=np.array([1000, 5000, 10000, 15000]),
        N=np.array([1, 10, 20, 50]),
        n_out=300,
        n=1000,
        min_samples_leaf=300,
        K=50,
        K_grid=np.arange(30, 90, 5),
        B_valid=1000,
        N_lambda=300,
        naive_n=100,
    )

    end_time = time.time()

    running_time = end_time - start_time
    print(f"The simulation took {running_time} seconds to run.")
