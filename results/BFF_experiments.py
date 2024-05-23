from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest, tune_loforest_LFI
from CP2LFI.scores import LambdaScore
from CP2LFI.simulations import Simulations, naive, predict_naive_quantile

from scipy import stats
from clover import LocartSplit

from copy import deepcopy

from tqdm import tqdm
import itertools
import time

import os
from os import path

original_path = os.getcwd()


def obtain_quantiles(
    kind_model,
    thetas,
    N,
    rng,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=200,
    B_valid=1000,
    N_lambda=500,
    K=50,
    naive_n=500,
    K_grid=np.concatenate((np.array([0]), np.arange(20, 95, 5))),
):
    # fitting and predicting naive
    naive_quantiles = naive(
        stat="BFF",
        kind_model=kind_model,
        alpha=alpha,
        rng=rng,
        B=B,
        N=N,
        naive_n=naive_n,
    )
    naive_list = predict_naive_quantile(kind_model, thetas, naive_quantiles)

    # state of rng
    rng_state = rng.bit_generator.state

    # simulating to fit models
    sim_obj = Simulations(rng=rng, kind_model=kind_model)
    thetas_sim, model_lambdas = sim_obj.BFF_sample(B=B, N=N)

    if thetas_sim.ndim == 1:
        model_thetas = thetas_sim.reshape(-1, 1)
    else:
        model_thetas = thetas_sim

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
    model.fit(model_thetas, model_lambdas)

    # tuned loforest
    if kind_model == "1d_normal":
        theta_valid = sim_obj.rng.uniform(-5, 5, B_valid)
    elif kind_model == "gmm":
        theta_valid = sim_obj.rng.uniform(0, 5, B_valid)
    elif kind_model == "lognormal":
        theta_valid = np.c_[
            sim_obj.rng.uniform(-2.5, 2.5, B_valid),
            sim_obj.rng.uniform(0.15, 1.25, B_valid),
        ]

    lambda_valid = np.zeros((theta_valid.shape[0], 500))

    if theta_valid.ndim == 1:
        K_valid_thetas = theta_valid.reshape(-1, 1)
    else:
        K_valid_thetas = theta_valid

    i = 0
    for theta in theta_valid:
        lambda_valid[i, :] = sim_obj.BFF_sim_lambda(theta, B=N_lambda, N=N)
        i += 1

    K_loforest = tune_loforest_LFI(
        loforest_object, K_valid_thetas, lambda_valid, K_grid=K_grid
    )

    if thetas.ndim == 1:
        valid_thetas = thetas.reshape(-1, 1)
    else:
        valid_thetas = thetas

    # locart quantiles
    idxs = locart_object.cart.apply(valid_thetas)
    list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # loforest
    loforest_cutoffs = loforest_object.compute_cutoffs(valid_thetas)

    # boosting
    boosting_quantiles = model.predict(valid_thetas)

    # tuned loforest
    loforest_cutoffs_tuned = loforest_object.compute_cutoffs(valid_thetas, K=K_loforest)

    print("Tuned K: ", K_loforest)
    # lr quantile
    lr_quantiles = np.tile(stats.chi2.ppf(1 - alpha, df=1), thetas.shape[0])

    # dictionary of quantiles
    quantile_dict = {
        "naive": naive_list,
        "locart": list_locart_quantiles,
        "loforest": loforest_cutoffs,
        "tuned_loforest": loforest_cutoffs_tuned,
        "boosting": boosting_quantiles,
    }

    return quantile_dict, K_loforest, rng_state


def compute_MAE_N(
    kind_model,
    thetas,
    N=np.array([1, 10, 100, 1000]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    n=1000,
    seed=45,
    n_estimators=200,
    K=40,
    min_samples_leaf=100,
    B_valid=1000,
    N_lambda=500,
    K_grid=np.concatenate((np.array([0]), np.arange(20, 95, 5))),
    n_it=30,
    naive_n=100,
):
    # fixing path
    folder_path = "/results/res_data"
    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)

    N_list = []
    methods_list = []
    B_list = []
    mae_list = []
    se_list = []
    rng_simulate_list = []
    rng_test_list = []
    K_loforest_list = []

    rng = np.random.default_rng(seed)
    sim_obj = Simulations(rng=rng, kind_model=kind_model)
    for N_fixed in N:
        for B_fixed in B:
            mae_vector = np.zeros((n_it, 5))
            K_list = []
            print(f"Computing for B = {B_fixed} and N = {N_fixed}")
            # computing all quantiles for fixed N
            for it in range(0, n_it):
                quantiles_dict, K_loforest, rng_state = obtain_quantiles(
                    kind_model,
                    thetas,
                    N=N_fixed,
                    B=B_fixed,
                    alpha=alpha,
                    min_samples_leaf=min_samples_leaf,
                    n_estimators=n_estimators,
                    K=K,
                    naive_n=naive_n,
                    B_valid=B_valid,
                    N_lambda=N_lambda,
                    K_grid=K_grid,
                    rng=rng,
                )
                rng_simulate_list.append(rng_state)
                K_list.append(K_loforest)

                err_data = np.zeros((thetas.shape[0], 5))
                l = 0
                for theta in thetas:
                    sim_rng_state = sim_obj.rng.bit_generator.state
                    rng_test_list.append(sim_rng_state)

                    # simulating lambdas for testing
                    stat = sim_obj.BFF_sim_lambda(theta=theta, B=n, N=N_fixed)

                    # comparing coverage of methods
                    locart_cover = np.mean(stat <= quantiles_dict["locart"][l])
                    loforest_cover = np.mean(stat <= quantiles_dict["loforest"][l])
                    loforest_tuned_cover = np.mean(
                        stat <= quantiles_dict["tuned_loforest"][l]
                    )
                    boosting_cover = np.mean(stat <= quantiles_dict["boosting"][l])
                    naive_cover = np.mean(stat <= quantiles_dict["naive"][l])

                    # appending the errors
                    err_locart = np.abs(locart_cover - (1 - alpha))
                    err_loforest = np.abs(loforest_cover - (1 - alpha))
                    err_loforest_tuned = np.abs(loforest_tuned_cover - (1 - alpha))
                    err_boosting = np.abs(boosting_cover - (1 - alpha))
                    err_naive = np.abs(naive_cover - (1 - alpha))

                    # saving in numpy array
                    err_data[l, :] = np.array(
                        [
                            err_locart,
                            err_loforest,
                            err_loforest_tuned,
                            err_boosting,
                            err_naive,
                        ]
                    )
                    l += 1
                mae_vector[it, :] = np.mean(err_data, axis=0)

            mae_list.extend(np.mean(err_data, axis=0).tolist())
            se_list.extend((np.std(mae_vector, axis=0) / (np.sqrt(n_it))).tolist())
            methods_list.extend(
                [
                    "LOCART",
                    "LOFOREST",
                    "tuned LOFOREST",
                    "boosting",
                    "monte-carlo",
                ]
            )
            N_list.extend([N_fixed] * 5)
            B_list.extend([B_fixed] * 5)
            K_loforest_list.extend(np.mean(np.array(K_list)))

            # obtaining MAE and standard error for each method
            stats_data = pd.DataFrame(
                {
                    "methods": methods_list,
                    "N": N_list,
                    "B": B_list,
                    "MAE": mae_list,
                    "se": se_list,
                    "K_tuned": K_loforest_list,
                }
            )
            stats_data.to_csv(
                original_path
                + folder_path
                + "/BFF_{}_stats_data.csv".format(kind_model)
            )

            # saving rng state lists
            rng_simulate_array = np.array(rng_simulate_list)
            np.savez(
                original_path
                + folder_path
                + "/BFF_{}_rng_simulate_list".format(kind_model),
                rng_simulate_array,
                allow_pickle=True,
            )

            rng_test_array = np.array(rng_test_list)
            np.savez(
                original_path
                + folder_path
                + "/BFF_{}_rng_test_list".format(kind_model),
                rng_test_array,
                allow_pickle=True,
            )


if __name__ == "__main__":
    print("We will now compute all MAE statistics for the LR statistic")
    n_it = int(input("Input the desired number of experiment repetition to be made: "))
    kind_model = input("Choose your model between: 1d_normal, gmm and lognormal ")
    if kind_model == "1d_normal":
        n_out = 750
        thetas = np.linspace(-4.999, 4.999, n_out)
    elif kind_model == "gmm":
        n_out = 300
        thetas = np.linspace(0, 4.999, n_out)
    elif kind_model == "lognormal":
        n_out = 50
        a_s = np.linspace(-2.4999, 2.4999, n_out)
        b_s = np.linspace(0.15001, 0.9999, n_out)
        thetas = np.c_[list(itertools.product(a_s, b_s))]

    start_time = time.time()
    stats_df = compute_MAE_N(
        kind_model=kind_model,
        thetas=thetas,
        B=np.array([1000, 5000, 10000, 15000]),
        N=np.array([5, 10, 20, 50]),
        n_it=n_it,
        n=1000,
        min_samples_leaf=300,
        n_estimators=200,
        K=50,
        K_grid=np.concatenate((np.array([0]), np.arange(20, 95, 5))),
        B_valid=1000,
        N_lambda=500,
        naive_n=500,
    )

    end_time = time.time()

    running_time = end_time - start_time
    print(f"The simulation took {running_time} seconds to run.")
