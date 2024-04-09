from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy import stats
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
import itertools

original_path = os.getcwd()

# simulator
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
        lambdas[i] = sim_gamma(gamma_shape=shape, 
                               gamma_scale=scale, 
                               n=N, 
                               rng=rng, 
                               threshold=threshold)
        i += 1
    return thetas, lambdas


# naive method
def naive(alpha, rng, B=1000, N=100, seed=250, naive_n=100, threshold=None):
    np.random.seed(seed)
    n_grid = round(np.sqrt(B / naive_n))
    a_s = np.linspace(4.0001, 9.9999, n_grid)
    b_s = np.linspace(2.0001, 7.9999, n_grid)
    
    quantiles = {}
    for shape, scale in itertools.product(b_s, a_s):
        lambda_stat = np.zeros(naive_n)
        for i in range(naive_n):
            lambda_stat[i] = sim_gamma(
                gamma_shape=shape, 
                gamma_scale=scale, 
                n=N, 
                rng=rng,
                threshold=threshold)
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
  
# function to obtain quantiles
def obtain_quantiles(
    thetas,
    N,
    rng,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=200,
    K=50,
    naive_n=500,
    threshold=None
):
    # fitting and predicting naive
    #naive_quantiles = naive(alpha, rng=rng, B=B, N=N, naive_n=naive_n, threshold=threshold)
    #naive_list = predict_naive_quantile(thetas, naive_quantiles)

    # simulating to fit models
    thetas_sim, model_lambdas = sample_gamma(n=B, N=N, rng=rng, threshold=threshold)

    model_thetas = thetas_sim.reshape(-1, 2)

    # locart_object = LocartSplit(
    #     LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    # )
    # locart_quantiles = locart_object.calib(
    #     model_thetas, model_lambdas, min_samples_leaf=100
    # )

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
    # model = HistGradientBoostingRegressor(
    #     loss="quantile",
    #     max_iter=100,
    #     max_depth=3,
    #     quantile=1 - alpha,
    #     random_state=105,
    #     n_iter_no_change=15,
    #     early_stopping=True,
    # )
    # model.fit(model_thetas, model_lambdas)

    # # naive quantiles
    # naive_list = predict_naive_quantile(thetas, naive_quantiles)

    # # locart quantiles
    # idxs = locart_object.cart.apply(thetas.reshape(-1, 2))
    # list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]

    # # loforest
    loforest_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 2))

    # # boosting
    # boosting_quantiles = model.predict(thetas.reshape(-1, 2))

    # dictionary of quantiles
    quantile_dict = {
        #"naive": naive_list,
        #"locart": list_locart_quantiles,
        "loforest": loforest_cutoffs,
        #"boosting": boosting_quantiles,
    }

    return quantile_dict

def compute_MAE_N(
    thetas,
    n_it=100,
    N=np.array([10, 100, 1000, 5000]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    n=1000,
    seed=45,
    min_samples_leaf=300,
    naive_n=500,
    threshold=30,
    n_estimators=200,
    K=50,
):
    folder_path = "/experiments/results_data"
            
    if not (path.exists(original_path + folder_path)):
        os.mkdir(original_path + folder_path)
    N_list = []
    methods_list = []
    B_list = []
    mae_list = []
    se_list = []
    j = 0
    rng = np.random.default_rng(seed)
    for N_fixed in tqdm(N, desc="Computing coverage for each N"):
        for B_fixed in tqdm(B, desc="Computing coverage for each B"):
            quantiles_dict = obtain_quantiles(
                thetas,
                N=N_fixed,
                B=B_fixed,
                alpha=alpha,
                min_samples_leaf=min_samples_leaf,
                naive_n=naive_n,
                rng=rng,
                threshold=threshold,
                n_estimators=n_estimators,
                K=K,
            )
            err_data = np.zeros((thetas.shape[0], 1))
            l = 0
            for theta in thetas:
                lambda_stat = np.zeros(n)
                for i in range(0, n):
                    lambda_stat[i] = sim_gamma(
                        gamma_shape=theta[0], 
                        gamma_scale=theta[1], 
                        n=N_fixed, 
                        rng=rng,
                        threshold=threshold,
                    )

                loforest_cover = np.mean(lambda_stat <= quantiles_dict["loforest"][l])

                err_loforest = np.abs(loforest_cover - (1 - alpha))

                err_data[l, :] = np.array(
                    [err_loforest]
                )

                l += 1
                
            mae_list.extend(np.mean(err_data, axis = 0).tolist())
            se_list.extend((np.std(err_data, axis = 0)/np.sqrt(thetas.shape[0])).tolist())
            methods_list.extend(["LOFOREST"])
            N_list.extend([N_fixed] * 1)
            B_list.extend([B_fixed] * 1)

    stats_data = pd.DataFrame(
        {
            "methods": methods_list,
            "N": N_list,
            "B": B_list,
            "MAE": mae_list,
            "se": se_list,
        }
    )
    
    return stats_data


original_path = os.getcwd()
new_path = original_path + "/experiments/results_data/"

if __name__ == "__main__":
    print("We will now compute all MAE statistics for the gamma example")
    n_out = 50
    a_s = np.linspace(4.0001, 9.9999, n_out)
    b_s = np.linspace(2.0001, 7.9999, n_out)
    thetas_grid = np.c_[list(itertools.product(a_s, b_s))]

    K_s = range(30, 90, 5)
    cov_5000 = dict()

    for k in K_s:
        cov_5000[k] = compute_MAE_N(
            thetas_grid,
            B=np.array([1000, 5000, 10000, 15000]),
            N=np.array([1, 10, 20, 50]),
            min_samples_leaf=300,
            n_estimators=200,
            naive_n=100,
            K=k,
            seed=1250,
        )
        print(f"Done for K = {k}")

        with open(new_path + "experiment_K_gamma.pkl", "wb") as f:
            pickle.dump(cov_5000, f)
