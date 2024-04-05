import pickle
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
    theoretical = np.e**(-theta)
    for k in range(0, B):
        exp = rng.exponential(1/theta, N)
        empirical = len([i for i in exp if i > 1])/len(exp)
        lambdas[k] = np.abs(theoretical - empirical)
    return lambdas

def train_naive(alpha, rng, B=1000, N=100, naive_n=500, lower=0.0001, upper=6.9999):
    n_grid = int(B / naive_n)
    if n_grid > 1:
        step = (upper - lower)/n_grid
        thetas_fixed = np.arange(lower, upper, step)
    else:
        step = (upper - lower)/2
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
        theoretical = np.e**(-theta)
        exp = rng.exponential(1/theta, N)
        empirical = (len([i for i in exp if i > 1])/len(exp))
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
    K=50,
):

    theta_sim, model_lambdas = generate_parameters_random(B=B, rng=rng, N=N)
    model_thetas = theta_sim.reshape(-1, 1)

    loforest_object = ConformalLoforest(
        LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
    )
    loforest_object.calibrate(
        model_thetas, model_lambdas, min_samples_leaf=min_samples_leaf, K=K, n_estimators=200,
    )

    loforest_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 1))

    quantile_dict = {
        "loforest": loforest_cutoffs,
    }

    return quantile_dict


def compute_MAE_N(
    thetas,
    n_it=100,
    N=np.array([1, 10, 100, 1000, 5000]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    n=1000,
    seed=45,
    min_samples_leaf=300,
    naive_n=500,
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
            h = 0
            mae_vector = np.zeros((n_it, 1))
            for it in tqdm(range(0, n_it), desc="Computing coverage for each theta"):
                quantiles_dict = obtain_quantiles(
                    thetas,
                    N=N_fixed,
                    B=B_fixed,
                    alpha=alpha,
                    min_samples_leaf=min_samples_leaf,
                    naive_n=naive_n,
                    rng=rng,
                    K=K,
                )
                err_data = np.zeros((thetas.shape[0], 1))
                l = 0
                for theta in thetas:
                    lambda_stat = sim_lambda(
                        B=n,
                        N=N_fixed,
                        theta=theta,
                        rng=rng,
                    )

                    loforest_cover = np.mean(lambda_stat <= quantiles_dict["loforest"][l])

                    err_loforest = np.abs(loforest_cover - (1 - alpha))

                    err_data[l, :] = np.array(
                        [err_loforest]
                    )

                    l += 1
                mae_vector[it, :] = np.mean(err_data, axis=0)
                j += 1
                h += 1
                
            mae_list.extend(np.mean(mae_vector, axis=0).tolist())
            se_list.extend((np.std(mae_vector, axis=0)/np.sqrt(n_it)).tolist())
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


if __name__ == "__main__":
    print("We will now compute all MAE statistics for the exponential example")
    n_out = 500
    thetas = np.linspace(0.0001, 6.9999, n_out)
  
    K_s = range(30, 90, 5)
    cov_5000 = dict()

    for k in K_s:

        cov_5000[k] = compute_MAE_N(
            thetas,
            B=np.array([1000, 5000, 10000, 15000]),
            naive_n=100,
            K=k,
            seed=1250,
        )
        print(f"Done for K = {k}")

        with open("experiment_K_exp.pkl", "wb") as f:
            pickle.dump(cov_5000, f)
