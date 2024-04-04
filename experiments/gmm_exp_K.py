from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import pickle
from multiprocessing import Pool

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest
from CP2LFI.scores import LambdaScore

from clover import Scores
from clover import LocartSplit

from copy import deepcopy
from tqdm import tqdm

from scipy import stats
from scipy.optimize import minimize_scalar

import time

import os
from os import path

import warnings
warnings.filterwarnings("ignore")

# main functions to simulate and compute quantiles for gmm
def sim_gmm(B, theta, rng):
    group = rng.binomial(n=1, p=0.5, size=B)
    X = ((group == 0) * (rng.normal(theta, 1, size=B))) + (
        (group == 1) * (rng.normal(-theta, 1, size=B))
    )
    return X

def sim_LRT(B, theta, rng):
    X = sim_gmm(B=B, theta=theta, rng=rng)
    return compute_lrt_statistic(theta, X)


def sim_lambda(theta, rng, B=1000, N=100):
    vLRT = np.vectorize(sim_LRT)
    LRT_stat = vLRT(np.tile(N, B), theta, rng)
    return LRT_stat


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
  
# function to obtain all quantiles for a given B, sample size N and theta grid
def obtain_quantiles(
    thetas,
    N,
    rng,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=200,
    K=40,
):
  
    # simulating to fit loforest
    theta_sim, model_lambdas = sample_gmm(n=B, N=N, rng=rng)
    model_thetas = theta_sim.reshape(-1, 1)

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

    # loforest
    loforest_cutoffs = loforest_object.compute_cutoffs(thetas.reshape(-1, 1))

    # dictionary of quantiles
    quantile_dict = {
        "loforest": loforest_cutoffs
    }

    return quantile_dict  


def compute_MAE_N(
    thetas,
    N=np.array([1, 10, 100, 1000]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    n=1000,
    seed=45,
    min_samples_leaf=100,
    n_estimators=100,
    K=50,
):
    N_list = []
    mae_list = []
    se_list = []
    methods_list = []
    B_list = []
    j = 0
    rng = np.random.default_rng(seed)
    for N_fixed in tqdm(N, desc="Computing coverage for each N"):
        k = 0
        for B_fixed in tqdm(B, desc="Computing coverage for each B"):
            # computing all quantiles for fixed N
            quantiles_dict = obtain_quantiles(
                thetas,
                N=N_fixed,
                B=B_fixed,
                alpha=alpha,
                min_samples_leaf=min_samples_leaf,
                n_estimators=n_estimators,
                K=K,
                rng=rng,
            )
            err_data = np.zeros((thetas.shape[0], 1))
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
                loforest_cover = np.mean(lambda_stat <= quantiles_dict["loforest"][l])
                
                # appending the errors
                err_loforest = np.abs(loforest_cover - (1 - alpha))

                # saving in numpy array
                err_data[l, :] = np.array(
                    [err_loforest]
                )

                l += 1

            mae_list.extend(np.mean(err_data, axis = 0).tolist())
            se_list.extend((np.std(err_data, axis = 0)/np.sqrt(thetas.shape[0])).tolist())
            methods_list.extend(["LOFOREST"])
            N_list.extend([N_fixed])
            B_list.extend([B_fixed])

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
    return stats_data


if __name__ == "__main__":

    n_out = 750
    thetas = np.linspace(-4.999, 4.999, n_out)
    K_s = range(30, 90, 5)
    cov_5000 = dict()

    for k in K_s:

        cov_5000[k] = compute_MAE_N(
            thetas,
            B=np.array([500, 1000, 5000, 10000]),
            N=np.array([1, 10, 20, 50]),
            min_samples_leaf=300,
            n_estimators = 200,
            K = k,
            seed=1250,
        )
        print(f"Done for K = {k}")
        
    # save the list of dataframes cov_5000
    with open("gmm_experiment/cov_5000.pkl", "wb") as f:
        pickle.dump(cov_5000, f)
