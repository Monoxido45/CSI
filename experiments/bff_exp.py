from sklearn.ensemble import HistGradientBoostingRegressor
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

from scipy import stats
from scipy.optimize import minimize_scalar

import time

import os
from os import path

original_path = os.getcwd()

def sim_X(n, theta):
    X = np.random.normal(theta, 1, n)
    return X

def sim_lambda(B, N, theta, sigma=0.25):
    lambdas = np.zeros(B)
    for i in range(0, B):
        X = sim_X(N, theta)
        lambdas[i] = compute_pdf_posterior(theta, X, sigma=sigma)
    return lambdas


def sample_posterior(n, N, seed=45, sigma=0.25):
    np.random.seed(seed)
    thetas = np.random.uniform(-5, 5, size=n)
    lambdas = np.zeros(n)
    i = 0
    for theta in thetas:
        X = sim_X(N, theta)
        lambdas[i] = compute_pdf_posterior(theta, X, sigma=sigma)
        i += 1
    return thetas, lambdas


def compute_pdf_posterior(theta, x, sigma=0.25):
    n = x.shape[0]
    mu_value = (1 / ((1 / sigma) + n)) * (np.sum(x))
    sigma_value = ((1 / sigma) + n) ** (-1)
    return -stats.norm.pdf(theta, loc=mu_value, scale=np.sqrt(sigma_value))


# naive method
def naive(alpha, B=1000, N=100, lower=-5, upper=5, seed=250, naive_n=100, sigma=0.25):
    np.random.seed(seed)
    n_grid = int(B / naive_n)
    thetas = np.linspace(lower, upper, n_grid)
    quantiles = {}
    for theta in thetas:
        lambdas = sim_lambda(B=naive_n, N=N, theta=theta, sigma=sigma)
        quantiles[theta] = np.quantile(lambdas, q=1 - alpha)
    return quantiles


# naive predict function
def predict_naive_quantile(theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for theta in theta_grid:
        idx = thetas_values[int(np.argmin(np.abs(theta - thetas_values)))]
        quantiles_list.append(quantiles_dict[idx])
    return quantiles_list

def obtain_quantiles(
    thetas,
    N,
    B=1000,
    alpha=0.05,
    naive_seed=45,
    min_samples_leaf=100,
    naive_n=500,
    sigma=0.25,
    sample_seed=25,
):
    # fitting and predicting naive
    naive_quantiles = naive(alpha=alpha, B=B, N=N, naive_n=naive_n, sigma=sigma, seed=naive_seed)
    naive_list = predict_naive_quantile(thetas, naive_quantiles)

    # simulating to fit models
    theta_sim, model_lambdas = sample_posterior(n=B, N=N, seed=sample_seed)
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
        model_thetas, model_lambdas, min_samples_leaf=min_samples_leaf
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

    # naive quantiles
    naive_list = predict_naive_quantile(thetas, naive_quantiles)

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
    n_it = 100,
    N=np.array([1, 10, 100, 1000]),
    B=np.array([500, 1000, 5000, 10000, 15000, 20000]),
    alpha=0.05,
    n=1000,
    seed=45,
    min_samples_leaf=300,
    naive_n=500,
    sigma=0.25,
):
  folder_path = (
            "/experiments/results_data"
            )
            
  if not (path.exists(original_path + folder_path)):
    os.mkdir(original_path + folder_path)
  N_list = []
  methods_list = []
  B_list = []
  mae_list = []
  se_list = []
  j = 0
  np.random.seed(seed)
  for N_fixed in tqdm(N, desc="Computing coverage for each N"):
    for B_fixed in B:
      print("Running example simulating {} samples".format(B_fixed))
      seeds = np.random.randint(
              0, 10**8,
              n_it,
              )
      sample_seeds = np.random.randint(
              0, 10**8,
              n_it,
        )
      h = 0
      mae_vector = np.zeros((n_it, 4))
      for it in range(0, n_it):
        # computing all quantiles for fixed N
        quantiles_dict = obtain_quantiles(
            thetas,
            N=N_fixed,
            B=B_fixed,
            alpha=alpha,
            naive_seed=seeds[it],
            min_samples_leaf=min_samples_leaf,
            naive_n=naive_n,
            sample_seed=sample_seeds[it],
            sigma=sigma,
        )
        err_data = np.zeros((thetas.shape[0], 4))
        l = 0
        for theta in thetas:
          # generating several lambdas
          lambda_stat = sim_lambda(
                B=n,
                N=N_fixed,
                theta=theta,
                sigma=sigma,
            )

          # comparing coverage of methods
          locart_cover = np.mean(lambda_stat <= quantiles_dict["locart"][l])
          loforest_cover = np.mean(lambda_stat <= quantiles_dict["loforest"][l])
          boosting_cover = np.mean(lambda_stat <= quantiles_dict["boosting"][l])
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
        mae_vector[it, :] = np.mean(err_data, axis = 0)
        j += 1
        h += 1
        
      mae_list.extend(np.mean(mae_vector, axis = 0).tolist())
      se_list.extend((np.std(mae_vector, axis = 0)/np.sqrt(n_it)).tolist())
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
    stats_data.to_csv(original_path + folder_path + "/bff_data.csv")

if __name__ == "__main__":
    print("We will now compute all MAE statistics for the BFF example")
    n_out = 500
    thetas = np.linspace(-4.999, 4.999, n_out)
    n_it = int(input("Input the desired numper of experiment repetition to be made: "))
    compute_MAE_N(thetas, n_it = n_it, naive_n = 500)

