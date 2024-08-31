from hypothesis.benchmark import weinberg
from CP2LFI.loforest import ConformalLoforest
from CP2LFI.scores import LambdaScore
import torch
import numpy as np
import os
import pandas as pd

# general path
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
tune_path = "/results/LFI_objects/tune_data/"
stats_eval_path = "/results/LFI_objects/stat_data/"


# function to compute TRUST++ cutoffs for several candidates for K
def obtain_quantiles(
    score,
    theta_grid_eval,
    simulator,
    prior,
    N,
    B=1000,
    alpha=0.05,
    min_samples_leaf=100,
    n_estimators=200,
    K_grid=np.array([100, 125, 150, 175, 200]),
    disable_tqdm=True,
    log_transf=False,
    split_calib=False,
):
    # simulating to fit TRUST++ models
    thetas_sim = prior.sample((B,))

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

    quantile_dict = {}

    # grid of thetas for evaluation
    if theta_grid_eval.ndim == 1:
        model_eval = theta_grid_eval.reshape(-1, 1)
    else:
        model_eval = theta_grid_eval

    # loforest quantiles for each K
    for K in K_grid:
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

        quantile_dict[f"TRUST_plus_{K}"] = loforest_object.compute_cutoffs(model_eval)

    return quantile_dict


# function to compute MAE for each case for large B in training
def compute_MAE_N_B(
    kind,
    score_name,
    theta_grid_eval,
    simulator,
    prior,
    N=1,
    B=1e5,
    alpha=0.05,
    min_samples_leaf=300,
    n_estimators=200,
    K_grid=np.array([100, 125, 150, 175, 200]),
    disable_tqdm=True,
    seed=45,
    log_transf=False,
    split_calib=False,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    score = pd.read_pickle(
        original_path + stats_path + f"{kind}_{score_name}_{N}.pickle"
    )
    B = int(B)

    quantiles_dict = obtain_quantiles(
        score=score,
        theta_grid_eval=theta_grid_eval,
        simulator=simulator,
        prior=prior,
        N=N,
        B=B,
        alpha=alpha,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        disable_tqdm=disable_tqdm,
        K_grid=K_grid,
        log_transf=log_transf,
        split_calib=split_calib,
    )
    stat_dict = pd.read_pickle(
        original_path + stats_eval_path + f"{kind}_{score_name}_eval_{N}.pickle"
    )

    # column names in list
    str_list = []
    for K in K_grid:
        str_list.append(f"TRUST_plus_{K}")

    err_data = np.zeros((theta_grid_eval.shape[0], K_grid.shape[0]))
    l = 0
    for theta in theta_grid_eval:
        if theta_grid_eval.ndim == 1:
            stat = stat_dict[theta]
        else:
            theta = tuple(theta)
            stat = stat_dict[theta]

        # comparing coverage between methods
        trust_list = []
        for K in K_grid:
            trust_list.append(
                np.abs(
                    np.mean(stat <= quantiles_dict[f"TRUST_plus_{K}"][l]) - (1 - alpha)
                )
            )

        err_data[l, :] = np.array(trust_list)
        l += 1

    mae_array = np.mean(err_data, axis=0)
    se_array = 2 * np.std(err_data, axis=0) / np.sqrt(err_data.shape[0])

    # saving the K column
    stats_data = pd.DataFrame(
        {
            "K": str_list,
            "MAE": mae_array,
            "SE": se_array,
        }
    )
    # saving checkpoint
    return stats_data


# evaluation grid
simulator = weinberg.Simulator(default_beam_energy=40.0)
prior = weinberg.Prior()

n_out = 300
thetas_valid = np.linspace(0.51, 1.49, n_out)
stat_data = compute_MAE_N_B(
    kind="weinberg",
    score_name="bff",
    B=3e5,
    theta_grid_eval=thetas_valid,
    simulator=simulator,
    prior=prior,
)
