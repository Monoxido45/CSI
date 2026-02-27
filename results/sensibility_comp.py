import numpy as np
import matplotlib.pyplot as plt
import sbibm
from sklearn.metrics import pairwise_distances
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
from argparse import ArgumentParser

import time
from CSI.scores import LambdaScore
from CSI.loforest import ConformalLoforest

from CSI.utils import CPU_Unpickler
import itertools
from tqdm import tqdm
import os
import pickle
import torch

# paths
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
tune_path = "/results/LFI_objects/tune_data/"
stats_eval_path = "/results/LFI_objects/stat_data/"

alpha = 0.05
B = 10000
N = 5
# two moons task
task = sbibm.get_task(
"two_moons"
)  # See sbibm.get_available_tasks() for all tasks
simulator = task.get_simulator()
prior = task.get_prior()
two_moons = True

n_par = 20
pars_1 = np.linspace(-0.99, 0.99, n_par)
thetas_valid = np.c_[list(itertools.product(pars_1, pars_1))]

# importing BFF score
with open(
    original_path + stats_path + f"two_moons_bff_{N}.pickle", "rb"
) as f:
    score = CPU_Unpickler(f).load()
score.base_model.device = torch.device("cpu")
score.base_model.model.device = torch.device("cpu")
score.base_model.model.to(torch.device("cpu"))

# importing statistics for evaluation
with open(
    original_path
    + stats_eval_path
    + f"two_moons_bff_eval_{N}.pickle",
    "rb",
) as f:
    stat_dict = CPU_Unpickler(f).load()

# simulating B = 10000 samples for this
thetas_sim = prior(num_samples=B)
if thetas_sim.ndim == 1:
    model_thetas = thetas_sim.reshape(-1, 1)
else:
    model_thetas = thetas_sim

repeated_thetas = thetas_sim.repeat_interleave(repeats=N, dim=0)
X_net = simulator(repeated_thetas)
X_dim = X_net.shape[1]

X_net = X_net.reshape(B, N * X_dim)
model_lambdas = score.compute(
    model_thetas.numpy(), X_net.numpy(), disable_tqdm=False
)

def run_sensitivity_analysis(n_repetitions=10, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_runs_data = []
    
    # Ranges to test (Based on Section 4 defaults)
    test_params = {
        'n_estimators': [50, 100, 200, 400],    # Paper default: 200 
        'min_samples_leaf': [100, 200, 300, 500, 750, 1000], # Paper defaults: 300 
        'max_depth': [5, 10, 20, None]           # Paper default: None 
    }
    
    # Defaults
    base_config = {
        'n_estimators': 200,
        'min_samples_leaf': 300,
        'max_depth': None,
        'alpha': 0.05
    }
    
    # Outer loop for repetitions
    for rep in tqdm(range(n_repetitions), desc="Running Sensitivity Analysis Repetitions"):
        print(f"\n--- Starting Repetition {rep + 1}/{n_repetitions} ---")
        current_seed = 42 + rep
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        
        # 1. Regenerate simulation data for this repetition
        thetas_sim = prior(num_samples=B)
        model_thetas = thetas_sim.reshape(-1, 1) if thetas_sim.ndim == 1 else thetas_sim
        repeated_thetas = thetas_sim.repeat_interleave(repeats=N, dim=0)
        X_net = simulator(repeated_thetas).reshape(B, N * X_dim)
        
        # Compute the scores (lambdas) for calibration
        model_lambdas = score.compute(model_thetas.numpy(), X_net.numpy(), disable_tqdm=True)

        # 2. Iterate through parameters
        for param, values in test_params.items():
            for val in values:
                config = base_config.copy()
                config[param] = val
                
                start_time = time.time()
                
                # Fit and Calibrate TRUST++ MV (Majority Vote)
                trust_plus_object = ConformalLoforest(
                    LambdaScore, None, alpha=config['alpha'],
                    is_fitted=True, split_calib=False,
                )
                
                trust_plus_object.calibrate(
                    thetas_sim,
                    model_lambdas,
                    n_estimators=config['n_estimators'],
                    min_samples_leaf=config['min_samples_leaf'],
                    max_depth=config['max_depth'],
                    K=np.ceil(config['n_estimators'] / 2).astype(int),
                )
                
                # 3. Evaluate on the valid grid (using pre-simulated evaluation stats)
                model_eval = thetas_valid.reshape(-1, 1) if thetas_valid.ndim == 1 else thetas_valid
                loforest_cutoffs = trust_plus_object.compute_cutoffs(model_eval) 

                err_loforest = []
                for l, theta in enumerate(thetas_valid):
                    lookup_theta = theta if thetas_valid.ndim == 1 else tuple(theta)
                    stat = stat_dict[lookup_theta]
                    
                    # Compute coverage and deviation from nominal level
                    coverage = np.mean(stat <= loforest_cutoffs[l])
                    err_loforest.append(np.abs(coverage - (1 - config['alpha'])))
                
                mae = np.mean(err_loforest)
                runtime = time.time() - start_time
                
                all_runs_data.append({
                    'Parameter': param,
                    'Value': val,
                    'Repetition': rep,
                    'MAE': mae,
                    'Runtime (s)': runtime
                })

    # 4. Group by and aggregate results
    results_df = pd.DataFrame(all_runs_data)
    
    # We group by Parameter and Value to average across repetitions
    final_summary = results_df.groupby(['Parameter', 'Value']).agg({
        'MAE': ['mean', 'std'],
        'Runtime (s)': ['mean', 'std']
    }).reset_index()

    # Flatten the multi-index columns for cleaner output
    final_summary.columns = ['Parameter', 'Value', 'MAE_mean', 'MAE_std', 'Time_mean', 'Time_std']
    
    return final_summary

# Execute the analysis
summary_df = run_sensitivity_analysis(n_repetitions=10)

# Display results sorted by parameter
print("\n--- Sensitivity Analysis Summary (Averaged over 10 reps) ---")
print(summary_df.sort_values(['Parameter', 'Value']))

