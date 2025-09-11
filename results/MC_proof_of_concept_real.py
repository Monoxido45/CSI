import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from CSI.utils import CPU_Unpickler, naive, predict_naive_quantile
from hypothesis.benchmark import tractable, sir
import os
import torch
import pandas as pd
import pickle


# general path
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
stats_eval_path = "/results/LFI_objects/stat_data/"


def real_naive_proof_of_concept(
    using_cpu=False,
    N=5,
    kind="tractable",
    score_list=["bff", "waldo"],
    B_values=[15000, 20000],
):
    alpha = 0.05
    log_transf = False
    naive_n = 500
    results = {}

    if kind == "tractable":
        simulator = tractable.Simulator()

        # defining validation grid
        n_par = 5
        pars = np.linspace(-2.9, 2.9, n_par)
        pars[pars == 0] = 0.01
        thetas_valid = np.c_[list(itertools.product(pars, pars, pars, pars, pars))]
    elif kind == "sir":
        simulator = sir.Simulator()

        # defining validation grid
        n_par = 22
        pars_1 = np.linspace(0.01, 0.49, n_par)
        thetas_valid = np.c_[list(itertools.product(pars_1, pars_1))]

    for score_name in score_list:
        print(f"\n{'='*60}")
        print(f"Testing {kind} model")
        print(f"{'='*60}")

        model_results = {
            "B_values": B_values,
            "coverage_rates": [],
            "predicted_cutoffs": {},
            "cutoff_dicts": {},
        }

        # importing stats
        if using_cpu:
            with open(
                original_path + stats_path + f"{kind}_{score_name}_{N}.pickle", "rb"
            ) as f:
                score = CPU_Unpickler(f).load()
        else:
            score = pd.read_pickle(
                original_path + stats_path + f"{kind}_{score_name}_{N}.pickle"
            )

        score.base_model.device = torch.device("cpu")
        score.base_model.model.device = torch.device("cpu")
        score.base_model.model.to(torch.device("cpu"))

        # importing stat dictionary
        if using_cpu:
            with open(
                original_path
                + stats_eval_path
                + f"{kind}_{score_name}_eval_{N}.pickle",
                "rb",
            ) as f:
                stat_dict = CPU_Unpickler(f).load()
        else:
            stat_dict = pd.read_pickle(
                original_path + stats_eval_path + f"{kind}_{score_name}_eval_{N}.pickle"
            )

        for B in tqdm(B_values, desc=f"Testing different B values for {kind}"):
            naive_quantiles = naive(
                kind=kind,
                simulator=simulator,
                score=score,
                alpha=alpha,
                B=B,
                N=N,
                naive_n=naive_n,
                log_transf=log_transf,
                disable_tqdm=True,
                disable_tqdm_naive=False,
            )

            naive_list = predict_naive_quantile(
                kind,
                thetas_valid,
                naive_quantiles,
            )

            l = 0
            mae_naive = np.zeros(thetas_valid.shape[0])
            for theta in thetas_valid:
                if thetas_valid.ndim == 1:
                    stat = stat_dict[theta]
                else:
                    theta = tuple(theta)
                    stat = stat_dict[theta]

                naive_cover = np.mean(stat <= naive_list[l])
                mae_naive[l] = np.abs(naive_cover - (1 - alpha))
                l += 1

            model_results["coverage_rates"].append(np.mean(mae_naive))
            model_results["cutoff_dicts"][B] = naive_quantiles
            model_results["predicted_cutoffs"][B] = naive_list

            print(f"Coverage MAE: {np.mean(mae_naive):.3f}")

        results[score_name] = model_results

    return results


# obtaining also for N = 20
# first, evaluating specific results and cutoffs
all_res = real_naive_proof_of_concept(score_list=["bff"])

with open("all_res.pickle", "wb") as f:
    pickle.dump(all_res, f)

all_res_sir = real_naive_proof_of_concept(
    kind="sir",
    score_list=["bff"],
    B_values=[10000, 15000],
)

with open("all_res_sir.pickle", "wb") as f:
    pickle.dump(all_res_sir, f)
