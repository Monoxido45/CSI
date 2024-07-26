import torch
from torch.distributions.beta import Beta
import pickle
import pandas as pd
import numpy as np
import itertools

# statistics and posterior model
from CP2LFI.scores import WaldoScore, BFFScore, E_valueScore

# utils functions
from CP2LFI.utils import fit_post_model

# functions to make simulator or prior
from hypothesis.benchmark import sir, tractable, mg1, weinberg
import sbibm

# pip install sbibm
import os
from tqdm import tqdm

# general path
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
folder_path = "/results/LFI_objects/tune_data/"
# waldo_score.base_model.device("cpu")
# bff_score.base_model.device("cpu")
# e_value_score.base_model.device("cpu")
# waldo_score.base_model.model.device("cpu")


def generate_tuning_matrix(
    kind,
    n,
    B=750,
    n_lambda=350,
    log_transf=False,
):
    save_path = original_path + folder_path
    if not os.path.isdir(save_path):
        # creating kind and score directory if not existing
        os.makedirs(save_path)

    two_moons = False
    # importing simulator and prior for each kind of statistic
    if kind == "weinberg":
        simulator = weinberg.Simulator(default_beam_energy=40.0)
        prior = weinberg.Prior()
    elif kind == "sir":
        simulator = sir.Simulator()
        prior = sir.Prior()
    elif kind == "tractable":
        simulator = tractable.Simulator()
        prior = tractable.Prior()
    elif kind == "mg1":
        simulator = mg1.MG1Simulator()
        prior = mg1.Prior()
        log_transf = True
    elif kind == "two moons":
        task = sbibm.get_task(
            "two_moons"
        )  # See sbibm.get_available_tasks() for all tasks
        simulator = task.get_simulator()
        prior = task.get_prior()
        two_moons = True

    # computing and saving dictionary for each statistic
    if two_moons:
        theta_tune = prior(num_samples=B)
    else:
        theta_tune = prior.sample((B,))

    if theta_tune.ndim == 1:
        K_valid_thetas = theta_tune.reshape(-1, 1)
    else:
        K_valid_thetas = theta_tune

    waldo_tune_sample, bff_tune_sample, e_value_tune_sample = (
        np.zeros((B, n_lambda)),
        np.zeros((B, n_lambda)),
        np.zeros((B, n_lambda)),
    )

    # importing waldo from pickle file
    waldo_stat = pd.read_pickle(original_path + stats_path + f"{kind}_waldo_{n}.pickle")

    # importing BFF from pickle file
    bff_stat = pd.read_pickle(original_path + stats_path + f"{kind}_bff_{n}.pickle")

    # import e-value from pickle file
    e_value_stat = pd.read_pickle(
        original_path + stats_path + f"{kind}_e_value_{n}.pickle"
    )

    i = 0
    for theta in tqdm(theta_tune, desc="Simulating all tuning samples"):
        if theta_tune.ndim == 1:
            theta_repeated = (
                torch.tensor([theta])
                .reshape(1, -1)
                .repeat_interleave(repeats=n_lambda * n, dim=0)
            )
        else:
            theta_repeated = theta.reshape(1, -1).repeat_interleave(
                repeats=n_lambda * n, dim=0
            )

        X_net = simulator(theta_repeated)
        if log_transf:
            X_net = torch.log(X_net)
        X_dim = X_net.shape[1]
        X_net = X_net.reshape(n_lambda, n * X_dim)

        # waldo tuning samples
        waldo_tune_sample[i, :] = waldo_stat.compute(
            thetas=theta_repeated.numpy()[0:n_lambda, :],
            X=X_net.numpy(),
            disable_tqdm=True,
        )

        bff_tune_sample[i, :] = bff_stat.compute(
            thetas=theta_repeated.numpy()[0:n_lambda, :],
            X=X_net.numpy(),
            disable_tqdm=True,
        )

        e_value_tune_sample[i, :] = e_value_stat.compute(
            thetas=theta_repeated.numpy()[0:n_lambda, :],
            X=X_net.numpy(),
            disable_tqdm=True,
        )

        i += 1

        # Save checkpoint of the scores into pickle files
        pickle.dump(
            K_valid_thetas, open(save_path + f"{kind}_theta_tune_{n}.pickle", "wb")
        )
        pickle.dump(
            waldo_tune_sample, open(save_path + f"{kind}_waldo_tune_{n}.pickle", "wb")
        )
        pickle.dump(
            bff_tune_sample, open(save_path + f"{kind}_bff_tune_{n}.pickle", "wb")
        )
        pickle.dump(
            e_value_tune_sample,
            open(save_path + f"{kind}_e_value_tune_{n}.pickle", "wb"),
        )


n_list = [1, 5, 10, 20, 50]

if __name__ == "__main__":
    print("We will now save all evaluation grid elements")
    kind = input("Which kind of simulator would like to use? ")
    n_lambda = int(input("What is the size of each statistic vector? "))
    B = int(input("How much thetas do you want to simulate? "))
    print(f"Generating tuning samples for the {kind} problem")
    for n in n_list:
        print("Fitting for n = {}".format(n))
        generate_tuning_matrix(kind=kind, n=n, n_lambda=n_lambda, B=B)
