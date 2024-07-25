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
import os
from tqdm import tqdm

# general path
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
folder_path = "/results/LFI_objects/stat_data/"


def simulate_theta_grid_eval(
    kind,
    theta_grid,
    n,
    n_lambda=500,
    log_transf=False,
):
    two_moons = False

    save_path = original_path + folder_path
    if not os.path.isdir(save_path):
        # creating kind and score directory if not existing
        os.makedirs(save_path)

    # importing simulator and prior for each kind of statistic
    if kind == "weinberg":
        simulator = weinberg.Simulator(default_beam_energy=40.0)
    elif kind == "sir":
        simulator = sir.Simulator()
    elif kind == "tractable":
        simulator = tractable.Simulator()
    elif kind == "mg1":
        simulator = mg1.MG1Simulator()
        log_transf = True
    elif kind == "two moons":
        task = sbibm.get_task(
            "two_moons"
        )  # See sbibm.get_available_tasks() for all tasks
        simulator = task.get_simulator()

    # computing and saving dictionary for each statistic
    waldo_dict, bff_dict, e_value_dict = {}, {}, {}

    # importing waldo from pickle file
    waldo_stat = pd.read_pickle(original_path + stats_path + f"{kind}_waldo_{n}.pickle")
    bff_stat = pd.read_pickle(original_path + stats_path + f"{kind}_bff_{n}.pickle")
    e_value_stat = pd.read_pickle(
        original_path + stats_path + f"{kind}_e_value_{n}.pickle"
    )

    for theta in tqdm(theta_grid, desc="Creating stat grid"):
        if theta_grid.ndim == 1:
            theta_repeated = (
                torch.tensor([theta])
                .reshape(1, -1)
                .repeat_interleave(repeats=n_lambda * n, dim=0)
            )
        else:
            theta_repeated = torch.tensor([theta]).repeat_interleave(
                repeats=n_lambda * n, dim=0
            )

        # simulating lambdas for testing
        X_net = simulator(theta_repeated)
        if log_transf:
            X_net = torch.log(X_net)
        X_dim = X_net.shape[1]
        X_net = X_net.reshape(n_lambda, n * X_dim)

        # saving for weinberg
        if theta.shape == ():
            waldo_dict[theta] = waldo_stat.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )
            bff_dict[theta] = bff_stat.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )
            e_value_dict[theta] = e_value_stat.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )
        # saving for other problems
        else:
            waldo_dict[tuple(theta)] = waldo_stat.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )
            bff_dict[tuple(theta)] = bff_stat.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )
            e_value_dict[tuple(theta)] = e_value_stat.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )

        # Save checkpoint of the scores into pickle files
        pickle.dump(waldo_dict, open(save_path + f"{kind}_waldo_eval_{n}.pickle", "wb"))
        pickle.dump(bff_dict, open(save_path + f"{kind}_bff_eval_{n}.pickle", "wb"))
        pickle.dump(
            e_value_dict, open(save_path + f"{kind}_e_value_eval_{n}.pickle", "wb")
        )


n_list = [1, 5, 10, 20, 50]

if __name__ == "__main__":
    print("We will now save all evaluation grid elements")
    kind = input("Which kind of simulator would like to use? ")
    n_lambda = int(input("What is the size of each statistic vector from the grid? "))
    print(f"Generating evaluation grid for the {kind} problem")
    for n in n_list:
        print("Fitting for n = {}".format(n))

        if kind == "tractable":
            # obtaining evaluation grid
            n_par = 5
            pars = np.linspace(-2.9, 2.9, n_par)
            pars[pars == 0] = 0.01
            thetas_valid = np.c_[list(itertools.product(pars, pars, pars, pars, pars))]

        elif kind == "weinberg":
            n_out = 300
            thetas_valid = np.linspace(0.51, 1.49, n_out)

        elif kind == "mg1":
            n_par = 8
            pars_1 = np.linspace(0.1, 9.9, n_par)
            pars_2 = np.linspace(0.01, 1 / 3 - 0.01, n_par)
            thetas_valid = np.c_[list(itertools.product(pars_1, pars_1, pars_2))]

        elif kind == "sir":
            n_par = 22
            pars_1 = np.linspace(0.01, 0.49, n_par)
            thetas_valid = np.c_[list(itertools.product(pars_1, pars_1))]

        elif kind == "two moons":
            n_par = 20
            pars_1 = np.linspace(-0.99, 0.99, n_par)
            thetas_valid = np.c_[list(itertools.product(pars_1, pars_1))]

        simulate_theta_grid_eval(
            kind=kind, theta_grid=thetas_valid, n=n, n_lambda=n_lambda
        )
