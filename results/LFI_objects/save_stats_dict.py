import torch
from torch.distributions.beta import Beta
import pickle
import pandas as pd
import numpy as np
import itertools

# functions to make simulator or prior
from hypothesis.benchmark import sir, tractable, mg1, weinberg
import sbibm
import os
from tqdm import tqdm
import io

# general path
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
folder_path = "/results/LFI_objects/stat_data/"


# for CPU usage
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def simulate_theta_grid_eval(
    kind,
    theta_grid,
    n,
    n_lambda=500,
    log_transf=False,
    completing=False,
    using_CPU=False,
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

    # importing all statistics estimators
    # importing all statistics
    if using_CPU:
        # importing waldo
        with open(original_path + stats_path + f"{kind}_waldo_{n}.pickle", "rb") as f:
            waldo_stat = CPU_Unpickler(f).load()

        # importing e-value
        with open(original_path + stats_path + f"{kind}_e_value_{n}.pickle", "rb") as f:
            e_value_stat = CPU_Unpickler(f).load()

        # importing bff
        with open(original_path + stats_path + f"{kind}_bff_{n}.pickle", "rb") as f:
            bff_stat = CPU_Unpickler(f).load()

        waldo_stat.base_model.device = torch.device("cpu")
        bff_stat.base_model.device = torch.device("cpu")
        e_value_stat.base_model.device = torch.device("cpu")
        waldo_stat.base_model.model.device = torch.device("cpu")
    else:
        # importing waldo
        waldo_stat = pd.read_pickle(
            original_path + stats_path + f"{kind}_waldo_{n}.pickle"
        )

        # importing bff
        bff_stat = pd.read_pickle(original_path + stats_path + f"{kind}_bff_{n}.pickle")

        # import e-value from pickle file
        e_value_stat = pd.read_pickle(
            original_path + stats_path + f"{kind}_e_value_{n}.pickle"
        )

    # computing and saving dictionary for each statistic
    if completing:
        waldo_dict = pd.read_pickle(
            original_path + folder_path + f"{kind}_waldo_eval_{n}.pickle"
        )
        bff_dict = pd.read_pickle(
            original_path + folder_path + f"{kind}_bff_eval_{n}.pickle"
        )
        e_value_dict = pd.read_pickle(
            original_path + folder_path + f"{kind}_e_value_eval_{n}.pickle"
        )
    else:
        waldo_dict, bff_dict, e_value_dict = {}, {}, {}

    if completing:
        if theta_grid.ndim == 1:
            new_theta_grid = theta_grid[len(waldo_dict) :]
        else:
            new_theta_grid = theta_grid[len(waldo_dict) :, :]

        for theta in tqdm(new_theta_grid, desc="Creating stat grid"):
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
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                bff_dict[theta] = bff_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                e_value_dict[theta] = e_value_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
            # saving for other problems
            else:
                waldo_dict[tuple(theta)] = waldo_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                bff_dict[tuple(theta)] = bff_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                e_value_dict[tuple(theta)] = e_value_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )

            # Save checkpoint of the scores into pickle files
            pickle.dump(
                waldo_dict, open(save_path + f"{kind}_waldo_eval_{n}.pickle", "wb")
            )
            pickle.dump(bff_dict, open(save_path + f"{kind}_bff_eval_{n}.pickle", "wb"))
            pickle.dump(
                e_value_dict, open(save_path + f"{kind}_e_value_eval_{n}.pickle", "wb")
            )

    else:
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
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                bff_dict[theta] = bff_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                e_value_dict[theta] = e_value_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
            # saving for other problems
            else:
                waldo_dict[tuple(theta)] = waldo_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                bff_dict[tuple(theta)] = bff_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )
                e_value_dict[tuple(theta)] = e_value_stat.compute(
                    theta_repeated.numpy()[0:n_lambda, :],
                    X_net.numpy(),
                    disable_tqdm=True,
                )

            # Save checkpoint of the scores into pickle files
            pickle.dump(
                waldo_dict, open(save_path + f"{kind}_waldo_eval_{n}.pickle", "wb")
            )
            pickle.dump(bff_dict, open(save_path + f"{kind}_bff_eval_{n}.pickle", "wb"))
            pickle.dump(
                e_value_dict, open(save_path + f"{kind}_e_value_eval_{n}.pickle", "wb")
            )


n_list = np.array([1, 5, 10, 20, 50])

if __name__ == "__main__":
    print("We will now save all evaluation grid elements")
    kind = input("Which kind of simulator would like to use? ")
    n_lambda = int(input("What is the size of each statistic vector? "))
    use_CPU = (
        input("Do you want to use a CPU instead of a GPU to compute the statistics? ")
        == "yes"
    )
    n_unique = (
        input("Do you whish to compute all evaluation grid for a single n? ") == "yes"
    )
    completing = input("Are you completing any process? ") == "yes"
    if completing:
        n_complete = int(input("In which n did you stopped? "))

    print(f"Generating evaluation grid for the {kind} problem")
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

    if completing:
        if n_unique:
            print("Fitting for n = {}".format(n_complete))
            simulate_theta_grid_eval(
                kind=kind,
                theta_grid=thetas_valid,
                n=n_complete,
                n_lambda=n_lambda,
                completing=True,
                using_CPU=use_CPU,
            )
        else:
            # updating n_list if n not unique
            n_list = n_list[np.where(n_list >= n_complete)]
            for n in n_list:
                print("Fitting for n = {}".format(n))
                complete_now = n == n_list[0]
                simulate_theta_grid_eval(
                    kind=kind,
                    theta_grid=thetas_valid,
                    n=n,
                    n_lambda=n_lambda,
                    completing=complete_now,
                    using_CPU=use_CPU,
                )
    else:
        if n_unique:
            n = int(input("Which n do you want to fix? "))
            print("Fitting for n = {}".format(n))
            simulate_theta_grid_eval(
                kind=kind,
                theta_grid=thetas_valid,
                n=n,
                n_lambda=n_lambda,
                using_CPU=use_CPU,
            )
        else:
            print("Computing for a list of n")
            for n in n_list:
                print("Fitting for n = {}".format(n))
                simulate_theta_grid_eval(
                    kind=kind,
                    theta_grid=thetas_valid,
                    n=n,
                    n_lambda=n_lambda,
                    using_CPU=use_CPU,
                )
