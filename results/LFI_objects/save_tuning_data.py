import torch
from torch.distributions.beta import Beta
import pickle
import pandas as pd
import numpy as np

# functions to make simulator or prior
from hypothesis.benchmark import sir, tractable, mg1, weinberg
import sbibm

# pip install sbibm
import os
from tqdm import tqdm
import io

# general path
original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
folder_path = "/results/LFI_objects/tune_data/"


# for CPU usage
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def generate_tuning_matrix(
    kind,
    n,
    B=750,
    n_lambda=350,
    log_transf=False,
    completing=False,
    using_CPU=False,
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

    if completing:
        # importing theta_tune
        K_valid_thetas = pd.read_pickle(
            original_path + folder_path + f"{kind}_theta_tune_{n}.pickle"
        )

        # reading all tune sample
        waldo_tune_sample = pd.read_pickle(
            original_path + folder_path + f"{kind}_waldo_tune_{n}.pickle"
        )
        bff_tune_sample = pd.read_pickle(
            original_path + folder_path + f"{kind}_bff_tune_{n}.pickle"
        )
        e_value_tune_sample = pd.read_pickle(
            original_path + folder_path + f"{kind}_e_value_tune_{n}.pickle"
        )

        # returning index from tune sample
        idxs = np.where(np.all(waldo_tune_sample == 0, axis=1) == True)
        new_theta_tune = K_valid_thetas[idxs]
        i = idxs[0][0]

        for theta in tqdm(new_theta_tune, desc="Simulating remaining tuning samples"):
            if new_theta_tune.ndim == 1:
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

            # Save checkpoint of the scores into pickle files
            pickle.dump(
                K_valid_thetas,
                open(save_path + f"{kind}_theta_tune_{n}.pickle", "wb"),
            )

            pickle.dump(
                waldo_tune_sample,
                open(save_path + f"{kind}_waldo_tune_{n}.pickle", "wb"),
            )
            pickle.dump(
                bff_tune_sample, open(save_path + f"{kind}_bff_tune_{n}.pickle", "wb")
            )
            pickle.dump(
                e_value_tune_sample,
                open(save_path + f"{kind}_e_value_tune_{n}.pickle", "wb"),
            )

            i += 1
    else:
        print("a")
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
                K_valid_thetas,
                open(save_path + f"{kind}_theta_tune_{n}.pickle", "wb"),
            )

            pickle.dump(
                waldo_tune_sample,
                open(save_path + f"{kind}_waldo_tune_{n}.pickle", "wb"),
            )
            pickle.dump(
                bff_tune_sample,
                open(save_path + f"{kind}_bff_tune_{n}.pickle", "wb"),
            )
            pickle.dump(
                e_value_tune_sample,
                open(save_path + f"{kind}_e_value_tune_{n}.pickle", "wb"),
            )


n_list = np.array([1, 5, 10, 20, 50])

if __name__ == "__main__":
    print("We will now save all evaluation grid elements")
    kind = input("Which kind of simulator would like to use? ")
    n_lambda = int(input("What is the size of each statistic vector? "))
    B = int(input("How much thetas do you want to simulate? "))
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
        print(f"Generating tuning samples for the {kind} problem")
        if n_unique:
            print("Fitting for n = {}".format(n_complete))
            generate_tuning_matrix(
                kind=kind,
                n=n_complete,
                n_lambda=n_lambda,
                B=B,
                completing=True,
                using_CPU=use_CPU,
            )
        else:
            n_list = n_list[np.where(n_list >= n_complete)]
            for n in n_list:
                complete_now = n == n_list[0]
                print("Fitting for n = {}".format(n))
                generate_tuning_matrix(
                    kind=kind,
                    n=n,
                    n_lambda=n_lambda,
                    B=B,
                    completing=complete_now,
                    using_CPU=use_CPU,
                )
    else:
        print(f"Generating tuning samples for the {kind} problem")
        if n_unique:
            n = int(input("Which n do you want to fix? "))
            print("Fitting for n = {}".format(n))
            generate_tuning_matrix(
                kind=kind,
                n=n,
                n_lambda=n_lambda,
                B=B,
                completing=False,
                using_CPU=use_CPU,
            )
        else:
            starting_another = input(f"Are you starting from another n") == "yes"
            if starting_another:
                n_new = int(input("Which n do you want to fix? "))
                print("Computing for a new list of n")
                n_list = n_list[np.where(n_list >= n_new)]
                for n in n_list:
                    print("Fitting for n = {}".format(n))
                    generate_tuning_matrix(
                        kind=kind,
                        n=n,
                        n_lambda=n_lambda,
                        B=B,
                        using_CPU=use_CPU,
                    )
            else:
                for n in n_list:
                    print("Fitting for n = {}".format(n))
                    generate_tuning_matrix(
                        kind=kind,
                        n=n,
                        n_lambda=n_lambda,
                        B=B,
                        using_CPU=use_CPU,
                    )
