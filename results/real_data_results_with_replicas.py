import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
from CP2LFI.utils import obtain_quantiles

# general path
original_path = os.getcwd()


def compute_MAE_N_B(
    kind,
    score,
    theta_grid_eval,
    simulator,
    prior,
    folder_path="/results/LFI_real_results/",
    n_replica=30,
    N=5,
    B=10000,
    alpha=0.05,
    min_samples_leaf=300,
    n_estimators=200,
    K=50,
    B_valid=500,
    N_lambda=250,
    K_grid=np.concatenate((np.array([0]), np.arange(15, 95, 5))),
    naive_n=500,
    disable_tqdm=True,
    seed=45,
    n_lambda=300,
    log_transf=False,
    split_calib=False,
    using_beta=False,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    mae_array = np.zeros((n_replica, 5))
    K_array = np.zeros(n_replica)

    # first column: TRUST
    # second column: TRUST++ majority votes
    # third column: TRUST++ tuned
    # fourth column: boosting
    # fifth column: monte-carlo

    # creating folder if not existing
    var_path = kind + "/" + score
    save_path = original_path + folder_path + var_path

    if not os.path.isdir(original_path + folder_path + var_path):
        # creating kind and score directory if not existing
        os.makedirs(save_path)

    for i in range(n_replica):
        quantiles_dict, K_tuned = obtain_quantiles(
            kind=kind,
            score=score,
            theta_grid_eval=theta_grid_eval,
            simulator=simulator,
            prior=prior,
            N=N,
            B=B,
            alpha=alpha,
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
            K=K,
            disable_tqdm=disable_tqdm,
            B_valid=B_valid,
            N_lambda=N_lambda,
            K_grid=K_grid,
            naive_n=naive_n,
            log_transf=log_transf,
            split_calib=split_calib,
            using_beta=using_beta,
        )
        mae_list, se_list, methods_list, N_list, B_list = [], [], [], [], []
        err_data = np.zeros((theta_grid_eval.shape[0], 5))
        l = 0
        for theta in tqdm(theta_grid_eval, desc="Evaluating coverage in this setting"):
            if theta_grid_eval.ndim == 1:
                theta_repeated = (
                    torch.tensor([theta])
                    .reshape(1, -1)
                    .repeat_interleave(repeats=n_lambda * N, dim=0)
                )
            else:
                theta_repeated = torch.tensor([theta]).repeat_interleave(
                    repeats=n_lambda * N, dim=0
                )

            # simulating lambdas for testing
            X_net = simulator(theta_repeated)
            if log_transf:
                X_net = torch.log(X_net)
            X_dim = X_net.shape[1]
            X_net = X_net.reshape(n_lambda, N * X_dim)

            stat = score.compute(
                theta_repeated.numpy()[0:n_lambda, :], X_net.numpy(), disable_tqdm=True
            )

            # comparing coverage of methods
            locart_cover = np.mean(stat <= quantiles_dict["locart"][l])
            loforest_cover = np.mean(stat <= quantiles_dict["loforest_fixed"][l])
            loforest_tuned_cover = np.mean(stat <= quantiles_dict["loforest_tuned"][l])
            boosting_cover = np.mean(stat <= quantiles_dict["boosting"][l])
            naive_cover = np.mean(stat <= quantiles_dict["naive"][l])

            # appending the errors
            err_locart = np.abs(locart_cover - (1 - alpha))
            err_loforest = np.abs(loforest_cover - (1 - alpha))
            err_loforest_tuned = np.abs(loforest_tuned_cover - (1 - alpha))
            err_boosting = np.abs(boosting_cover - (1 - alpha))
            err_naive = np.abs(naive_cover - (1 - alpha))

            # saving in numpy array
            err_data[l, :] = np.array(
                [err_locart, err_loforest, err_loforest_tuned, err_boosting, err_naive]
            )
            l += 1

        mae_array[i, :] = np.mean(err_data, axis=0)
        K_array[i] = K_tuned

        # saving the K column
        stats_data = pd.DataFrame(
            data=mae_array,
            columns=["TRUST", "TRUST++ MV", "TRUST++ tuned", "boosting", "MC"],
        )
        stats_data["K_tuned"] = K_array

        # saving checkpoint
        print("Saving data checkpoint on iteration {}".format(i + 1))
        stats_data.to_csv(save_path + "MAE_data.csv")

    stats_data = pd.DataFrame(
        data=mae_array,
        columns=["TRUST", "TRUST++ MV", "TRUST++ tuned", "boosting", "MC"],
    )
    return stats_data


# Note: The inner loop where theta is processed, and the obtain_quantiles function call
# need to be adapted based on the actual implementation details of obtain_quantiles and t
n_list = [1, 5, 10, 20]
B_list = [10000, 15000, 20000, 30000]

if __name__ == "__main__":
    print("We will now compute the MAE across several replicas for each LFI problem")
    kind = input("Which kind of simulator would like to use? ")
    score = input(
        "Which kind of statistics would you like to use to derive confidence regions? "
    )
    it = int(input("How many iterations? "))
    print(f"Starting experiment for {kind} simulator with {score} statistic")

    # creating

    for n in n_list:
        # TODO: open already fitted statistic with posterior estimator to avoid too much time fitting all
        # also, save the tuning samples for TRUST++ in each case
        for B in B_list:
            # fill in with what remains.
            compute_MAE_N_B()