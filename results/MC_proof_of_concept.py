import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from CSI.simulations import Simulations, naive, predict_naive_quantile
import itertools


def naive_proof_of_concept():
    """
    Proof of concept for the naive method using BFF statistic under different simulation budgets B
    for tractable and lognormal models
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Define simulation parameters
    alpha = 0.05
    N = 20  # Number of observations (reduced for faster computation)
    naive_n = 500  # Number of samples for naive method (reduced)

    # Different simulation budgets to test
    B_values = [1000, 5000, 10000]

    # Test both models
    kind_model = "lognormal"
    stats = ["BFF", "FBST"]

    results = {}

    for sel_stat in stats:
        print(f"\n{'='*60}")
        print(f"Testing {kind_model} model")
        print(f"{'='*60}")

        n_out = 50
        a_s = np.linspace(-2.4999, 2.4999, n_out)
        b_s = np.linspace(0.15001, 1.2499, n_out)
        thetas = np.c_[list(itertools.product(a_s, b_s))]

        model_results = {
            "B_values": B_values,
            "coverage_rates": [],
            "predicted_cutoffs": {},
            "cutoff_dicts": {},
            "oracle_cutoffs": {},
        }

        for B in tqdm(B_values, desc=f"Testing different B values for {kind_model}"):
            print(f"\nTesting with simulation budget B = {B}")

            # Create simulation object
            sim_obj = Simulations(rng=rng, kind_model=kind_model)

            # Run naive method with current B
            naive_quantiles = naive(
                stat=sel_stat,
                kind_model=kind_model,
                alpha=alpha,
                rng=rng,
                B=B,
                N=N,
                naive_n=naive_n,
            )

            # Predict quantiles for our test theta
            naive_list = predict_naive_quantile(kind_model, thetas, naive_quantiles)

            # Generate test statistics to evaluate coverage
            n_test = 1000  # Number of test statistics to generate
            l = 0
            mae_naive = np.zeros(thetas.shape[0])
            oracle_cutoff = np.zeros(thetas.shape[0])
            for theta in thetas:
                # simulating lambdas for testing
                if sel_stat == "BFF":
                    stat = sim_obj.BFF_sim_lambda(theta=theta, B=n_test, N=N)
                else:
                    stat = sim_obj.FBST_sim_lambda(theta=theta, B=n_test, N=N)

                # computing oracle cutoff also
                oracle_cutoff[l] = np.quantile(stat, 1 - alpha)

                naive_cover = np.mean(stat <= naive_list[l])
                mae_naive[l] = np.abs(naive_cover - (1 - alpha))

                l += 1

            model_results["coverage_rates"].append(np.mean(mae_naive))
            model_results["cutoff_dicts"][B] = naive_quantiles
            model_results["predicted_cutoffs"][B] = naive_list
            model_results["oracle_cutoffs"][B] = oracle_cutoff

            print(f"Coverage MAE: {np.mean(mae_naive):.3f}")

        results[sel_stat] = model_results

    return results


# first, evaluating specific results and cutoffs
all_res = naive_proof_of_concept()

# now, visualizing what is happening for two different budgets in lognormal model using all_res
all_res["BFF"]["cutoff_dicts"][1000]
all_res["BFF"]["cutoff_dicts"][5000]

# plotting how the cutoff is behaving in grid
# Extract cutoff_dicts for BFF statistic
cutoff_dicts = all_res["BFF"]["cutoff_dicts"]

quantile_dict = all_res["BFF"]["cutoff_dicts"][5000]
thetas_values = np.array(list(quantile_dict.keys()))
theta = np.array([1.0, 0.5])

distances = np.linalg.norm(thetas_values - theta, axis=1)
idx = thetas_values[np.argmin(distances)]

all_cutoff_values = [value for d in cutoff_dicts.values() for value in d.values()]
vmin = min(all_cutoff_values)
vmax = max(all_cutoff_values)

fig, axes = plt.subplots(1, len(cutoff_dicts), figsize=(17, 5), sharey=True)

for idx, (B, cutoff_dict) in enumerate(cutoff_dicts.items()):
    thetas = np.array(list(cutoff_dict.keys()))
    cutoffs = np.array(list(cutoff_dict.values()))

    # 3. Use vmin and vmax to standardize the color scale in each plot
    sc = axes[idx].scatter(
        thetas[:, 0], thetas[:, 1], c=cutoffs, cmap="viridis", vmin=vmin, vmax=vmax
    )

    axes[idx].set_title(f"BFF Cutoff, B={B}")
    axes[idx].set_xlabel(r"$\theta_1$")
    if idx == 0:
        axes[idx].set_ylabel(r"$\theta_2$")

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), label="Cutoff Value", pad=0.02)
plt.tight_layout(rect=[0, 0, 0.98, 1])
plt.show()


all_res["BFF"]["coverage_rates"]
