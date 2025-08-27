import torch

# loforest and locart functions
from CSI.loforest import ConformalLoforest, tune_loforest_LFI
from clover import LocartSplit
from CSI.scores import LambdaScore
from CSI.utils import naive, predict_naive_quantile

# quantile regression
from sklearn.ensemble import HistGradientBoostingRegressor

# plotting and numpy
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd

# utils functions
from CSI.utils import obtain_quantiles
import os

# package to simulate from two moons
import sbibm
import pickle

original_path = os.getcwd()
stats_path = "/results/LFI_objects/data/"
tune_path = "/results/LFI_objects/tune_data/"
stats_eval_path = "/results/LFI_objects/stat_data/"


# Obtaining confidence regions, first for two moons
# selecting two moons task
task = sbibm.get_task("two_moons")  # See sbibm.get_available_tasks() for all tasks
simulator = task.get_simulator()
prior = task.get_prior()

# Confidence region for two moons
# parameters of simulation: n = 5 and B = 10k
n = 20
B = 1.5e4
alpha = 0.05
naive_n = 500
kind, score_name = "two moons", "e_value"
torch.manual_seed(125)
torch.cuda.manual_seed(125)

# reading score
score = pd.read_pickle(original_path + stats_path + f"{kind}_{score_name}_{n}.pickle")
# training all models and tuning TRUST++
# simulating from prior
thetas_sim = prior(num_samples=int(B))

if thetas_sim.ndim == 1:
    model_thetas = thetas_sim.reshape(-1, 1)
else:
    model_thetas = thetas_sim

repeated_thetas = thetas_sim.repeat_interleave(repeats=n, dim=0)
X_net = simulator(repeated_thetas)
X_dim = X_net.shape[1]
X_net = X_net.reshape(int(B), n * X_dim)

# fitting models that use simulations
model_lambdas = score.compute(model_thetas.numpy(), X_net.numpy(), disable_tqdm=False)

locart_object = LocartSplit(
    LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
)
locart_quantiles = locart_object.calib(
    model_thetas.numpy(),
    model_lambdas,
    min_samples_leaf=300,
)

# loforest quantiles
loforest_object = ConformalLoforest(
    LambdaScore, None, alpha=alpha, is_fitted=True, split_calib=False
)
loforest_object.calibrate(
    model_thetas.numpy(),
    model_lambdas,
    min_samples_leaf=300,
    n_estimators=200,
    K=100,
)

# boosting quantiles
boosting_model = HistGradientBoostingRegressor(
    loss="quantile",
    max_iter=100,
    max_depth=3,
    quantile=1 - alpha,
    random_state=105,
    n_iter_no_change=15,
    early_stopping=True,
)
boosting_model.fit(model_thetas.numpy(), model_lambdas)

# fitting monte-carlo
naive_quantiles = naive(
    kind=kind,
    simulator=simulator,
    score=score,
    alpha=alpha,
    B=B,
    N=n,
    naive_n=naive_n,
    disable_tqdm=True,
    disable_tqdm_naive=False,
)

# importing tuning samples
# first, importing theta_tune object
theta_tune = pd.read_pickle(original_path + tune_path + f"{kind}_theta_tune_{n}.pickle")

# then, importing lambda_tune object
lambda_tune = pd.read_pickle(
    original_path + tune_path + f"{kind}_{score_name}_tune_{n}.pickle"
)

K_grid = np.concatenate((np.array([0]), np.arange(15, 105, 5)))

if theta_tune.ndim == 1:
    K_valid_thetas = theta_tune.reshape(-1, 1)
else:
    K_valid_thetas = theta_tune

# fitting tuned loforest
K_loforest = tune_loforest_LFI(
    loforest_object,
    theta_data=K_valid_thetas.numpy(),
    lambda_data=lambda_tune,
    K_grid=K_grid,
)


# obtaining intervals for specific sample
torch.manual_seed(75)
torch.cuda.manual_seed(75)
n_par = 75
pars_1 = np.linspace(0.14, 0.55, n_par)
pars_2 = np.linspace(-0.55, -0.14, n_par)
theta_true = np.array([0.5, -0.25])
theta_grid = np.c_[list(itertools.product(pars_1, pars_2))]

# computing cutoffs
if theta_grid.ndim == 1:
    model_eval = theta_grid.reshape(-1, 1)
else:
    model_eval = theta_grid

idxs = locart_object.cart.apply(model_eval)
list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]
locart_cutoffs = np.array(list_locart_quantiles)

# loforest and CI for loforest
loforest_cutoffs, lower_trust_plus, upper_trust_plus = loforest_object.compute_cutoffs(
    model_eval, compute_CI=True
)

# tuned loforest
loforest_tuned_cutoffs, lower_tuned_trust_plus, upper_tuned_trust_plus = (
    loforest_object.compute_cutoffs(model_eval, compute_CI=True, K=K_loforest)
)

boosting_quantiles = boosting_model.predict(model_eval)
naive_list = predict_naive_quantile(kind, theta_grid, naive_quantiles)

# evaluating regions according to sample
repeated_true_thetas = (
    torch.from_numpy(theta_true).reshape(1, 2).repeat_interleave(repeats=n, dim=0)
)
X_sample = simulator(repeated_true_thetas)
X_sample = X_sample.reshape(1, n * X_dim)

# repeating the sample for all theta grid and computing stats
X_rep = X_sample.repeat_interleave(repeats=int(n_par**2), dim=0)

# computing e-value
lambdas_eval = score.compute(
    model_eval,
    X_rep.numpy(),
    disable_tqdm=False,
    N=500,
)

######### if oracle is not computed
# computing oracle cutoffs
theta_oracle = np.repeat(theta_grid, 100 * n, axis=0)
X_oracle = simulator(theta_oracle)
X_oracle = X_oracle.reshape(100 * theta_grid.shape[0], n * X_dim)
theta_oracle = np.repeat(theta_grid, 100, axis=0)

lambdas_oracle = score.compute(
    theta_oracle,
    X_oracle.numpy(),
    disable_tqdm=False,
    N=750,
)

oracle_cutoffs = np.zeros(theta_grid.shape[0])
j, k = 0, 100
alpha = 0.05
for i in range(theta_grid.shape[0]):
    oracle_cutoffs[i] = np.quantile(lambdas_oracle[j:k], q=1 - alpha)
    j += 100
    k += 100

# Save the oracle cutoffs into a pickle file
with open(original_path + "/results/" + f"{kind}_oracle_cutoffs_{n}.pickle", "wb") as f:
    pickle.dump(oracle_cutoffs, f)

############ If oracle is computed
# Load the oracle cutoffs from the pickle file
with open(original_path + "/results/" + f"{kind}_oracle_cutoffs_{n}.pickle", "rb") as f:
    oracle_cutoffs = pickle.load(f)


# getting confidence regions
TRUST_plus_MV_filter = np.where(lambdas_eval <= loforest_cutoffs)
TRUST_plus_MV_conf = theta_grid[TRUST_plus_MV_filter]

TRUST_filter = np.where(lambdas_eval <= locart_cutoffs)
TRUST_conf = theta_grid[TRUST_filter]

naive_filter = np.where(lambdas_eval <= np.array(naive_list))
naive_conf = theta_grid[naive_filter]

boosting_filter = np.where(lambdas_eval <= boosting_quantiles)
boosting_conf = theta_grid[boosting_filter]

# getting uncertainty region
accept_filter = np.where(lambdas_eval <= lower_trust_plus)
rej_filter = np.where(lambdas_eval >= upper_trust_plus)
agnostic_filter = np.where(
    (lambdas_eval > lower_trust_plus) & (lambdas_eval < upper_trust_plus)
)

oracle_filter = np.where(lambdas_eval <= oracle_cutoffs)
oracle_conf = theta_grid[oracle_filter]

# Create a grid for plotting
grid_y, grid_x = np.meshgrid(pars_2, pars_1)

# Initialize a grid for each method
TRUST_grid = np.zeros(n_par**2)
boosting_grid = np.zeros(n_par**2)
naive_grid = np.zeros(n_par**2)
TRUST_plus_MV_grid = np.zeros(n_par**2)
TRUST_plus_uncertainty = np.zeros(n_par**2)
oracle_grid = np.zeros(n_par**2)

# Fill the grids based on the confidence regions
TRUST_grid[TRUST_filter] = 1
TRUST_plus_MV_grid[TRUST_plus_MV_filter] = 1
boosting_grid[boosting_filter] = 1
naive_grid[naive_filter] = 1
oracle_grid[oracle_filter] = 1
TRUST_plus_uncertainty[accept_filter] = 1
TRUST_plus_uncertainty[agnostic_filter] = 1 / 2

TRUST_grid = TRUST_grid.reshape(n_par, n_par)
TRUST_plus_MV_grid = TRUST_plus_MV_grid.reshape(n_par, n_par)
oracle_grid = oracle_grid.reshape(n_par, n_par)
boosting_grid = boosting_grid.reshape(n_par, n_par)
naive_grid = naive_grid.reshape(n_par, n_par)
TRUST_plus_uncertainty = TRUST_plus_uncertainty.reshape(n_par, n_par)


# Plotting the confidence regions as colored grids
from matplotlib import colors as c
import seaborn as sns

sns.set(style="ticks", font_scale=2)
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
plt.rcParams.update({"font.size": 14})

cMap1 = c.ListedColormap(["white", "darkred"])
cMap2 = c.ListedColormap(["white", "salmon", "maroon"])
cMap3 = c.ListedColormap(["white", "darkgray"])
cMap4 = c.ListedColormap(["white", "darkgreen"])
cMap5 = c.ListedColormap(["white", "darkorange"])

# TRUST
axs[0, 0].pcolormesh(grid_x, grid_y, TRUST_plus_MV_grid, alpha=0.675, cmap=cMap1)
axs[0, 0].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=15
)
axs[0, 0].set_title("Trust++ tuned")
axs[0, 0].set_ylabel("")
axs[0, 0].set_xlabel("")

# TRUST++ Tuned
axs[0, 1].pcolormesh(grid_x, grid_y, TRUST_plus_uncertainty, alpha=0.675, cmap=cMap2)
axs[0, 1].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=15
)
axs[0, 1].set_title("Trust ++ uncertainty")
axs[0, 1].set_ylabel("")
axs[0, 1].set_xlabel("")

# Oracle
axs[0, 2].pcolormesh(grid_x, grid_y, oracle_grid, alpha=0.675, cmap=cMap3)
axs[0, 2].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=15
)
axs[0, 2].set_title("Oracle")
axs[0, 2].set_ylabel("")
axs[0, 2].set_xlabel("")

# Boosting
axs[1, 0].pcolormesh(grid_x, grid_y, boosting_grid, alpha=0.675, cmap=cMap4)
axs[1, 0].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=15
)
axs[1, 0].set_title("Boosting")
axs[1, 0].set_ylabel("")
axs[1, 0].set_xlabel("")

# Oracle
axs[1, 1].pcolormesh(grid_x, grid_y, naive_grid, alpha=0.675, cmap=cMap5)
axs[1, 1].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=15
)
axs[1, 1].set_title("Monte-Carlo")
axs[1, 1].set_ylabel("")
axs[1, 1].set_xlabel("")

# Remove the axis [1, 2] plot
fig.delaxes(axs[1, 2])

axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticklabels([])
axs[0, 1].set_yticklabels([])
axs[0, 2].set_yticklabels([])
axs[1, 1].set_yticklabels([])

fig.supxlabel(r"$\theta_1$")
fig.supylabel(r"$\theta_2$")

# Add a single legend for scatter points
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.065), ncol=2)
plt.tight_layout()
plt.savefig(
    "results/figures/CI_e_value_two_moons.pdf", format="pdf", bbox_inches="tight"
)
plt.show()

# probability of coverage
torch.manual_seed(780)
torch.cuda.manual_seed(780)
n_prob = 1250
theta_true_repeat = np.repeat(theta_true.reshape(1, -1), n_prob * n, axis=0)
# simulating X
X_theta_true = simulator(theta_true_repeat)
X_theta_true = X_theta_true.reshape(n_prob, n * X_dim)
theta_repeat = np.repeat(theta_true.reshape(1, -1), n_prob, axis=0)

lambdas_true = score.compute(
    theta_repeat,
    X_theta_true.numpy(),
    disable_tqdm=False,
    N=1000,
)

TRUST_plus_cutoff = loforest_object.compute_cutoffs(theta_true.reshape(1, -1))
naive_cutoff = predict_naive_quantile(kind, theta_true.reshape(1, -1), naive_quantiles)
boosting_cutoff = boosting_model.predict(theta_true.reshape(1, -1))

# computing prob of coverage
TRUST_plus_cover = np.mean(lambdas_true <= TRUST_plus_cutoff[0])
boosting_cover = np.mean(lambdas_true <= boosting_cutoff[0])
naive_cover = np.mean(lambdas_true <= naive_cutoff[0])

import seaborn as sns

# Create a figure with subplots
fig, axs = plt.figure(figsize=(10, 6))
colors = [
    "darkorange",
    "darkgreen",
    "firebrick",
]
custom_palette = sns.set_palette(sns.color_palette(colors))

# List of dataframes and titles
data = pd.DataFrame(
    {
        "cover": [naive_cover, boosting_cover, TRUST_plus_cover],
        "method": ["MC", "Boosting", "TRUST++"],
    }
)
sns.barplot(x="method", y="cover", data=data, palette=custom_palette)
plt.axhline(0.95, linestyle="--", color="black")
plt.ylim(0.9, 1)
plt.xlabel("Methods")
plt.ylabel("Probability of coverage")
plt.rcParams.update({"font.size": 16})
plt.show()
fig.savefig("results/figures/prob_of_coverage_two_moons.pdf", format="pdf")
