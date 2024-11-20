import torch

# loforest and locart functions
from CP2LFI.loforest import ConformalLoforest, tune_loforest_LFI
from clover import LocartSplit
from CP2LFI.scores import LambdaScore
from CP2LFI.utils import naive, predict_naive_quantile

# quantile regression
from sklearn.ensemble import HistGradientBoostingRegressor

# plotting and numpy
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd

# utils functions
from CP2LFI.utils import obtain_quantiles
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
TRUST_plus_MV_grid = np.zeros(n_par**2)
TRUST_plus_uncertainty = np.zeros(n_par**2)
oracle_grid = np.zeros(n_par**2)

# Fill the grids based on the confidence regions
TRUST_grid[TRUST_filter] = 1
TRUST_plus_MV_grid[TRUST_plus_MV_filter] = 1
oracle_grid[oracle_filter] = 1
TRUST_plus_uncertainty[accept_filter] = 1
TRUST_plus_uncertainty[agnostic_filter] = 1 / 2

TRUST_grid = TRUST_grid.reshape(n_par, n_par)
TRUST_plus_MV_grid = TRUST_plus_MV_grid.reshape(n_par, n_par)
oracle_grid = oracle_grid.reshape(n_par, n_par)
TRUST_plus_uncertainty = TRUST_plus_uncertainty.reshape(n_par, n_par)


# Plotting the confidence regions as colored grids
from matplotlib import colors as c
import seaborn as sns

sns.set(style="ticks", font_scale=2)
fig, axs = plt.subplots(1, 3, figsize=(12, 8))
plt.rcParams.update({"font.size": 14})

cMap1 = c.ListedColormap(["white", "darkred"])
cMap2 = c.ListedColormap(["white", "salmon", "maroon"])
cMap3 = c.ListedColormap(["white", "darkgray"])

# TRUST
axs[0].pcolormesh(grid_x, grid_y, TRUST_plus_MV_grid, alpha=0.675, cmap=cMap1)
axs[0].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=12
)
axs[0].set_title("Trust++ tuned")
axs[0].set_ylabel(r"$\theta_1$")
axs[0].set_xlabel("")

# TRUST++ Tuned
axs[1].pcolormesh(grid_x, grid_y, TRUST_plus_uncertainty, alpha=0.675, cmap=cMap2)
axs[1].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=12
)
axs[1].set_title("Trust ++ uncertainty")
axs[1].set_ylabel("")
axs[1].set_xlabel(r"$\theta_2$")

# Oracle
axs[2].pcolormesh(grid_x, grid_y, oracle_grid, alpha=0.675, cmap=cMap3)
axs[2].scatter(
    theta_true[0], theta_true[1], marker="x", color="blue", label=r"True $\theta$", s=12
)
axs[2].set_title("Oracle")
axs[2].set_ylabel("")
axs[2].set_xlabel("")

axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
# Add a single legend for scatter points
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5)
plt.tight_layout()
plt.savefig(
    "results/figures/CI_e_value_two_moons.pdf", format="pdf", bbox_inches="tight"
)
plt.show()
