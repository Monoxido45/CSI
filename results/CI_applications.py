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
from scipy.spatial import ConvexHull

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
B = 1e4
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


# obtaining intervals for specific sample
torch.manual_seed(125)
torch.cuda.manual_seed(125)
n_par = 100
pars_1 = np.linspace(-0.99, 0.99, n_par)
theta_true = np.array([0.5, -0.25])
theta_grid = np.c_[list(itertools.product(pars_1, pars_1))]

# computing cutoffs
if theta_grid.ndim == 1:
    model_eval = theta_grid.reshape(-1, 1)
else:
    model_eval = theta_grid

idxs = locart_object.cart.apply(model_eval)
list_locart_quantiles = [locart_quantiles[idx] for idx in idxs]
locart_cutoffs = np.array(list_locart_quantiles)

# loforest
loforest_cutoffs = loforest_object.compute_cutoffs(model_eval)

# tuned loforest
loforest_tuned_cutoffs = loforest_object.compute_cutoffs(model_eval, K=K_loforest)

# boosting
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

# getting confidence regions
TRUST_plus_MV_filter = np.where(lambdas_eval <= loforest_cutoffs)
TRUST_plus_MV_conf = theta_grid[TRUST_plus_MV_filter]

TRUST_plus_tuned_filter = np.where(lambdas_eval <= loforest_tuned_cutoffs)
TRUST_plus_tuned_conf = theta_grid[TRUST_plus_tuned_filter]

TRUST_filter = np.where(lambdas_eval <= locart_cutoffs)
TRUST_conf = theta_grid[TRUST_filter]

naive_filter = np.where(lambdas_eval <= np.array(naive_list))
naive_conf = theta_grid[naive_filter]

boosting_filter = np.where(lambdas_eval <= boosting_quantiles)
boosting_conf = theta_grid[boosting_filter]

# Create a grid for plotting
grid_x, grid_y = np.meshgrid(pars_1, pars_1)

# Initialize a grid for each method
TRUST_grid = np.zeros(n_par**2)
TRUST_plus_MV_grid = np.zeros(n_par**2)
TRUST_plus_tuned_grid = np.zeros(n_par**2)
naive_grid = np.zeros(n_par**2)
boosting_grid = np.zeros(n_par**2)

# Fill the grids based on the confidence regions
TRUST_grid[TRUST_filter] = 1
TRUST_plus_MV_grid[TRUST_plus_MV_filter] = 1
TRUST_plus_tuned_grid[TRUST_plus_tuned_filter] = 1
naive_grid[naive_filter] = 1
boosting_grid[boosting_filter] = 1

TRUST_grid = TRUST_grid.reshape(n_par, n_par)
TRUST_plus_MV_grid = TRUST_plus_MV_grid.reshape(n_par, n_par)
TRUST_plus_tuned_grid = TRUST_plus_tuned_grid.reshape(n_par, n_par)
naive_grid = naive_grid.reshape(n_par, n_par)
boosting_grid = boosting_grid.reshape(n_par, n_par)


# Plotting the confidence regions as colored grids
from matplotlib import colors as c
import seaborn as sns

sns.set(style="ticks", font_scale=2)
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plt.rcParams.update({"font.size": 14})

cMap1 = c.ListedColormap(["white", "rebeccapurple"])
cMap2 = c.ListedColormap(["white", "darkred"])
cMap3 = c.ListedColormap(["white", "darkgreen"])
cMap4 = c.ListedColormap(["white", "darkorange"])

# TRUST
axs[0, 0].pcolormesh(grid_y, grid_x, TRUST_grid, alpha=0.65, cmap=cMap1)
axs[0, 0].scatter(
    theta_true[0], theta_true[1], color="black", label=r"True $\theta$", s=5
)
axs[0, 0].set_title("Trust")
axs[0, 0].set_ylabel(r"$\theta_1$")
axs[0, 0].set_xlabel(r"$\theta_2$")

# TRUST++ Tuned
axs[0, 1].pcolormesh(grid_y, grid_x, TRUST_plus_tuned_grid, alpha=0.65, cmap=cMap2)
axs[0, 1].scatter(
    theta_true[0], theta_true[1], color="black", label=r"True $\theta$", s=5
)
axs[0, 1].set_title("Trust ++ Tuned")
axs[0, 1].set_ylabel(r"$\theta_1$")
axs[0, 1].set_xlabel(r"$\theta_2$")

# Boosting
axs[1, 0].pcolormesh(grid_y, grid_x, boosting_grid, alpha=0.65, cmap=cMap3)
axs[1, 0].scatter(
    theta_true[0], theta_true[1], color="black", label=r"True $\theta$", s=5
)
axs[1, 0].set_title("Boosting")
axs[1, 0].set_ylabel(r"$\theta_1$")
axs[1, 0].set_xlabel(r"$\theta_2$")

# MC
axs[1, 1].pcolormesh(grid_y, grid_x, naive_grid, alpha=0.65, cmap=cMap4)
axs[1, 1].scatter(
    theta_true[0], theta_true[1], color="black", label=r"True $\theta$", s=5
)
axs[1, 1].set_title("MC")
axs[1, 1].set_ylabel(r"$\theta_1$")
axs[1, 1].set_xlabel(r"$\theta_2$")
# Add a single legend for scatter points
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5)
plt.tight_layout()
plt.savefig(
    "results/figures/CI_e_value_two_moons.pdf", format="pdf", bbox_inches="tight"
)
plt.show()
