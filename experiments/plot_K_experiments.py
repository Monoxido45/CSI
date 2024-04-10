import os
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc

# extended palette
palette = sns.color_palette(cc.glasbey, n_colors=12)

# plotting data of interest

# --------------------------------------------------
# exponential experiment
# --------------------------------------------------
original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/experiment_K_exp.pkl"

obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
    data = obj_exp[key]
    data = data.assign(K=str(key))
    df_list.append(data)

# obtaining final data
exp_data = pd.concat(df_list)

# obtaining graph

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="K",
    palette=palette,
    margin_titles=True,
    sharey=False,
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/exp_K_experiment.pdf")

exp_data["K"] = exp_data["K"].astype(int)

# Select the optimal K for each B and N in exp_data
optimal_K = (
    exp_data.groupby(["B", "N"])
    .apply(lambda x: x.nsmallest(n=1, columns="MAE"))
    .reset_index(drop=True)
)


# Plotting K for each B
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "K")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/exp_optimal_K_versus_B.pdf")
plt.show()

# plotting each loss for B and K considering the optimal choosings of K
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "MAE")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/exp_optimal_K_loss.pdf")
plt.show()


# --------------------------------------------------
# bff experiment
# --------------------------------------------------

original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/experiment_K_bff.pkl"
obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
    data = obj_exp[key]
    data = data.assign(K=str(key))
    df_list.append(data)

# obtaining final data
bff_data = pd.concat(df_list)
# obtaining graph
sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    bff_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="K",
    palette=palette,
    margin_titles=True,
    sharey=False,
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()
# savefigure
plt.savefig(original_path + "/experiments/figures/bff_K_experiment.pdf")


exp_data["K"] = exp_data["K"].astype(int)

# Select the optimal K for each B and N in exp_data
optimal_K = (
    bff_data.groupby(["B", "N"])
    .apply(lambda x: x.nsmallest(n=1, columns="MAE"))
    .reset_index(drop=True)
)


# Plotting K for each B
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "K")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/bff_optimal_K_versus_B.pdf")
plt.show()

# plotting each loss for B and K considering the optimal choosings of K
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "MAE")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/bff_optimal_K_loss.pdf")
plt.show()


# --------------------------------------------------
# gamma experiment
# --------------------------------------------------
original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/experiment_K_gamma.pkl"

obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
    data = obj_exp[key]
    data = data.assign(K=str(key))
    df_list.append(data)

# obtaining final data
gamma_data = pd.concat(df_list)

# obtaining graph
sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    gamma_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="K",
    palette=palette,
    margin_titles=True,
    sharey=False,
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gamma_K_experiment.pdf")

exp_data["K"] = exp_data["K"].astype(int)

# Select the optimal K for each B and N in exp_data
optimal_K = (
    gamma_data.groupby(["B", "N"])
    .apply(lambda x: x.nsmallest(n=1, columns="MAE"))
    .reset_index(drop=True)
)


# Plotting K for each B
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "K")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/gamma_optimal_K_versus_B.pdf")
plt.show()

# plotting each loss for B and K considering the optimal choosings of K
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "MAE")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/gamma_optimal_K_loss.pdf")
plt.show()

# --------------------------------------------------
# GMM experiment
# --------------------------------------------------

original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/gmm_experiments/cov_5000.pkl"

obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
    data = obj_exp[key]
    data = data.assign(K=str(key))
    df_list.append(data)

# obtaining final data
gamma_data = pd.concat(df_list)

# obtaining graph
sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    gamma_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="K",
    palette=palette,
    margin_titles=True,
    sharey=False,
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gmm_K_experiment.pdf")

exp_data["K"] = exp_data["K"].astype(int)

# Select the optimal K for each B and N in exp_data
optimal_K = (
    gamma_data.groupby(["B", "N"])
    .apply(lambda x: x.nsmallest(n=1, columns="MAE"))
    .reset_index(drop=True)
)


# Plotting K for each B
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "K")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/gmm_optimal_K_versus_B.pdf")
plt.show()

# plotting each loss for B and K considering the optimal choosings of K
g = sns.FacetGrid(
    optimal_K,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    palette=palette,
    margin_titles=True,
    sharey=False,
)

g.map(sns.lineplot, "B", "MAE")
plt.tight_layout()
plt.savefig(original_path + "/experiments/figures/gmm_optimal_K_loss.pdf")
plt.show()

