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

# exponential empirical data
original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/experiment_K_exp.pkl"

obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
  data = obj_exp[key]
  data = data.assign(K = str(key))
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
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/exp_K_experiment.pdf")

# bff experiment
original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/experiment_K_bff.pkl"

obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
  data = obj_exp[key]
  data = data.assign(K = str(key))
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
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/bff_K_experiment.pdf")


# gamma experiment
original_path = os.getcwd()
data_path = original_path + "/experiments/results_data/experiment_K_gamma.pkl"

obj_exp = pd.read_pickle(data_path)
df_list = []
# creating data frame list to concatenate
for key in obj_exp.keys():
  data = obj_exp[key]
  data = data.assign(K = str(key))
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
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gamma_K_experiment.pdf")

