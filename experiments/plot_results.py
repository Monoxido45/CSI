import os
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plotting data of interest

# exponential empirical data
original_path = os.getcwd()

data_path = original_path + "/experiments/results_data/exp_data.csv"

exp_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/exp_data_results.pdf")


data_path = original_path + "/experiments/results_data/bff_data.csv"

exp_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/bff_data_results.pdf")


# gamma experiment
data_path = original_path + "/experiments/results_data/gamma_data.csv"

exp_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gamma_data_results.pdf")


# gmm experiment
data_path = original_path + "/experiments/results_data/gmm_data.csv"

exp_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gmm_data_results.pdf")

# plotting results with tuned loforest
# plotting exponential data results
data_path = original_path + "/experiments/results_data/exponential_data_tuned.csv"

exp_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/exp_tuned_data_results.pdf")


exp_new = exp_data[exp_data['methods'] != 'NAIVE']

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    exp_new,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/exp_tuned_no_naive_data_results.pdf")

# plotting bff results
data_path = original_path + "/experiments/results_data/bff_data_tuned.csv"

bff_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    bff_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/bff_tuned_data_results.pdf")

# plotting gamma results
data_path = original_path + "/experiments/results_data/gamma_data_tuned.csv"

gamma_data = pd.read_csv(data_path)

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    gamma_data,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gamma_tuned_data_results.pdf")

# plotting gamma results without naive
data_path = original_path + "/experiments/results_data/gamma_data_tuned.csv"

gamma_data = pd.read_csv(data_path)
gamma_new = gamma_data[gamma_data['methods'] != 'NAIVE']

sns.set(style="white", font_scale=1.5)
g = sns.FacetGrid(
    gamma_new,
    col="N",
    col_wrap=2,
    height=8,
    aspect=1.60,
    hue="methods",
    palette="Set1",
    margin_titles=True,
    sharey = False
)
g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
g.add_legend()

# savefigure
plt.savefig(original_path + "/experiments/figures/gamma_tuned_data_results_no_naive.pdf")
