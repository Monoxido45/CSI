import pandas as pd
import numpy as np
import os
from pandas.api.types import CategoricalDtype

original_path = os.getcwd()
folder_path = "/results/LFI_real_results/"


def calculate_means_and_adjusted_stds(csv_file_path, overall=False):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    if overall:
        # Group by 'N' and 'B' columns
        grouped = df.groupby(["N", "B"])

        # Calculate the mean of each group
        means = grouped.mean()

        # Calculate the standard deviation of each group
        stds = grouped.std()

        # Divide the standard deviations by sqrt(30)
        adjusted_stds = stds / np.sqrt(30)

    else:
        # Calculate the mean of each column
        means = df.mean()

        # Calculate the standard deviation of each column
        stds = df.std()

        # Divide the standard deviations by sqrt(30)
        adjusted_stds = stds / np.sqrt(30)

    return means, adjusted_stds


def resume_all_methods(stats_name, kind_dict, n_dict, B_dict):
    methods_list = ["TRUST", "TRUST++ MV", "TRUST++ tuned", "boosting", "MC"]
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["n", "B", "kind", "stat", "method", "MAE", "SE"])

    # Loop over the kind_list
    for stat in stats_name:
        kind_list = kind_dict[stat]
        for kind in kind_list:
            # Loop over the n_list
            n_list = n_dict[f"{kind}_{stat}"]
            for n in n_list:
                # Get the corresponding B_list for the current n
                B_list = B_dict[f"{kind}_{stat}_{n}"]

                # Loop over the B_list
                for B in B_list:
                    B = int(B)
                    # Construct the file path
                    file_path = f"{kind}/{stat}/MAE_data_{n}_{B}.csv"

                    # Calculate the means and adjusted stds
                    means, adjusted_stds = calculate_means_and_adjusted_stds(
                        original_path + folder_path + file_path, overall=False
                    )

                    # selecting rows and columns
                    means = means[1:6].to_numpy()
                    adjusted_stds = adjusted_stds[1:6].to_numpy() * 2

                    # Create a new row for the results DataFrame
                    new_row = pd.DataFrame(
                        {
                            "n": np.repeat(n, 5),
                            "B": np.repeat(B, 5),
                            "kind": np.repeat(kind, 5),
                            "stat": np.repeat(stat, 5),
                            "method": methods_list,
                            "MAE": means,
                            "SE": adjusted_stds,
                        }
                    )

                    # Append the new row to the results DataFrame
                    results_df = results_df._append(new_row, ignore_index=True)

    return results_df.reset_index()


# creating all dicts
# first, list of names os statistics
stats_name = ["bff", "waldo", "e_value"]

# kind dictionary
kind_dict = {
    "bff": ["sir", "two moons", "weinberg", "mg1", "tractable"],
    "waldo": ["sir", "two moons", "weinberg", "mg1", "tractable"],
    "e_value": ["sir", "two moons", "weinberg", "mg1", "tractable"],
}

# n dictionary
n_dict = {
    # bff
    "sir_bff": [1, 5, 10, 20],
    "tractable_bff": [1, 5, 20],
    "two moons_bff": [1, 5, 10, 20],
    "weinberg_bff": [1, 5, 10, 20],
    "mg1_bff": [1, 5, 10, 20],
    # waldo
    "sir_waldo": [1, 5, 10],
    "two moons_waldo": [1, 5, 10, 20],
    "weinberg_waldo": [1, 5, 10, 20],
    "mg1_waldo": [1, 5, 10],
    "tractable_waldo": [1, 5],
    # e-value
    "mg1_e_value": [1, 5],
    "weinberg_e_value": [1, 5, 10, 20],
    "two moons_e_value": [1, 5, 10, 20],
    "tractable_e_value": [1, 5],
    "sir_e_value": [1, 5],
}

# B dictionary
B_dict = {
    # all bff entries
    # sir (completed)
    "sir_bff_1": [1e4, 1.5e4, 2e4, 3e4],  # n = 10 ,20
    "sir_bff_5": [1e4, 1.5e4, 2e4, 3e4],
    "sir_bff_10": [1e4, 1.5e4, 2e4, 3e4],
    "sir_bff_20": [1e4, 1.5e4, 2e4, 3e4],
    # tractable (almost completed)
    "tractable_bff_1": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_bff_5": [1e4, 1.5e4, 2e4, 3e4],  # n = 10
    "tractable_bff_20": [1e4, 1.5e4, 2e4, 3e4],  # 3e4
    # two moons (completed)
    "two moons_bff_1": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_bff_5": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_bff_10": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_bff_20": [1e4, 1.5e4, 2e4, 3e4],
    # weinberg (completed)
    "weinberg_bff_1": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_bff_5": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_bff_10": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_bff_20": [1e4, 1.5e4, 2e4, 3e4],
    # mg1 (completed)
    "mg1_bff_1": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_bff_5": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_bff_10": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_bff_20": [1e4, 1.5e4, 2e4, 3e4],
    # all waldo entries
    # sir
    "sir_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "sir_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "sir_waldo_10": [1e4, 1.5e4, 2e4, 3e4],
    # two moons (completed)
    "two moons_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_waldo_10": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_waldo_20": [1e4, 1.5e4, 2e4, 3e4],
    # weinberg (completed)
    "weinberg_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_waldo_10": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_waldo_20": [1e4, 1.5e4, 2e4, 3e4],
    # mg1
    "mg1_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_waldo_10": [1e4],
    # tractable
    "tractable_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    # all e-value entries
    # mg1
    "mg1_e_value_1": [1e4, 1.5e4, 2e4, 3e4],  # 2e4, 3e4
    "mg1_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    # weinberg (completed)
    "weinberg_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
    # two moons
    "two moons_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
    # tractable
    "tractable_e_value_1": [1e4, 1.5e4],  # 2e3, 3e4
    "tractable_e_value_5": [1e4, 1.5e4],
    # sir
    "sir_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "sir_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
}

# grouping all measures together
all_measures = resume_all_methods(stats_name, kind_dict, n_dict, B_dict)

method_custom_order = CategoricalDtype(
    ["TRUST", "TRUST++ MV", "TRUST++ tuned", "boosting", "MC"],
    ordered=True,
)
# estabilishing method as an categorical variable
all_measures["method"] = all_measures["method"].astype(method_custom_order)

# function to compare and number of times each method performed best
# copying t_analysis from clover
# filtering by the best methods
filtered_data = all_measures.groupby(["n", "B", "stat", "kind"], as_index=False).apply(
    lambda df: df.nsmallest(n=1, columns="MAE", keep="all")
)

# analysing whether other methods also configure as the best by
for i in range(filtered_data.shape[0]):
    # selecting values for n, B, kind and stats
    series_values = filtered_data.iloc[i, :]
    n_best, B_best = series_values["n"], series_values["B"]
    kind_best, stat_best = series_values["kind"], series_values["stat"]
    best_method = series_values["method"]
    best_MAE, best_SE = series_values["MAE"], series_values["SE"]

    # comparing it to the other methods
    measures_comp = (
        all_measures.query("n == @n_best")
        .query("B == @B_best")
        .query("kind == @kind_best")
        .query("kind == @kind_best")
        .query("stat == @stat_best")
        .query("method != @best_method")
    )

    # extracting each method
    methods_MAE = measures_comp.iloc[:, 6].to_numpy()
    methods_SE = measures_comp.iloc[:, 7].to_numpy()

    # comparing to best_mae and best_SE
    lim_inf = methods_MAE - methods_SE
    lim_sup = best_MAE + best_SE

    # method names in numpy array
    methods_names = measures_comp.iloc[:, 5].to_numpy()

    # obtaining indexes
    idx_add = np.where(lim_inf <= lim_sup)
    if idx_add[0].shape[0] > 0:
        names_sel = methods_names[idx_add]
        mae_sel = methods_MAE[idx_add]
        se_sel = methods_SE[idx_add]
        sel_size = se_sel.shape[0]
        # creating additional pandas data frame
        new_row = pd.DataFrame(
            {
                "n": np.repeat(n_best, sel_size),
                "B": np.repeat(B_best, sel_size),
                "kind": np.repeat(kind_best, sel_size),
                "stat": np.repeat(stat_best, sel_size),
                "method": names_sel,
                "MAE": mae_sel,
                "SE": se_sel,
            }
        )
        filtered_data = filtered_data._append(new_row, ignore_index=True)

filtered_data

# counting the method without grouping by n and B
filtered_data["method"].value_counts()


# plotting as a function of B and N
import seaborn as sns
import matplotlib.pyplot as plt

# plotting object (if needed)
plt.style.use("seaborn-white")
sns.set_palette("tab10")
plt.rcParams.update({"font.size": 12})

# Set the style of the plot
sns.set(style="ticks", font_scale=2.75)

# counting all of data
method_counts = filtered_data.value_counts(["n", "B", "method"])
method_counts_data = method_counts.reset_index()

# Create a facet grid using catplot
g = sns.catplot(
    data=method_counts_data,
    x="method",
    y="count",
    kind="bar",
    col="n",
    row="B",
    legend=True,
    sharey=False,
    palette="Set1",
    height=6,
    aspect=1,
)

# Set the labels and titles
g.set_titles("n = {col_name}, B = {row_name}")
g.set_ylabels("")
g.set_xlabels("")
g.tick_params(axis="x", rotation=75)
g.fig.supylabel("Number of times each method performed better")

# Show the plot
plt.tight_layout()

g.savefig("results/figures/all_real_comparissons.pdf", format="pdf")
