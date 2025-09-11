import pandas as pd
import numpy as np
import os
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

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
    "tractable_bff": [1, 5, 10, 20],
    "two moons_bff": [1, 5, 10, 20],
    "weinberg_bff": [1, 5, 10, 20],
    "mg1_bff": [1, 5, 10, 20],
    # waldo
    "sir_waldo": [1, 5, 10, 20],
    "two moons_waldo": [1, 5, 10, 20],
    "weinberg_waldo": [1, 5, 10, 20],
    "mg1_waldo": [1, 5, 10, 20],
    "tractable_waldo": [1, 5, 10, 20],
    # e-value
    "mg1_e_value": [1, 5, 10, 20],
    "weinberg_e_value": [1, 5, 10, 20],
    "two moons_e_value": [1, 5, 10, 20],
    "tractable_e_value": [1, 5, 10, 20],
    "sir_e_value": [1, 5, 10, 20],
}

# B dictionary
B_dict = {
    # all bff entries
    # sir (completed)
    "sir_bff_1": [1e4, 1.5e4, 2e4, 3e4],
    "sir_bff_5": [1e4, 1.5e4, 2e4, 3e4],
    "sir_bff_10": [1e4, 1.5e4, 2e4, 3e4],
    "sir_bff_20": [1e4, 1.5e4, 2e4, 3e4],
    # tractable (completed)
    "tractable_bff_1": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_bff_5": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_bff_10": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_bff_20": [1e4, 1.5e4, 2e4, 3e4],
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
    # sir (completed)
    "sir_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "sir_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "sir_waldo_10": [1e4, 1.5e4, 2e4, 3e4],
    "sir_waldo_20": [1e4, 1.5e4, 2e4, 3e4],
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
    # mg1 (completed)
    "mg1_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_waldo_10": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_waldo_20": [1e4, 1.5e4, 2e4, 3e4],
    # tractable (completed)
    "tractable_waldo_1": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_waldo_5": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_waldo_10": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_waldo_20": [1e4, 1.5e4, 2e4, 3e4],
    # all e-value entries
    # mg1 (completed)
    "mg1_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "mg1_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
    # weinberg (completed)
    "weinberg_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "weinberg_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
    # two moons (completed)
    "two moons_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "two moons_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
    # tractable
    "tractable_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "tractable_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
    # sir (completed)
    "sir_e_value_1": [1e4, 1.5e4, 2e4, 3e4],
    "sir_e_value_5": [1e4, 1.5e4, 2e4, 3e4],
    "sir_e_value_10": [1e4, 1.5e4, 2e4, 3e4],
    "sir_e_value_20": [1e4, 1.5e4, 2e4, 3e4],
}

# grouping all measures together
all_measures = resume_all_methods(stats_name, kind_dict, n_dict, B_dict)

method_custom_order = CategoricalDtype(
    ["TRUST++ tuned", "TRUST++ MV", "boosting", "TRUST", "MC"],
    ordered=True,
)
# estabilishing method as an categorical variable
all_measures["method"] = all_measures["method"].astype(method_custom_order)

# new custom order
new_custom_order = CategoricalDtype(
    ["TRUST++ tuned", "TRUST++ MV", "TRUST", "boosting", "MC"],
    ordered=True,
)

# making heatmaps for each statistic
# making heatmaps separated by N for each statistic
B_s = all_measures["B"].unique()
stats = all_measures["stat"].unique()
stats_dict = {}
signif_dict = {}

# grouping simulation budget and model together
for stat in stats:
    stat_df = all_measures[all_measures["stat"] == stat]
    stat_df = stat_df.sort_values(by=["B", "n"], ascending=[True, True])

    stat_df["model_B"] = stat_df["kind"].astype(str) + "-" + stat_df["B"].astype(str)

    stat_df["method"] = stat_df["method"].astype(
        CategoricalDtype(
            ["TRUST++ tuned", "TRUST++ MV", "TRUST", "boosting", "MC"],
            ordered=True,
        )
    )

    stat_df["method"] = stat_df["method"].cat.rename_categories(
        {
            "TRUST++ tuned": "TRUST++\n tuned",
            "TRUST++ MV": "TRUST++\n MV",
        }
    )

    # Sort model_B by original model order, then by B ascending within each model
    # Extract original model order from stat_df["kind"]
    original_models = stat_df["kind"].unique().tolist()
    # Build ordered_model_B list
    ordered_model_B = []
    for model in original_models:
        # Get all B values for this model
        model_Bs = [
            mb for mb in stat_df["model_B"].unique() if mb.startswith(model + "-")
        ]
        # Sort B values numerically
        model_Bs_sorted = sorted(model_Bs, key=lambda mb: int(mb.split("-")[-1]))
        ordered_model_B.extend(model_Bs_sorted)
    stat_df["model_B"] = pd.Categorical(
        stat_df["model_B"], categories=ordered_model_B, ordered=True
    )

    # separating everything in lists for different N's
    N_s = np.sort(stat_df["n"].unique())
    for n in N_s:
        n_df = stat_df[stat_df["n"] == n].copy()
        # obtaining MAE array
        mae_matrix = n_df.pivot(index="method", columns="model_B", values="MAE")
        se_matrix = n_df.pivot(index="method", columns="model_B", values="SE") * 2

        # also deriving significance matrix
        significance_matrix = np.zeros(mae_matrix.shape)
        idxs = np.arange(mae_matrix.shape[0])
        for i in range(mae_matrix.shape[1]):
            col_mae = mae_matrix.iloc[:, i]
            col_se = se_matrix.iloc[:, i]
            min_idx = col_mae.idxmin()
            num_min_idx = int(col_mae.index.get_loc(min_idx))
            significance_matrix[num_min_idx, i] = 1

            idxs = np.arange(mae_matrix.shape[0])
            # checking significance of other methods
            excluded_mae_array = np.delete(
                col_mae,
                num_min_idx,
            )

            excluded_se_array = np.delete(
                col_se,
                num_min_idx,
            )

            excluded_idxs_array = np.delete(
                idxs,
                num_min_idx,
            )

            lim_sup = col_mae[num_min_idx] + col_se[num_min_idx]
            lim_inf = excluded_mae_array - excluded_se_array

            add_indexes = np.where(lim_sup >= lim_inf)[0]
            if add_indexes.size > 0:
                selected_indexes = excluded_idxs_array[add_indexes]
                significance_matrix[selected_indexes, i] = 1

        stats_dict[(stat, n)] = [mae_matrix, se_matrix]
        signif_dict[(stat, n)] = significance_matrix


# now plotting heatmaps
plt.rcParams.update({"font.size": 16})
for stat in stats:
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))

    for idx, N in enumerate(N_s):
        ax = axes.flatten()[idx]
        mae_matrix = stats_dict[(stat, N)][0]
        se_matrix = stats_dict[(stat, N)][1]
        significance_matrix = signif_dict[(stat, N)]

        # Define a discrete colormap with two colors: green for significant, white for not significant
        cmap = ListedColormap(["white", "mediumseagreen"])

        # Plot the heatmap with the discrete colormap
        heatmap = ax.imshow(
            significance_matrix,
            cmap=cmap,
            aspect="auto",
        )

        # Add gridlines to separate tiles
        ax.set_xticks(np.arange(-0.5, len(mae_matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mae_matrix.index), 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Add text (MAE and SE values) to each tile
        for i in range(mae_matrix.shape[0]):
            for j in range(mae_matrix.shape[1]):
                value = mae_matrix.iloc[i, j]
                se_value = se_matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.3f}\n({se_value:.3f})",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9.5,
                )

        # Set axis labels and ticks
        ax.set_xlabel("Benchmarks")
        if idx in [0, 2]:
            ax.set_ylabel("Methods")
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", labelsize=14)
        if idx in [2, 3]:
            ax.set_xticks(range(len(mae_matrix.columns)))
            ax.set_xticklabels(mae_matrix.columns, rotation=45, ha="right")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        if idx in [1, 3]:
            ax.set_yticklabels([])
        else:
            ax.set_yticks(range(len(mae_matrix.index)))
            ax.set_yticklabels(mae_matrix.index)

        for tick, label in zip(ax.get_yticklabels(), mae_matrix.index):
            if label in ["TRUST++\n tuned", "TRUST++\n MV", "TRUST"]:
                tick.set_fontweight("bold")
        ax.set_title(f"n: {N}")

    # Remove empty subplot if N_s has less than 6 elements
    if len(N_s) < axes.size:
        for idx in range(len(N_s), axes.size):
            fig.delaxes(axes.flatten()[idx])

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()  # Commented out to prevent overriding wspace
    plt.savefig(f"results/heatmap_{stat}_real.pdf")
    plt.show()

# function to compare and number of times each method performed best
# filtering by the best methods
filtered_data = all_measures.groupby(["n", "B", "stat", "kind"], as_index=False).apply(
    lambda df: df.nsmallest(n=1, columns="MAE", keep="all")
)

n_count = filtered_data.shape[0]

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
# plotting object (if needed)
plt.style.use("seaborn-white")
sns.set(style="ticks", font_scale=2.75)
plt.rcParams.update({"font.size": 16})
colors = [
    "firebrick",
    "darkblue",
    "rebeccapurple",
    "darkgreen",
    "darkorange",
    "goldenrod",
]
custom_palette = sns.set_palette(sns.color_palette(colors))

# counting all of data
method_counts = filtered_data.value_counts(["n", "B", "method"])
method_counts_data = method_counts.reset_index()
method_counts_data["method"] = method_counts_data["method"].astype(new_custom_order)

# Create a facet grid using catplot
g = sns.catplot(
    data=method_counts_data,
    x="method",
    y="count",
    kind="bar",
    col="n",
    row="B",
    legend=True,
    sharey=True,
    palette=custom_palette,
    height=6,
    aspect=1,
)

# Set the labels and titles
g.set_titles("n = {col_name}, B = {row_name}")
g.set_ylabels("")
g.set_xlabels("")
g.tick_params(axis="x", rotation=75)
g.fig.supylabel("Number of times each method performed better", fontsize=35, x=-0.005)


count = 0
for ax in g.axes.flatten():
    if count >= 12:
        for idx in [0, 1, 2]:
            ax.get_xticklabels()[idx].set_fontweight("bold")
    count += 1

# Show the plot
plt.tight_layout()

g.savefig("results/figures/all_real_comparissons.pdf", format="pdf")


####### Unused visualizations ##########
# making now barplots for coverage distance
all_measures["method"] = all_measures["method"].astype(new_custom_order)
# first, for distance < 0.05
filtered_data_dist_05 = all_measures.query("MAE <= 0.05")
value_counts_05 = np.round(filtered_data_dist_05["method"].value_counts() / n_count, 2)

# now for distance < 0.04
filtered_data_dist_04 = all_measures.query("MAE <= 0.035")
value_counts_04 = np.round(filtered_data_dist_04["method"].value_counts() / n_count, 2)

# now for distance < 0.02
filtered_data_dist_02 = all_measures.query("MAE <= 0.02")
value_counts_02 = np.round(filtered_data_dist_02["method"].value_counts() / n_count, 2)

# now for distance < 0.01
filtered_data_dist_01 = all_measures.query("MAE <= 0.0125")
value_counts_01 = np.round(filtered_data_dist_01["method"].value_counts() / n_count, 2)

df_lists = [
    value_counts_05,
    value_counts_04,
    value_counts_02,
    value_counts_01,
    filtered_data_dist_05,
    filtered_data_dist_04,
    filtered_data_dist_02,
    filtered_data_dist_01,
]


dist_list = [0.05, 0.035, 0.02, 0.0125, 0.05, 0.035, 0.02, 0.0125]

sns.set(style="ticks", font_scale=2)
plt.rcParams.update({"font.size": 14})
colors = [
    "firebrick",
    "darkblue",
    "rebeccapurple",
    "darkgreen",
    "darkorange",
    "goldenrod",
]
custom_palette = sns.set_palette(sns.color_palette(colors))
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
count = 0
for ax, distance, df in zip(axes, dist_list, df_lists):
    if count == 0:
        ax.set_ylabel("Proportion")
    elif count == 4:
        ax.set_ylabel("MAE")
    else:
        ax.set_ylabel("")

    if count <= 3:
        sns.barplot(x=df.index, y=df.values, palette=custom_palette, ax=ax)
        ax.set_title(r"$MAE \leq {}$".format(distance))
        ax.set_xlabel("")
        ax.set_xticks([])
    else:
        sns.boxplot(
            data=df,
            x="method",
            y="MAE",
            hue="method",
            palette=custom_palette,
            boxprops=dict(alpha=0.65),
            ax=ax,
        )
        if count > 4:
            ax.set_ylabel("")
        ax.get_legend().remove()
        ax.set_xlabel("Method")
        ax.tick_params(axis="x", rotation=90)
        for idx in [0, 1, 2]:
            ax.get_xticklabels()[idx].set_fontweight("bold")

    count += 1
plt.tight_layout()
plt.savefig("results/figures/method_counts_all_MAE_real.pdf", format="pdf")
plt.show()

########### Extra supplementary material visualizations #############
stats_list = ["bff", "waldo", "e_value"]
model_list = ["sir", "two moons", "weinberg", "mg1", "tractable"]

colors = [
    "firebrick",
    "darkblue",
    "rebeccapurple",
    "darkgreen",
    "darkorange",
]
custom_palette = sns.color_palette(colors)

new_custom_order = CategoricalDtype(
    ["TRUST++ tuned", "TRUST++ MV", "TRUST", "boosting", "MC"],
    ordered=True,
)

for stat_sel in stats_list:
    for model in model_list:
        df_sel = all_measures.query("stat == @stat_sel").query("kind == @model")
        df_sel["method"] = df_sel["method"].astype(new_custom_order)
        sns.set(style="ticks", font_scale=2)
        plt.rcParams.update({"font.size": 14})
        g = sns.FacetGrid(
            df_sel,
            col="n",
            col_wrap=2,
            height=6,
            aspect=1.50,
            hue="method",
            palette=custom_palette,
            margin_titles=True,
            sharey=False,
        )
        g.map(plt.errorbar, "B", "MAE", "SE", fmt="-o")
        g.add_legend()

        # savefigure
        plt.savefig(
            f"results/sup_figures/{stat_sel}_{model}_MAE_plot.pdf", format="pdf"
        )


######### Unused visualizations ##########
# Making violinplot of total MAE
sns.set(style="ticks", font_scale=2.75)
plt.rcParams.update({"font.size": 16})
plt.figure(figsize=(12, 8))
colors = [
    "firebrick",
    "darkblue",
    "rebeccapurple",
    "darkgreen",
    "darkorange",
    "goldenrod",
]
custom_palette = sns.set_palette(sns.color_palette(colors))

ax = sns.violinplot(
    data=all_measures,
    y="method",
    x="MAE",
    palette=custom_palette,
    fill=False,
    saturation=0.7,
    cut=0,
)
plt.yticks(
    ticks=range(len(method_custom_order.categories)),
    labels=method_custom_order.categories,
)
plt.title("Distribution of MAE for Each Method")
plt.ylabel("Method")
plt.xlabel("MAE")
plt.xticks(rotation=75)
# changing alpha
for patch in ax.collections:
    patch.set_alpha(0.65)
plt.tight_layout()
plt.savefig("results/figures/MAE_violinplot_overall_real_data.pdf", format="pdf")
plt.show()

# making boxplot version
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=all_measures,
    y="method",
    x="MAE",
    palette=custom_palette,
    boxprops=dict(alpha=0.65),
)
plt.yticks(
    ticks=range(len(method_custom_order.categories)),
    labels=method_custom_order.categories,
)
plt.title("Distribution of MAE for Each Method")
plt.ylabel("Method")
plt.xlabel("MAE")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("results/figures/MAE_boxplot_overall_real_data.pdf", format="pdf")
plt.show()

# boxplots for each n and B
g = sns.catplot(
    data=all_measures,
    x="method",
    y="MAE",
    kind="box",
    hue="method",
    boxprops=dict(alpha=0.65),
    col="n",
    row="B",
    legend=False,
    sharey=False,
    palette=custom_palette,
    height=6,
    aspect=1,
)
# Set the labels and titles
g.set_titles("n = {col_name}, B = {row_name}")
g.set_ylabels("")
g.set_xlabels("")
g.tick_params(axis="x", rotation=90)


# Adjust the layout to put the y label outside of the graph
g.fig.subplots_adjust(left=0.5)
g.fig.supylabel("Mean Absolute Error regarding nominal coverage", fontsize=35, x=-0.005)
g.set(yscale="log")
plt.tight_layout()
g.savefig("results/figures/MAE_boxplots_N_B_real.pdf", format="pdf")
