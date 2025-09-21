import pandas as pd
import os
import re
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# plotting object (if needed)
plt.style.use("seaborn-white")
sns.set_palette("tab10")
plt.rcParams.update({"font.size": 12})

# Define the path to the folder containing the CSV files
folder_path = "results/res_data"

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through each file in the specified folder
for file in os.listdir(folder_path):
    # Check if the file is a CSV file
    if file.endswith(".csv"):
        # Construct the full file path
        file_path = os.path.join(folder_path, file)
        # Read the CSV file into a DataFrame
        temp_df = pd.read_csv(file_path, index_col=0)

        # Extract the statistic and model from the file name using regex
        # Adjusted regex to match the corrected requirement
        match = re.match(r"(\w+)_([^_]+)_.*stats_data", file)
        if match:
            statistic = match.group(1)
            model = match.group(2)
            print(statistic)
            print(model)
        else:
            statistic = "Unknown"
            model = "Unknown"

        # Add the statistic and model as columns to the temp DataFrame
        temp_df["Statistic"] = statistic
        temp_df["Model"] = model

        # Append the temp DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# renaming df
combined_df["methods"] = combined_df["methods"].replace(
    {"KS": "asymptotic", "LR": "asymptotic"}
)

# Display the combined DataFrame
method_custom_order = CategoricalDtype(
    ["tuned LOFOREST", "LOFOREST", "boosting", "LOCART", "monte-carlo", "asymptotic"],
    ordered=True,
)

combined_df["methods"] = combined_df["methods"].astype(method_custom_order)
combined_df["methods"] = combined_df["methods"].cat.rename_categories(
    {
        "LOCART": "TRUST",
        "LOFOREST": "TRUST++ MV",
        "tuned LOFOREST": "TRUST++ tuned",
        "monte-carlo": "MC",
    }
)

combined_df = combined_df[(combined_df["N"] != 5) & (combined_df["N"] != 1)]

model_custom_order = CategoricalDtype(
    ["1dnormal", "gmm", "lognormal"],
    ordered=True,
)
combined_df["Model"] = combined_df["Model"].astype(model_custom_order)

# making heatmaps separated by N for each statistic
B_s = combined_df["B"].unique()
stats = combined_df["Statistic"].unique()
stats_dict = {}
signif_dict = {}

# grouping simulation budget and model together
for stat in stats:
    stat_df = combined_df[combined_df["Statistic"] == stat]
    stat_df = stat_df.sort_values(by=["B", "N"], ascending=[True, True])

    if stat == "BFF":
        stat_df["methods"] = stat_df["methods"].astype(
            CategoricalDtype(
                ["TRUST++ tuned", "TRUST++ MV", "TRUST", "boosting", "MC"],
                ordered=True,
            )
        )
    else:
        stat_df["methods"] = stat_df["methods"].astype(
            CategoricalDtype(
                [
                    "TRUST++ tuned",
                    "TRUST++ MV",
                    "TRUST",
                    "boosting",
                    "MC",
                    "asymptotic",
                ],
                ordered=True,
            )
        )

    stat_df["methods"] = stat_df["methods"].cat.rename_categories(
        {
            "TRUST++ tuned": "TRUST++\n tuned",
            "TRUST++ MV": "TRUST++\n MV",
        }
    )

    stat_df["model_B"] = stat_df["Model"].astype(str) + "-" + stat_df["B"].astype(str)

    # Sort model_B by original model order, then by B ascending within each model
    original_models = ["1dnormal", "gmm", "lognormal"]
    ordered_model_B = []
    for model in original_models:
        # Get all model_B values for this model
        model_Bs = [
            mb for mb in stat_df["model_B"].unique() if mb.startswith(model + "-")
        ]
        # Sort B values numerically (extract after "-")
        model_Bs_sorted = sorted(model_Bs, key=lambda mb: int(mb.split("-")[-1]))
        ordered_model_B.extend(model_Bs_sorted)

    # Change "-" to ": B=" in model_B column
    stat_df["model_B"] = stat_df["model_B"].str.replace("-", ": B=")

    # Also update ordered_model_B accordingly
    ordered_model_B = [mb.replace("-", ": B=") for mb in ordered_model_B]

    stat_df["model_B"] = pd.Categorical(
        stat_df["model_B"], categories=ordered_model_B, ordered=True
    )

    # separating everything in lists for different N's
    N_s = np.sort(stat_df["N"].unique())
    for n in N_s:
        n_df = stat_df[stat_df["N"] == n].copy()

        # obtaining MAE array
        mae_matrix = n_df.pivot(index="methods", columns="model_B", values="MAE")
        se_matrix = n_df.pivot(index="methods", columns="model_B", values="se") * 2

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
    print(f"Plotting heatmaps for statistic: {stat}")
    N_s = [10, 20, 50, 100]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

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
            interpolation="none",
            extent=[
                -0.5,
                significance_matrix.shape[1] - 0.5,
                significance_matrix.shape[0] - 0.5,
                -0.5,
            ],
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
        #        if stat != "KS" and stat != "LR":
        #            if idx in [3, 4]:
        #                ax.set_xticks(range(len(mae_matrix.columns)))
        #                ax.set_xticklabels(mae_matrix.columns, rotation=45, ha="right")
        #            else:
        #                ax.set_xlabel("")
        #                ax.set_xticklabels([])
        #        elif stat == "LR":
        #            if idx in [0, 3, 4]:
        #                ax.set_xticks(range(len(mae_matrix.columns)))
        #                ax.set_xticklabels(mae_matrix.columns, rotation=45, ha="right")
        #           else:
        #               ax.set_xlabel("")
        #               ax.set_xticklabels([])
        #       else:
        if idx in [2, 3]:
            ax.set_xticks(range(len(mae_matrix.columns)))
            ax.set_xticklabels(mae_matrix.columns, rotation=45, ha="right")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        #        if stat != "KS":
        #            if idx in [1, 2, 4]:
        #                ax.set_yticklabels([])
        #            else:
        #                ax.set_yticks(range(len(mae_matrix.index)))
        #                ax.set_yticklabels(mae_matrix.index)
        #        else:
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
    plt.savefig(f"results/heatmap_{stat}_sim.pdf", format="pdf")


# making general barplot for performance
filtered_data = combined_df.groupby(
    ["N", "B", "Model", "Statistic"], as_index=False
).apply(lambda df: df.nsmallest(n=1, columns="MAE"))

n_methods = filtered_data.shape[0]
for i in range(n_methods):
    # selecting values for n, B, kind and stats
    series_values = filtered_data.iloc[i, :]
    n_best, B_best = series_values["N"], series_values["B"]
    model_best, stat_best = series_values["Model"], series_values["Statistic"]
    best_method = series_values["methods"]
    best_MAE, best_SE = series_values["MAE"], 2 * series_values["se"]

    # comparing it to the other methods
    measures_comp = (
        combined_df.query("N == @n_best")
        .query("B == @B_best")
        .query("Model == @model_best")
        .query("Statistic == @stat_best")
        .query("methods != @best_method")
    )

    # extracting each method
    methods_MAE = measures_comp.iloc[:, 3].to_numpy()
    methods_SE = measures_comp.iloc[:, 4].to_numpy() * 2

    # comparing to best_mae and best_SE
    lim_inf = methods_MAE - methods_SE
    lim_sup = best_MAE + best_SE

    # method names in numpy array
    methods_names = measures_comp.iloc[:, 0].to_numpy()

    # obtaining indexes
    idx_add = np.where((lim_inf <= lim_sup) | (methods_MAE == best_MAE))
    if idx_add[0].shape[0] > 0:
        names_sel = methods_names[idx_add]
        mae_sel = methods_MAE[idx_add]
        se_sel = methods_SE[idx_add]
        sel_size = se_sel.shape[0]

        # creating additional pandas data frame
        new_row = pd.DataFrame(
            {
                "methods": names_sel,
                "N": np.repeat(n_best, sel_size),
                "B": np.repeat(B_best, sel_size),
                "MAE": mae_sel,
                "se": se_sel,
                "K_tuned": 0,
                "Statistic": np.repeat(stat_best, sel_size),
                "Model": np.repeat(model_best, sel_size),
            }
        )
        filtered_data = filtered_data._append(new_row, ignore_index=True)

filtered_data

# counting the method without grouping by n and B
filtered_data["methods"].value_counts()

# new custom order
new_custom_order = CategoricalDtype(
    ["TRUST++ tuned", "TRUST++ MV", "TRUST", "boosting", "MC", "asymptotic"],
    ordered=True,
)
method_counts = filtered_data.value_counts(["N", "B", "methods"])
method_counts_data = method_counts.reset_index()
method_counts_data["methods"] = method_counts_data["methods"].astype(new_custom_order)

# creating custom palette
sns.set_theme(style="ticks", font_scale=2.75)
plt.rcParams.update({"font.size": 14})
colors = [
    "firebrick",
    "darkblue",
    "rebeccapurple",
    "darkgreen",
    "darkorange",
    "goldenrod",
]
custom_palette = colors  # Pass this directly to palette argument

# Create a facet grid using catplot
g = sns.catplot(
    data=method_counts_data,
    x="methods",
    y="count",
    kind="bar",
    col="N",
    row="B",
    legend=False,
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
# Adjust the layout to put the y label outside of the graph
g.figure.subplots_adjust(left=0.5)
g.figure.supylabel(
    "Number of times each method performed better",
    fontsize=35,
    x=-0.005,
)
plt.tight_layout()
count = 0
for ax in g.axes.flatten():
    if count >= 12:
        for idx in [0, 1, 2]:
            ax.get_xticklabels()[idx].set_fontweight("bold")
    count += 1

g.savefig("results/figures/all_comparissons_v2.pdf", format="pdf")


#### Unused visualizations ##########
# changing combined_df method
combined_df["methods"] = combined_df["methods"].astype(new_custom_order)
# making now barplots for coverage distance
# avoiding BFF for comparing all methods to asymptotic
# first, for distance < 0.05
filtered_data_dist_05 = combined_df.query("MAE <= 0.05").query("Statistic != 'BFF'")
value_counts_05 = np.round(
    filtered_data_dist_05["methods"].value_counts() / n_methods, 2
)

# first, for distance < 0.035
filtered_data_dist_04 = combined_df.query("MAE <= 0.035").query("Statistic != 'BFF'")
value_counts_04 = np.round(
    filtered_data_dist_04["methods"].value_counts() / n_methods, 2
)

# first, for distance < 0.02
filtered_data_dist_03 = combined_df.query("MAE <= 0.02").query("Statistic != 'BFF'")
value_counts_03 = np.round(
    filtered_data_dist_03["methods"].value_counts() / n_methods, 2
)

# first, for distance < 0.01
filtered_data_dist_02 = combined_df.query("MAE <= 0.01").query("Statistic != 'BFF'")
value_counts_02 = np.round(
    filtered_data_dist_02["methods"].value_counts() / n_methods, 2
)

df_lists = [
    value_counts_05,
    value_counts_04,
    value_counts_03,
    value_counts_02,
    filtered_data_dist_05,
    filtered_data_dist_04,
    filtered_data_dist_03,
    filtered_data_dist_02,
]
dist_list = [0.05, 0.035, 0.02, 0.01, 0.05, 0.035, 0.02, 0.01]


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
            x="methods",
            y="MAE",
            hue="methods",
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
plt.savefig("results/figures/method_counts_all_MAE.pdf", format="pdf")
plt.show()

# now, excluding asymptotic and using all statistics
filtered_data_dist_05 = combined_df.query("MAE <= 0.05").query(
    "methods != 'asymptotic'"
)
value_counts_05 = filtered_data_dist_05["methods"].value_counts()

# first, for distance < 0.035
filtered_data_dist_04 = combined_df.query("MAE <= 0.035").query(
    "methods != 'asymptotic'"
)
value_counts_04 = filtered_data_dist_04["methods"].value_counts()

# first, for distance < 0.02
filtered_data_dist_03 = combined_df.query("MAE <= 0.02").query(
    "methods != 'asymptotic'"
)
value_counts_03 = filtered_data_dist_03["methods"].value_counts()

# first, for distance < 0.01
filtered_data_dist_02 = combined_df.query("MAE <= 0.01").query(
    "methods != 'asymptotic'"
)
value_counts_02 = filtered_data_dist_02["methods"].value_counts()

df_lists = [
    value_counts_05,
    value_counts_04,
    value_counts_03,
    value_counts_02,
    filtered_data_dist_05,
    filtered_data_dist_04,
    filtered_data_dist_03,
    filtered_data_dist_02,
]
dist_list = [0.05, 0.035, 0.02, 0.01, 0.05, 0.035, 0.02, 0.01]


sns.set(style="ticks", font_scale=2)
plt.rcParams.update({"font.size": 14})
colors = [
    "firebrick",
    "darkblue",
    "rebeccapurple",
    "darkgreen",
    "darkorange",
]

new_drop_custom_order = CategoricalDtype(
    ["TRUST++ tuned", "TRUST++ MV", "TRUST", "boosting", "MC"],
    ordered=True,
)

custom_palette = sns.set_palette(sns.color_palette(colors))
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
count = 0
for ax, distance, df in zip(axes, dist_list, df_lists):
    if count == 0:
        ax.set_ylabel("Count")
    elif count == 4:
        ax.set_ylabel("MAE")
    else:
        ax.set_ylabel("")
    if count <= 3:
        idx_used = df.index.astype(new_drop_custom_order)[:-1]
        values_used = df.values[:-1]
        sns.barplot(x=idx_used, y=values_used, palette=custom_palette, ax=ax)
        ax.set_title(r"$MAE \leq {}$".format(distance))
        ax.set_xlabel("")
        ax.set_xticks([])
    else:
        df["methods"] = df["methods"].astype(new_drop_custom_order)
        sns.boxplot(
            data=df,
            x="methods",
            y="MAE",
            hue="methods",
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
plt.savefig("results/figures/method_counts_all_MAE_without_asymp.pdf", format="pdf")
plt.show()

########### Supplementary material visualizations #############
stats_list = ["KS", "e_value", "LR", "BFF"]
model_list = ["lognormal", "gmm", "1dnormal"]

for stat in stats_list:
    for model in model_list:
        if stat != "BFF":
            colors = [
                "firebrick",
                "darkblue",
                "rebeccapurple",
                "darkgreen",
                "darkorange",
                "goldenrod",
            ]
            custom_palette = sns.color_palette(colors)

            new_custom_order = CategoricalDtype(
                [
                    "TRUST++ tuned",
                    "TRUST++ MV",
                    "TRUST",
                    "boosting",
                    "MC",
                    "asymptotic",
                ],
                ordered=True,
            )
        else:
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

        df_sel = combined_df.query("Statistic == @stat").query("Model == @model")
        df_sel["methods"] = df_sel["methods"].astype(new_custom_order)
        sns.set(style="ticks", font_scale=2)
        plt.rcParams.update({"font.size": 14})
        g = sns.FacetGrid(
            df_sel,
            col="N",
            col_wrap=2,
            height=6,
            aspect=1.50,
            hue="methods",
            palette=custom_palette,
            margin_titles=True,
            sharey=False,
        )
        g.map(plt.errorbar, "B", "MAE", "se", fmt="-o")
        g.add_legend()

        # savefigure
        plt.savefig(f"results/sup_figures/{stat}_{model}_MAE_plot.pdf", format="pdf")


######### Not used visualizations ##########
# Making also boxplots and violinplots graphs for
# numerical visualization
# Making dotplots separately according to n and B
sns.set(style="ticks", font_scale=2.75)
plt.rcParams.update({"font.size": 14})
colors = [
    "firebrick",
    "darkblue",
    "darkgreen",
    "rebeccapurple",
    "darkorange",
    "goldenrod",
]
custom_palette = sns.set_palette(sns.color_palette(colors))


# Create a facet grid using catplot
g = sns.catplot(
    data=combined_df,
    x="methods",
    y="MAE",
    kind="strip",
    hue="methods",
    col="N",
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
g.tick_params(axis="x", rotation=75)


# Adjust the layout to put the y label outside of the graph
g.fig.subplots_adjust(left=0.5)
g.fig.supylabel("Mean Absolute Error regarding nominal coverage", fontsize=35, x=-0.005)
plt.tight_layout()
g.savefig("results/figures/MAE_boxplots_N_B_sim.pdf", format="pdf")

# visualization without MC and asymptotic
g = sns.catplot(
    data=combined_df.query("methods not in ['MC', 'asymptotic']"),
    x="methods",
    y="MAE",
    kind="strip",
    hue="methods",
    col="N",
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
g.tick_params(axis="x", rotation=75)


# Adjust the layout to put the y label outside of the graph
g.fig.subplots_adjust(left=0.5)
g.fig.supylabel("Mean Absolute Error to nominal coverage", fontsize=35, x=-0.005)
plt.tight_layout()
g.savefig("results/figures/MAE_boxplots_N_B_sim_without_MC_asymp.pdf", format="pdf")


# Making violinplot of total MAE
sns.set(style="ticks", font_scale=2.75)
plt.rcParams.update({"font.size": 16})
colors = [
    "firebrick",
    "darkblue",
    "darkgreen",
    "rebeccapurple",
    "darkorange",
    "goldenrod",
]
custom_palette = sns.set_palette(sns.color_palette(colors))

plt.figure(figsize=(12, 8))
ax = sns.violinplot(
    data=combined_df,
    y="methods",
    x="MAE",
    palette=custom_palette,
    cut=0,
)
plt.title("Distribution of MAE for Each Method")
plt.ylabel("Method")
plt.xlabel("MAE")
plt.xticks(rotation=75)
# changing alpha
for patch in ax.collections:
    patch.set_alpha(0.65)
plt.tight_layout()
plt.savefig("results/figures/MAE_violinplot_overall_sim_data.pdf", format="pdf")
plt.show()

# making boxplot version
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=combined_df,
    y="methods",
    x="MAE",
    palette=custom_palette,
    boxprops=dict(alpha=0.65),
)
plt.title("Distribution of MAE for Each Method")
plt.ylabel("Method")
plt.xlabel("MAE")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig("results/figures/MAE_boxplot_overall_sim_data.pdf", format="pdf")
plt.show()
