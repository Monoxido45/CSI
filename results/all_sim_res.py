import pandas as pd
import os
import re
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
print(combined_df)

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
        "tuned LOFOREST": "Tuned TRUST++",
        "monte-carlo": "MC",
    }
)


# making general barplot for performance
filtered_data = combined_df.groupby(
    ["N", "B", "Model", "Statistic"], as_index=False
).apply(lambda df: df.nsmallest(n=1, columns="MAE", keep="all"))


for i in range(filtered_data.shape[0]):
    # selecting values for n, B, kind and stats
    series_values = filtered_data.iloc[i, :]
    n_best, B_best = series_values["N"], series_values["B"]
    model_best, stat_best = series_values["Model"], series_values["Statistic"]
    best_method = series_values["methods"]
    best_MAE, best_SE = series_values["MAE"], series_values["se"]

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
    idx_add = np.where(lim_inf <= lim_sup)
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
    ["Tuned TRUST++", "TRUST++ MV", "boosting", "TRUST", "MC", "asymptotic"],
    ordered=True,
)
method_counts = filtered_data.value_counts(["N", "B", "methods"])
method_counts_data = method_counts.reset_index()
method_counts_data["methods"] = method_counts_data["methods"].astype(new_custom_order)

# creating custom palette
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
    data=method_counts_data,
    x="methods",
    y="count",
    kind="bar",
    col="N",
    row="B",
    legend=True,
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
g.fig.supylabel("Number of times each method performed better", fontsize=35, x=-0.005)
plt.tight_layout()
count = 0
for ax in g.axes.flatten():
    if count >= 18:
        for idx in [0, 1, 3]:
            ax.get_xticklabels()[idx].set_fontweight("bold")
    count += 1

g.savefig("results/figures/all_comparissons.pdf", format="pdf")


# making now barplots for coverage distance
# avoiding BFF for comparing all methods to asymptotic
# first, for distance < 0.05
filtered_data_dist_05 = combined_df.query("MAE <= 0.05").query("Statistic != 'BFF'")
value_counts_05 = filtered_data_dist_05["methods"].value_counts()

# first, for distance < 0.035
filtered_data_dist_04 = combined_df.query("MAE <= 0.035").query("Statistic != 'BFF'")
value_counts_04 = filtered_data_dist_04["methods"].value_counts()

# first, for distance < 0.02
filtered_data_dist_03 = combined_df.query("MAE <= 0.02").query("Statistic != 'BFF'")
value_counts_03 = filtered_data_dist_03["methods"].value_counts()

# first, for distance < 0.01
filtered_data_dist_02 = combined_df.query("MAE <= 0.01").query("Statistic != 'BFF'")
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
    "darkgreen",
    "rebeccapurple",
    "darkorange",
    "goldenrod",
]
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
        for idx in [0, 1, 3]:
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
    "darkgreen",
    "rebeccapurple",
    "darkorange",
    "goldenrod",
]

new_drop_custom_order = CategoricalDtype(
    ["Tuned TRUST++", "TRUST++ MV", "boosting", "TRUST", "MC"],
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
        for idx in [0, 1, 3]:
            ax.get_xticklabels()[idx].set_fontweight("bold")
    count += 1
plt.tight_layout()
plt.savefig("results/figures/method_counts_all_MAE_without_asymp.pdf", format="pdf")
plt.show()


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
