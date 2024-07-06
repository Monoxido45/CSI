import pandas as pd
import os
import re
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt

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
        temp_df = pd.read_csv(file_path)

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
    ["LOCART", "LOFOREST", "tuned LOFOREST", "boosting", "monte-carlo", "asymptotic"],
    ordered=True,
)

combined_df["methods"] = combined_df["methods"].astype(method_custom_order)
combined_df["methods"] = combined_df["methods"].cat.rename_categories(
    {"LOCART": "TRUST", "LOFOREST": "TRUST++", "tuned LOFOREST": "Tuned TRUST++"}
)


# making general barplot
filtered_data = combined_df.groupby(
    ["N", "B", "Model", "Statistic"], as_index=False
).apply(lambda df: df.nsmallest(n=1, columns="MAE", keep="all"))
method_counts = filtered_data.value_counts(["N", "B", "methods"])
method_counts_data = method_counts.reset_index()

# Set the style of the plot
sns.set(style="ticks", font_scale=2.75)

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

# Assuming 'g' is the figure you want to save
g.savefig("results/figures/all_comparissons.pdf", format="pdf")
