#!/usr/bin/env python3
from pathlib import Path
import pickle
import sys
import pandas as pd
try:
    import joblib
except Exception:
    joblib = None
import matplotlib.pyplot as plt

path_list = ["mae_df_beta_dim_5_B_10000.pkl",
             "mae_df_beta_dim_10_B_10000.pkl",
             "mae_df_beta_dim_30_B_10000.pkl",
             "mae_df_beta_dim_50_B_10000.pkl",
             ]
dim_list = [5, 10, 30, 50]

def load_pickle(path: Path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f)

data_list = []
for path in path_list:
    PATH = Path("dim_comparisons/" + path)
    data_mae = load_pickle(PATH)
    data_mae["dim"] = dim_list[path_list.index(path)]
    data_list.append(data_mae)

df = pd.concat(data_list, ignore_index=True)

# choose the column containing MAE values
mae_col = "mean_MAE"

# detect a column that indicates the method; fallback to a single group if none found
method_col = "method"

# normalize method names: change "TRUST++" to "TRUST++ MV" (case-insensitive) and canonicalize common variants
df[method_col] = df[method_col].astype(str)
mask_trustpp = df[method_col].str.strip().str.lower() == "trust++"
mask_trustpp_mv = df[method_col].str.strip().str.lower() == "trust++ mv"
df.loc[mask_trustpp | mask_trustpp_mv, method_col] = "TRUST++ MV"

# mapping of canonical method names to the requested colors (keys are lowercased for case-insensitive lookup)
_color_map_raw = {
    "TRUST++ MV": "darkblue",
    "TRUST": "rebeccapurple",
    "Boosting": "darkgreen",
    "MC": "darkorange",
    "Asymptotic": "goldenrod",
}
color_map = {k.strip().lower(): v for k, v in _color_map_raw.items()}
# use existing mean and 2*se columns directly (no aggregation)
stats = df.copy()
stats["mean_mae"] = stats[mae_col]
# prefer the existing '2*se' column; if missing, fall back to 2*sem per group
stats["err"] = stats["2*se"]

# Increase default font sizes and line/marker sizes for all plots
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16,
})

# line plots with error bars, one line per method
plt.figure(figsize=(8, 5))
for method, sub in stats.groupby(method_col):
    sub = sub.sort_values("dim")
    key = str(method).strip().lower()
    # choose color by exact match or by substring match using lowercase keys; fallback to matplotlib default
    color = color_map.get(key)
    if color is None:
        for k, v in color_map.items():
            if k in key:
                color = v
                break

    plt.errorbar(
        sub["dim"],
        sub["mean_mae"],
        yerr=sub["err"],
        fmt="-o",
        capsize=4,
        label=str(method),
        color=color,
    )

plt.xlabel("Dimension")
plt.ylabel("Coverage MAE")
plt.title("Mean MAE vs Dimension")
plt.grid(True)
plt.legend(title="Methods")
plt.tight_layout()
plt.savefig("results_dim_comparison.png", dpi=500)
plt.show()
plt.close()

if not PATH.exists():
    print(f"File not found: {PATH}", file=sys.stderr)

obj = load_pickle(PATH)
print(repr(obj))

# sensitivity analysis for two moons example
sens_path = Path("sensitivity_results/sensitivity_hyperpar_summary.csv")
sens_data = pd.read_csv(sens_path)

# prepare sensitivity plotting for TRUST++ hyperparameters (Parameter, Value, MAE_mean, MAE_std, Time_mean, Time_std)
df_s = sens_data.copy()

params = ["max_depth", "min_samples_leaf", "n_estimators"]
# plotting grid: 3 rows x 2 cols, share y per column
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharey="col")

last_row = len(params) - 1
for row_idx, param in enumerate(params):
    ax_mae = axes[row_idx, 0]
    ax_time = axes[row_idx, 1]

    # filter rows for the current parameter
    sub = df_s[df_s["Parameter"] == param].copy()
    param_label = param.replace("_", " ")
    if sub.empty:
        ax_mae.text(0.5, 0.5, f"No data for {param_label}", ha="center", va="center")
        ax_time.text(0.5, 0.5, f"No data for {param_label}", ha="center", va="center")
        if row_idx == last_row:
            ax_mae.set_xlabel("Hyperparameter values")
            ax_time.set_xlabel("Hyperparameter values")
        continue

    # try to get numeric x values from Value, otherwise use as categorical
    x_num = pd.to_numeric(sub["Value"], errors="coerce")
    if not x_num.isna().all():
        sub["_x"] = x_num
        sub = sub.sort_values("_x")
        x = sub["_x"].values
    else:
        sub = sub.sort_values("Value")
        x = sub["Value"].values

    # y and error columns (use provided std as errorbars)
    y_mae = sub["MAE_mean"].values
    err_mae = sub["MAE_std"].values if "MAE_std" in sub.columns else None
    y_time = sub["Time_mean"].values
    err_time = sub["Time_std"].values if "Time_std" in sub.columns else None

    color = "darkblue"  # TRUST++ color

    ax_mae.errorbar(x, y_mae, yerr=err_mae, fmt="-o", capsize=4, color=color)
    ax_time.errorbar(x, y_time, yerr=err_time, fmt="-o", capsize=4, color=color)

    # titles
    ax_mae.set_title(f"{param_label} — MAE")
    ax_time.set_title(f"{param_label} — Time")

    # set shared y labels only on middle row for clarity
    if row_idx == 1:
        ax_mae.set_ylabel("MAE")
        ax_time.set_ylabel("Time (s)")

    # set x labels only on the last row, with generic text "Parameter"
    if row_idx == last_row:
        ax_mae.set_xlabel("Hyperparameter values")
        ax_time.set_xlabel("Hyperparameter values")

# single legend is unnecessary (single method), but add one if desired:
# fig.legend(["TRUST++"], loc="upper center")

plt.tight_layout()
plt.savefig("sensitivity_analysis.png", dpi=500)
plt.show()



