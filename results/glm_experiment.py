import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
from argparse import ArgumentParser

from CSI.scores import LambdaScore
from CSI.loforest import ConformalLoforest
from CSI.locart import LocartSplit
from scipy import stats

# importing functions to adapt estimated cutoffs
from sklearn.ensemble import HistGradientBoostingRegressor

from CSI.simulations import GLM_stat
import itertools
from tqdm import tqdm
import os
import pickle

parser = ArgumentParser()
parser.add_argument("-beta_dim", "--beta_dim", type=int, default=10, help="number of beta parameters in the GLM model")
parser.add_argument("-total_h_cutoffs", "--total_h_cutoffs", type=int, default=30, help="number of horizontal cutoffs to use for TRUST++")
parser.add_argument("-total_grid_size", "--total_grid_size", type=int, default=40000, 
                    help="total grid size to use for computing cutoffs with boosting and naive method")
parser.add_argument("-naive_n", "--naive_n", type=int, default=500, help="number of samples to use for computing cutoffs with naive method")
parser.add_argument("-seed", "--seed", type=int, default=75, help="seed for random generator")
parser.add_argument("-alpha", "--alpha",type=float, default=0.05, help="miscoverage level for conformal prediction")
parser.add_argument("-n_rep", "--n_rep", type=int, default=15, help="number of repetitions for computing coverage MAE")
parser.add_argument("-n_samples", "--n_samples", type=int, default=50, help="number of samples of observed data")
parser.add_argument("-B", "--B", type=int, default=1000, help="number of samples to use for training the methods and computing cutoffs")
args = parser.parse_args()

beta_dim = args.beta_dim
total_h_cutoffs = args.total_h_cutoffs
total_grid_size = args.total_grid_size
naive_n = args.naive_n
seed = args.seed
alpha = args.alpha
n_rep = args.n_rep
n_samples = args.n_samples
B = args.B

# naive nuisance cutoff for GLM
def predict_naive_quantile(kind, theta_grid, quantiles_dict):
    thetas_values = np.array(list(quantiles_dict.keys()))
    quantiles_list = []
    for theta in theta_grid:
        if kind == "weinberg":
            idx = thetas_values[int(np.argmin(np.abs(theta - thetas_values)))]
            quantiles_list.append(quantiles_dict[idx])
        else:
            distances = np.linalg.norm(thetas_values - theta, axis=1)
            idx = thetas_values[np.argmin(distances)]
            quantiles_list.append(quantiles_dict[tuple(idx)])
    return quantiles_list

def naive_method(
    kind,
    alpha,
    par_size = 6,
    B=1000,
    naive_n=500,
    glm_class = None,
):
    n_grid = int(B / naive_n)
    quantiles = {}

    if kind == "glm":
        beta_b_nuis_space = np.linspace(-1.00, 1.00, 
                                        int(np.ceil(n_grid ** (1 / par_size))))
        beta_0_nuis_space = np.linspace(-1.5, 1.5,
                                        int(np.ceil(n_grid ** (1 / par_size))))
        phi_nuis_space = np.linspace(0.05,1.675,
                                     int(np.ceil(n_grid ** (1 / par_size))))
        
        par_list = [beta_0_nuis_space]

        for i in range(1, par_size - 1):
            par_list.append(beta_b_nuis_space)
        
        par_list.append(phi_nuis_space)
        
        par_array = np.c_[list(itertools.product(*par_list))]
        
        # guaranteeing fair comparison by correcting the number of samples of the MC method
        n_new = np.ceil(B/par_array.shape[0])
        # avoiding surpassing the total budget
        for i in tqdm(range(par_array.shape[0]), desc="Computing lambda values"):
            theta = par_array[i]
            lambdas = glm_class.LR_sim_lambda(
            beta_value = theta[:-1],
            phi_value = theta[-1],
            B = n_new,
        )
            quantiles[tuple(theta)] = np.quantile(lambdas, q=1 - alpha)
        
    return quantiles

# beta_0 parameters space and prior: N(0,2)
# all other beta parameters space and prior: N(0,0.5)
# phi parameter space and prior: truncated exponential with scale 1, truncated at 1.75
def prior(n, rng, intercept_value = None, dim = 4):
    if intercept_value is None:
        betas = rng.normal(loc = 
                       np.repeat(0, dim+1), 
                       scale = np.concatenate(
                           (np.sqrt(np.array([2.0])), np.sqrt(np.repeat(0.5, dim)))
                           ),
                        size = (n, dim+1)
                       )
    else:
        betas = rng.normal(loc = 0,
                           scale = np.sqrt(0.5),
                           size = (n, dim))
        betas = np.column_stack((np.repeat(intercept_value, n),
                                 betas))
    
    # truncating exponential values at 1.75
    phi = rng.standard_exponential(n)
    phi[np.where(phi > 1.75)] = 1.75
    return betas, phi

# prior and X with more dimensions
n = n_samples
np.random.seed(seed)
rng = np.random.default_rng(seed)
# starting with 5 dimensions for betas, but will change it to 10 for further testing
X_mat = rng.uniform(-1, 1, (n, beta_dim-1))

# Generate response variable with Gamma noise
y = rng.gamma(
    shape=2, 
    scale=(1/2)*np.exp(0.5 * X_mat[:, 0] + 0.15 * np.mean(X_mat[:, 1:], axis=1)), 
    size=n)
X_new = sm.add_constant(X_mat)

# Fit a GLM model with Gamma distribution and log link function
glm_gamma = sm.GLM(y, X_new, family=sm.families.Gamma(link=sm.families.links.log()), )
result = glm_gamma.fit()

# Summarize the results
print(result.summary())

######################### Computing cutoffs for each method and each mu value in the validation grid #########################

def compute_coverage_MAE(
        n_rep = 15,
        seed = 45,
        B = 10000,
):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    valid_rng = np.random.default_rng(67)

    glm_class = GLM_stat(
    prior_func=prior,
    X_mat = X_mat,
    rng = rng,
    dist = "gamma",
    link_func = "log",
    )
    
    # random validation grid
    n_valid = 300
    beta_space, phi_space = prior(
        n = n_valid, 
        rng = valid_rng,
        dim = beta_dim - 1,
        )
    valid_thetas = np.concatenate(
        (beta_space, phi_space.reshape(-1, 1)), 
        axis=1,
        )

    stat_list = []
    # constructing the stats list
    print("Constructing stats list for validation thetas")
    for theta in tqdm(valid_thetas, desc="Computing lambda values for validation thetas"):
        stat = glm_class.LR_sim_lambda(
            beta_value = theta[:-1],
            phi_value = theta[-1],
            B = 1000,
        )
        stat_list.append(stat)

    mae_trust, mae_trust_plus = np.zeros(n_rep), np.zeros(n_rep)
    mae_boosting = np.zeros(n_rep)
    mae_naive = np.zeros(n_rep)
    mae_asymp = np.zeros(n_rep)
    for i in tqdm(range(0, n_rep), desc = "Repeating computation: "):
        # using GLM class
        thetas_sim, model_lambdas = glm_class.LR_sample(
            B = B, 
            fit_intercept = True,
            )

        if thetas_sim.ndim == 1:
            model_thetas = thetas_sim.reshape(-1, 1)
        else:
            model_thetas = thetas_sim

        print("Fitting TRUST ")
        trust_object = LocartSplit(
                LambdaScore, 
                None, 
                alpha=alpha, 
                is_fitted=True, 
                split_calib=False
            )
        trust_object.calib(
            model_thetas, model_lambdas, min_samples_leaf=150
        )

        # loforest quantiles
        print("Fitting TRUST++")
        trust_plus_object = ConformalLoforest(
            LambdaScore, 
            None, 
            alpha=alpha, 
            is_fitted=True, 
            split_calib=False,
        )
        trust_plus_object.calibrate(
            model_thetas,
            model_lambdas,
            min_samples_leaf=300,
            n_estimators=200,
            K=100,
        )

        # training boosting
        boosting_object =  HistGradientBoostingRegressor(
                loss="quantile",
                max_iter=100,
                max_depth=3,
                quantile= 0.95,
                random_state=105,
                n_iter_no_change=15,
                early_stopping=True,
            )
        boosting_object.fit(model_thetas, model_lambdas)

        # training monte carlo
        naive_quantiles = naive_method(
            kind = "glm",
            alpha = alpha,
            par_size = beta_dim + 1,
            B = B,
            glm_class = glm_class
        )

        cutoff_TRUST = trust_plus_object.compute_cutoffs(valid_thetas)
        cutoff_TRUST_plus = trust_plus_object.compute_cutoffs(valid_thetas)
        cutoff_boosting = boosting_object.predict(valid_thetas)
        cutoff_naive = predict_naive_quantile(
            kind = "glm",
            theta_grid = valid_thetas,
            quantiles_dict = naive_quantiles,
        )
        asymp_quantiles = np.tile(stats.chi2.ppf(1 - alpha, df=beta_dim + 1), 
                                valid_thetas.shape[0])
        
        coverage_trust = np.zeros(valid_thetas.shape[0])
        coverage_trust_plus = np.zeros(valid_thetas.shape[0])
        coverage_boosting = np.zeros(valid_thetas.shape[0])
        coverage_naive = np.zeros(valid_thetas.shape[0])
        coverage_asymp = np.zeros(valid_thetas.shape[0])

        for j in range(valid_thetas.shape[0]):
            stat = stat_list[j]
            cutoff_trust = cutoff_TRUST[j]
            cutoff_trust_plus = cutoff_TRUST_plus[j]
            cutoff_boosting = cutoff_boosting[j]
            cutoff_naive = cutoff_naive[j]
            cutoff_asymp = asymp_quantiles[j]

            coverage_trust[j] = np.mean(stat <= cutoff_trust)
            coverage_trust_plus[j] = np.mean(stat <= cutoff_trust_plus)
            coverage_boosting[j] = np.mean(stat <= cutoff_boosting)
            coverage_naive[j] = np.mean(stat <= cutoff_naive)
            coverage_asymp[j] = np.mean(stat <= cutoff_asymp)

        mae_trust[i] = np.mean(np.abs(coverage_trust - 0.95))
        mae_trust_plus[i] = np.mean(np.abs(coverage_trust_plus - 0.95))
        mae_boosting[i] = np.mean(np.abs(coverage_boosting - 0.95))
        mae_naive[i] = np.mean(np.abs(coverage_naive - 0.95))
        mae_asymp[i] = np.mean(np.abs(coverage_asymp - 0.95))

    
    methods = ["TRUST", "TRUST++", "Boosting", "MC", "Asymptotic"]
    means = [
        np.mean(mae_trust),
        np.mean(mae_trust_plus),
        np.mean(mae_boosting),
        np.mean(mae_naive),
        np.mean(mae_asymp),
    ]
    sds = [
        2*np.std(mae_trust, ddof=1)/np.sqrt(n_rep),
        2*np.std(mae_trust_plus, ddof=1)/np.sqrt(n_rep),
        2*np.std(mae_boosting, ddof=1)/np.sqrt(n_rep),
        2*np.std(mae_naive, ddof=1)/np.sqrt(n_rep),
        2*np.std(mae_asymp, ddof=1)/np.sqrt(n_rep),
    ]

    mae_df = pd.DataFrame({
        "method": methods,
        "mean_MAE": means,
        "2*se": sds,
    })

    print(mae_df)
    return mae_df

mae_df= compute_coverage_MAE(
    n_rep = n_rep,
    seed = seed,
    B = B
)

current_dir = os.getcwd()

out_dir = os.path.join(current_dir, "results/nuisance_results")
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, f"mae_df_beta_dim_{beta_dim}.pkl")
with open(outfile, "wb") as f:
    pickle.dump(mae_df, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved mae_df to {outfile}")