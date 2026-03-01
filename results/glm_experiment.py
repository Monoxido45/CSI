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
import gc
import pickle

parser = ArgumentParser()
parser.add_argument("-beta_dim", "--beta_dim", type=int, default=5, help="number of beta parameters in the GLM model")
parser.add_argument("-naive_n", "--naive_n", type=int, default=500, help="number of samples to use for computing cutoffs with naive method")
parser.add_argument("-seed", "--seed", type=int, default=45, help="seed for random generator")
parser.add_argument("-alpha", "--alpha",type=float, default=0.05, help="miscoverage level for conformal prediction")
parser.add_argument("-n_rep", "--n_rep", type=int, default=30, help="number of repetitions for computing coverage MAE")
parser.add_argument("-n_samples", "--n_samples", type=int, default=30, help="number of samples of observed data")
parser.add_argument("-B", "--B", type=int, default=10000, help="number of samples to use for training the methods and computing cutoffs")
args = parser.parse_args()

beta_dim = args.beta_dim
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
        if par_size <= 10:
            beta_b_nuis_space = np.linspace(-1.5, 1.5, 
                                            int(np.ceil(n_grid ** (1 / par_size))))
            beta_0_nuis_space = np.linspace(-2.5, 2.5,
                                            int(np.ceil(n_grid ** (1 / par_size))))
            phi_nuis_space = np.linspace(0.05,1.65,
                                         int(np.ceil(n_grid ** (1 / par_size))))
        else:
            beta_b_nuis_space = np.linspace(-0.25, 0.25, 
                                            int(np.ceil(n_grid ** (1 / par_size))))
            beta_0_nuis_space = np.linspace(-0.65, 0.65,
                                            int(np.ceil(n_grid ** (1 / par_size))))
            phi_nuis_space = np.linspace(0.05,1.45,
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

# beta_0 parameters space and prior: N(0,4)
# all other beta parameters space and prior: N(0,1)
# if high dimensional, N(0,0.1) and N(0,0.015)
# phi parameter space and prior: truncated exponential with scale 1, truncated at 1.75
def prior(n, rng, intercept_value = None, dim = 4):
    if intercept_value is None:
        if dim <= 10:
            betas = rng.normal(loc = 
                       np.repeat(0, dim+1), 
                       scale = np.concatenate(
                           (np.sqrt(np.array([4])), np.sqrt(np.repeat(1, dim)))
                           ),
                        size = (n, dim+1)
                       )
        else:
            betas = rng.normal(loc = 
                       np.repeat(0, dim+1), 
                       scale = np.concatenate(
                           (np.array([0.25]), np.repeat(0.1, dim))
                           ),
                        size = (n, dim+1)
                       )
    else:
        if dim <= 10:
            betas = rng.normal(loc = 0,
                           scale = np.sqrt(1),
                           size = (n, dim))
            betas = np.column_stack((np.repeat(intercept_value, n),
                                 betas))
        else:
            betas = rng.normal(loc = 0,
                           scale = 0.1,
                           size = (n, dim))
            betas = np.column_stack((np.repeat(intercept_value, n),
                                 betas))
    
    # truncating exponential values at 1.75 for 10 dimensional
    if dim <= 10:
        phi = rng.standard_exponential(n)
        phi[np.where(phi > 1.75)] = 1.75
    else:
        phi = rng.standard_exponential(n)
        phi[np.where(phi > 1.5)] = 1.5
    return betas, phi

# prior and X with more dimensions
n = n_samples
np.random.seed(seed)
rng = np.random.default_rng(seed)
# starting with 5 dimensions for betas, but will change it to 10 for further testing
X_mat = rng.uniform(-1, 1, (n, beta_dim-1))

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
    n_valid = 1000
    beta_space, phi_space = prior(
        n = n_valid, 
        rng = valid_rng,
        dim = beta_dim - 1,
        )
    valid_thetas = np.concatenate(
        (beta_space, phi_space.reshape(-1, 1)), 
        axis=1,
        )
    
    # ensure glm_results dir and stat_file path
    out_dir = os.path.join(os.getcwd(), "glm_results")
    os.makedirs(out_dir, exist_ok=True)
    stat_file = os.path.join(out_dir, f"stat_list_beta_dim_{beta_dim}.pkl")

    if os.path.exists(stat_file):
        # load existing stat_list to avoid re-building
        with open(stat_file, "rb") as f:
            stat_list = pickle.load(f)
        print(f"Loaded existing stat_list from {stat_file}")
    else:
        # build and save stat_list for the validation thetas
        print(f"{stat_file} not found — building stat_list for validation thetas")
        stat_list = []
        for theta in tqdm(valid_thetas, desc="Computing lambda values for validation thetas"):
            stat = glm_class.LR_sim_lambda(
                beta_value=theta[:-1],
                phi_value=theta[-1],
                B=1000,
            )
            stat_list.append(stat)

        with open(stat_file, "wb") as f:
            pickle.dump(stat_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved stat_list to {stat_file}")


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
          
        nan_lambda = np.isnan(model_lambdas)
        sum_nan = np.sum(nan_lambda)
        if sum_nan > 0:
            print(f"Warning: simulated data has {sum_nan} nan values")
            model_lambdas = model_lambdas[~nan_lambda]
            model_thetas = model_thetas[~nan_lambda, :]


        print("Fitting TRUST ")
        trust_object = LocartSplit(
                LambdaScore, 
                None, 
                alpha=alpha, 
                is_fitted=True, 
                split_calib=False
            )
        locart_quantiles = trust_object.calib(
            model_thetas, 
            model_lambdas, 
            min_samples_leaf=300
        )
        idxs = trust_object.cart.apply(valid_thetas)
        cutoff_TRUST = [locart_quantiles[idx] for idx in idxs]
        
        del trust_object
        gc.collect()

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
            min_samples_leaf=550,
            n_estimators=200,
            K=100,
        )
        cutoff_TRUST_plus = trust_plus_object.compute_cutoffs(valid_thetas)
        del trust_plus_object
        gc.collect()

        # training boosting
        print("Fitting boosting")
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
        cutoff_boosting = boosting_object.predict(valid_thetas)
        del boosting_object
        gc.collect()
        

        # training monte carlo
        print("Fitting MC")
        if beta_dim <= 10:
            naive_quantiles = naive_method(
                kind = "glm",
                alpha = alpha,
                par_size = beta_dim + 1,
                B = B,
                glm_class = glm_class
            )
            cutoff_naive = predict_naive_quantile(
                kind = "glm",
                theta_grid = valid_thetas,
                quantiles_dict = naive_quantiles,
            )
            
            coverage_naive = np.zeros(valid_thetas.shape[0])
            
        asymp_quantiles = np.tile(stats.chi2.ppf(1 - alpha, df=beta_dim + 1), 
                                valid_thetas.shape[0])
        
        coverage_trust = np.zeros(valid_thetas.shape[0])
        coverage_trust_plus = np.zeros(valid_thetas.shape[0])
        coverage_boosting = np.zeros(valid_thetas.shape[0])
        coverage_asymp = np.zeros(valid_thetas.shape[0])

        for j in range(valid_thetas.shape[0]):
            stat = stat_list[j]
            cutoff_trust = cutoff_TRUST[j]
            cutoff_trust_plus = cutoff_TRUST_plus[j]
            cutoff_boosting_sel = cutoff_boosting[j]
            cutoff_asymp_sel = asymp_quantiles[j]
            if beta_dim <= 10:
                cutoff_naive_sel = cutoff_naive[j]
                coverage_naive[j] = np.mean(stat <= cutoff_naive_sel)
              
            coverage_trust[j] = np.mean(stat <= cutoff_trust)
            coverage_trust_plus[j] = np.mean(stat <= cutoff_trust_plus)
            coverage_boosting[j] = np.mean(stat <= cutoff_boosting_sel)
            coverage_asymp[j] = np.mean(stat <= cutoff_asymp_sel)

        mae_trust[i] = np.mean(np.abs(coverage_trust - (1-alpha)))
        mae_trust_plus[i] = np.mean(np.abs(coverage_trust_plus - (1-alpha)))
        mae_boosting[i] = np.mean(np.abs(coverage_boosting - (1-alpha)))
        if beta_dim <= 10:
            mae_naive[i] = np.mean(np.abs(coverage_naive - (1-alpha)))
        mae_asymp[i] = np.mean(np.abs(coverage_asymp - (1-alpha)))
        if beta_dim <= 10:
            print(
                f"Coverage TRUST: {mae_trust[i]}, Coverage TRUST++: {mae_trust_plus[i]}, Coverage Boosting: {mae_boosting[i]}, Coverage Naive: {mae_naive[i]}, Coverage Asymptotic: {mae_asymp[i]}"
            )
        else:
            print(
                f"Coverage TRUST: {mae_trust[i]}, Coverage TRUST++: {mae_trust_plus[i]}, Coverage Boosting: {mae_boosting[i]}, Coverage Asymptotic: {mae_asymp[i]}"
            )

    if beta_dim <= 10:
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
    else:
        methods = ["TRUST", "TRUST++", "Boosting", "Asymptotic"]
        means = [
            np.mean(mae_trust),
            np.mean(mae_trust_plus),
            np.mean(mae_boosting),
            np.mean(mae_asymp),
        ]
        sds = [
            2*np.std(mae_trust, ddof=1)/np.sqrt(n_rep),
            2*np.std(mae_trust_plus, ddof=1)/np.sqrt(n_rep),
            2*np.std(mae_boosting, ddof=1)/np.sqrt(n_rep),
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
outfile = os.path.join(out_dir, f"mae_df_beta_dim_{beta_dim}_B_{B}.pkl")
with open(outfile, "wb") as f:
    pickle.dump(mae_df, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved mae_df to {outfile}")
