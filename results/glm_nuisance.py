import numpy as np
from sklearn.datasets import make_regression
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st

from CSI.scores import LambdaScore
from CSI.loforest import ConformalLoforest
from CSI.locart import LocartSplit

# importing functions to adapt estimated cutoffs
from CSI.nuissance import TRUST_nuisance_cutoffs, TRUST_plus_nuisance_cutoff
from sklearn.ensemble import HistGradientBoostingRegressor

import torch
from CSI.simulations import GLM_stat
import itertools
from tqdm import tqdm

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

def naive_nuisance(
    kind,
    alpha,
    nuisance_idx,
    par_size = 6,
    B=1000,
    naive_n=100,
    glm_class = None,
):
    n_grid = int(B / naive_n)
    quantiles = {}

    if kind == "glm":
        idx_array = np.arange(0,par_size)
        par_idx = np.setdiff1d(idx_array, nuisance_idx)

        beta_b_nuis_space = np.linspace(-1.00, 1.00, 
                                        int(np.ceil(n_grid ** (1 / par_size))))
        beta_0_nuis_space = np.linspace(-1.5, 1.5,
                                        int(np.ceil(n_grid ** (1 / par_size))))
        phi_nuis_space = np.linspace(0.15,1.5,
                                     int(np.ceil(n_grid ** (1 / par_size))))
        
        par_list = [beta_0_nuis_space]

        for i in range(1, par_size - 1):
            par_list.append(beta_b_nuis_space)
        
        par_list.append(phi_nuis_space)
        
        par_array = np.c_[list(itertools.product(*par_list))]
        for i in tqdm(range(par_array.shape[0]), desc="Computing lambda values"):
            theta = par_array[i]
            lambdas = glm_class.LR_sim_lambda(
            beta_value = theta[:-1],
            phi_value = theta[-1],
            B = naive_n,
            idx_1 = par_idx,
        )
            quantiles[tuple(theta)] = np.quantile(lambdas, q=1 - alpha)
        
    return quantiles

# naive nuisance cutoff for GLM with more dimensions
def naive_nuisance_cutoff_glm(
        naive_quantiles,
        nuisance_grid,
        nuisance_idx,
        par_values,
        beta_dim = 5,
        ):
    # cutoff array
    cutoff_nuis = np.zeros(par_values.shape[0])
    idx_array = np.arange(0, (beta_dim + 1))

    # reordering parameters order
    par_idx = np.setdiff1d(idx_array, nuisance_idx)
    par_reorder = np.argsort(
            np.concatenate((par_idx, nuisance_idx), axis = None)
        )
    
    i = 0
    for par in tqdm(par_values, desc = "Computing nuisance cutoffs for each parameter value"):
        par_array = np.tile(par, reps = (nuisance_grid.shape[0], 1))
        new_par = np.column_stack((par_array, nuisance_grid))

        # reordering columns
        new_par = new_par[:, par_reorder]

        # computing cutoffs with naive
        cutoff_vector = np.array(predict_naive_quantile(kind = "glm", theta_grid = new_par, quantiles_dict = naive_quantiles))

        # returning minimal value
        cutoff_nuis[i] = np.max(cutoff_vector)
        i += 1
    return cutoff_nuis

# boosting nuisance cutoff
def boosting_nuisance_cutoff(
        boosting_obj,
        nuisance_grid,
        nuissance_idx,
        par_values,
        ):
    # cutoff array
    cutoff_nuis = np.zeros(par_values.shape[0])

    # returning all index
    size = par_values.shape[1] + 1
    # idx list
    idx_array = np.arange(0, size)
    if isinstance(nuissance_idx, int):
        par_idx = idx_array[np.where(idx_array != nuissance_idx)]
    else:
        par_idx = np.setdiff1d(idx_array, nuissance_idx)
        
    par_reorder = np.argsort(
        np.concatenate((par_idx, nuissance_idx), axis = None)
        )

    i = 0
    for par in tqdm(par_values, desc = "Computing nuisance cutoffs for each parameter value in boosting"):
        par_array = np.tile(par, reps = (nuisance_grid.shape[0], 1))
        new_par = np.column_stack((par_array, nuisance_grid))

        # reordering columns
        new_par = new_par[:, par_reorder]

        # computing cutoffs for new par
        cutoff_vector = boosting_obj.predict(new_par)
        if(i == 0):
            print(cutoff_vector)

        # returning minimal value
        cutoff_nuis[i] = np.max(cutoff_vector)
        i += 1
    return cutoff_nuis

# beta_0 parameters space and prior: N(0,1)
# all other beta parameters space and prior: N(0,0.5)
# phi parameter space and prior: truncated exponential with scale 0.5, truncated at 1.75
def prior(n, rng, intercept_value = None, dim = 4):
    if intercept_value is None:
        betas = rng.normal(loc = 
                       np.repeat(0, dim+1), 
                       scale = np.concatenate(
                           (np.array([1.0]), np.repeat(0.5, dim))
                           ),
                        size = (n, dim+1)
                       )
    else:
        betas = rng.normal(loc = 0,
                           scale = 0.5,
                           size = (n, dim))
        betas = np.column_stack((np.repeat(intercept_value, n),
                                 betas))
    
    # truncating exponential values at 1.75
    phi = rng.standard_exponential(n)
    phi[np.where(phi > 1.75)] = 1.75
    return betas, phi

# prior and X with more dimensions
n = 50
np.random.seed(45)
rng = np.random.default_rng(45)
# starting with 5 dimensions for betas, but will change it to 10 for further testing
beta_dim = 5
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

# fitting GLM class
glm_class = GLM_stat(
    prior_func=prior,
    X_mat = X_mat,
    rng = rng,
    dist = "gamma",
    link_func = "log",
)
glm_class

############# Training step ##############
print("Sampling from the parameter prior and simulating sample:")
B = 10000
# using GLM class
thetas_sim, model_lambdas = glm_class.LR_sample(
    B = B, 
    idx_1 = np.array([1]), 
    fit_intercept = True,
    )

if thetas_sim.ndim == 1:
    model_thetas = thetas_sim.reshape(-1, 1)
else:
    model_thetas = thetas_sim

print("Fitting our methods: ")
print("Fitting TRUST ")
trust_object = LocartSplit(
        LambdaScore, None, alpha=0.05, is_fitted=True, split_calib=False
    )
trust_quantiles = trust_object.calib(
    thetas_sim, model_lambdas, min_samples_leaf=150
)

# loforest quantiles
print("Fitting TRUST++")
trust_plus_object = ConformalLoforest(
    LambdaScore, None, alpha=0.05, is_fitted=True, split_calib=False
)
trust_plus_object.calibrate(
    thetas_sim,
    model_lambdas,
    min_samples_leaf=550,
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
boosting_object.fit(thetas_sim, model_lambdas)

nuisance_idx = np.delete(np.arange(0, beta_dim + 1), 1)
# training monte carlo
naive_quantiles = naive_nuisance(
    kind = "glm",
    alpha = 0.05,
    nuisance_idx = nuisance_idx,
    par_size = beta_dim + 1,
    B = B,
    glm_class = glm_class
)


############# Nuisance cutoffs for each method ##############
# validation grid
valid_rng = np.random.default_rng(67)
# fixing only beta_1 values
beta_nuis_space = np.linspace(-1.25, 1.25, 15)

beta_space, phi_space = prior(n = 50, rng = valid_rng)
# joining parameters together
valid_thetas = np.concatenate(
    (beta_space, phi_space.reshape(-1, 1)), axis=1)

valid_thetas_del = np.delete(valid_thetas, 1, axis = 1)

# obtaining combination
valid_tile = np.tile(valid_thetas_del, (15,1))
beta_1_tile = np.tile(beta_nuis_space, 50)
valid_thetas = np.insert(valid_tile, 1, beta_1_tile, axis = 1)


idx_1 = np.array([1])
total_grid_size = 140000
par_size = beta_dim + 1 - len(idx_1)

beta_b_nuis_space = np.linspace(-1.25, 1.25, 
                                int(np.ceil(total_grid_size ** (1 / par_size))))
beta_0_nuis_space = np.linspace(-1.5, 1.5,
                                int(np.ceil(total_grid_size ** (1 / par_size))))
phi_nuis_space = np.linspace(0.05,1.75,
                                int(np.ceil(total_grid_size ** (1 / par_size))))
par_list = [beta_0_nuis_space]
for i in range(1, par_size - 1):
    par_list.append(beta_b_nuis_space)
par_list.append(phi_nuis_space)

nuisance_grid_boosting = np.c_[list(itertools.product(*par_list))]

# obtaining cutoffs for each mu
cutoff_beta_TRUST = TRUST_nuisance_cutoffs(
    trust_object, 
    nuissance_idx = nuisance_idx,
    par_values = beta_nuis_space.reshape(-1, 1), 
    trust_quantiles = trust_quantiles, 
    )

cutoff_beta_TRUST_plus, max_TRUST_plus = TRUST_plus_nuisance_cutoff(
    trust_plus_object,
    nuissance_idx = nuisance_idx,
    par_values = beta_nuis_space.reshape(-1, 1),
    K = 100,
    strategy = "horizontal_cutoffs",
    total_h_cutoffs = 40,
)

cutoff_beta_boosting = boosting_nuisance_cutoff(
    boosting_object, 
    nuisance_grid = nuisance_grid_boosting, 
    nuissance_idx= nuisance_idx,
    par_values = beta_nuis_space.reshape(-1, 1),
    )

cutoff_beta_naive = naive_nuisance_cutoff_glm(
    naive_quantiles,
    nuisance_grid = nuisance_grid_boosting,
    nuisance_idx = nuisance_idx,
    par_values = beta_nuis_space.reshape(-1,1),
)

# Function for computing coverage and distance to oracle for evaluation grid
def oracle_dist_glm(coverage_array_trust, 
                coverage_array_trust_plus,
                coverage_boosting,
                coverage_asymp,
                coverage_naive,
                cutoff_real,
                valid_thetas,
                glm_class,
                par_idx,
                new_seed = 45,
                n_lambda = 1000,
                n_rep = 50,
                ):
    glm_class.change_seed(new_seed)
    mean_diff_trust, mean_diff_trust_plus = np.zeros(n_rep), np.zeros(n_rep)
    mean_diff_boosting= np.zeros(n_rep)
    mean_diff_asymp = np.zeros(n_rep)
    mean_diff_naive = np.zeros(n_rep)

    for i in tqdm(range(0, n_rep), desc = "Repeating computation: "):
        diff_trust, diff_trust_plus = np.zeros(valid_thetas.shape[0]), np.zeros(valid_thetas.shape[0])
        diff_boosting = np.zeros(valid_thetas.shape[0])
        diff_naive = np.zeros(valid_thetas.shape[0])
        diff_asymp = np.zeros(valid_thetas.shape[0])
        j = 0
        # computing coverage for trust, trust++ and boosting
        for theta in valid_thetas:
            stat = glm_class.LR_sim_lambda(
            beta_value = theta[:-1],
            phi_value = theta[-1],
            B = n_lambda,
            idx_1 = par_idx,
        )
            
            # real coverage
            oracle_coverage = np.mean(stat <= cutoff_real[j])
            
            diff_trust[j] = np.abs(coverage_array_trust[j] - oracle_coverage)
            diff_trust_plus[j] = np.abs(
                coverage_array_trust_plus[j] - oracle_coverage)
            diff_boosting[j] = np.abs(coverage_boosting[j] - oracle_coverage)
            diff_asymp[j] = np.abs(coverage_asymp[j] - oracle_coverage)
            diff_naive[j] = np.abs(coverage_naive[j] - oracle_coverage)
            j += 1

        mean_diff_trust[i] = np.mean(diff_trust)
        mean_diff_trust_plus[i] = np.mean(diff_trust_plus)
        mean_diff_boosting[i] = np.mean(diff_boosting)
        mean_diff_asymp[i] = np.mean(diff_asymp)
        mean_diff_naive[i] = np.mean(diff_naive)
    
    mean_diff_trust_overall = np.mean(mean_diff_trust)
    mean_diff_trust_plus_overall= np.mean(mean_diff_trust_plus)
    mean_diff_boosting_overall = np.mean(mean_diff_boosting)
    mean_diff_asymp_overall = np.mean(mean_diff_asymp)
    mean_diff_naive_overall = np.mean(mean_diff_naive)

    n_exp = np.sqrt(n_rep)
    se_trust = 2*np.std(mean_diff_trust)/n_exp
    se_trust_plus = 2*np.std(mean_diff_trust_plus)/n_exp
    se_boosting = 2*np.std(mean_diff_boosting)/n_exp
    se_naive = 2*np.std(mean_diff_naive)/n_exp
    se_asymp = 2*np.std(mean_diff_asymp)/n_exp


    diff_data = pd.DataFrame({
        "methods": ["TRUST", "TRUST++", "Boosting", "MC", "Asymptotic"],
        "diff": [
            mean_diff_trust_overall, 
            mean_diff_trust_plus_overall, 
            mean_diff_boosting_overall,
            mean_diff_naive_overall,
            mean_diff_asymp_overall,
            ],
        "se*2": [se_trust, se_trust_plus, se_boosting, se_naive, se_asymp]
    })
    
    return [diff_trust,
            diff_trust_plus,
            diff_boosting,
            diff_data
            ]

def coverage_nuisance_glm(cutoff_array_trust, 
                      cutoff_array_trust_plus,
                      cutoff_boosting,
                      cutoff_naive,
                      valid_thetas,
                      par_space,
                      glm_class,
                      par_idx,
                      n_lambda = 500,
                      alpha = 0.05):
    coverage_trust, coverage_trust_plus = np.zeros(
        valid_thetas.shape[0]), np.zeros(
            valid_thetas.shape[0])
    coverage_boosting = np.zeros(valid_thetas.shape[0])
    coverage_naive = np.zeros(valid_thetas.shape[0])
    cutoff_trust_list, cutoff_trust_plus_list = [], []
    real_cutoff_list, boosting_cutoff_list = [], []
    naive_cutoff_list = []
    stat_list = []
    i = 0

    # asymptotic cutoff and coverage vector
    coverage_asymp = np.zeros(valid_thetas.shape[0])
    cutoff_asymp = st.chi2.ppf(1 - alpha, df = par_idx.shape[0])
    
    # computing coverage for trust, trust++, boosting and naive
    for theta in tqdm(valid_thetas, desc = "Assessing "):
        cut_idx = np.where(par_space == theta[par_idx])
        # generating stats
        stat = glm_class.LR_sim_lambda(
            beta_value = theta[:-1],
            phi_value = theta[-1],
            B = n_lambda,
            idx_1 = par_idx,
        )
        
        real_cutoff = np.quantile(stat, q = 0.95)
        coverage_trust[i] = np.mean(stat <= cutoff_array_trust[cut_idx])
        coverage_trust_plus[i] = np.mean(stat <= cutoff_array_trust_plus[cut_idx])
        coverage_boosting[i] = np.mean(stat <= cutoff_boosting[cut_idx])
        coverage_asymp[i] = np.mean(stat <= cutoff_asymp)
        coverage_naive[i] = np.mean(stat <= cutoff_naive[cut_idx])
        
        # adding every cutoff in the list
        cutoff_trust_plus_list.append(cutoff_array_trust_plus[cut_idx])
        boosting_cutoff_list.append(cutoff_boosting[cut_idx])
        naive_cutoff_list.append(cutoff_naive[cut_idx])

        # returning cutoff for TRUST to evaluate
        cutoff_trust_list.append(cutoff_array_trust[cut_idx])
        real_cutoff_list.append(real_cutoff)
        i += 1
    
    # transforming the lists into arrays
    trust_grid_cutoff_array = np.array(cutoff_trust_list)
    trust_grid_plus_cutoff_array = np.array(cutoff_trust_plus_list)
    boosting_cutoff_array = np.array(boosting_cutoff_list)
    naive_cutoff_array = np.array(naive_cutoff_list)
    
    # computing real nuisance cutoff using real cutoffs from past loop
    real_cutoff =  np.array(real_cutoff_list)
    real_nuisance_cutoff_list = []
    for theta in tqdm(valid_thetas, desc = "Computing real cutoffs: "):
        cut_idxs = np.where(
            valid_thetas[:, par_idx] == theta[par_idx]
            )[0]
        real_nuisance_cutoff = np.max(real_cutoff[cut_idxs])
        real_nuisance_cutoff_list.append(real_nuisance_cutoff)
    
    # transforming the list into array
    real_grid_cutoff_array = np.array(real_nuisance_cutoff_list)
    
    # computing TRUST, TRUST++, Boosting and real cutoffs only for mu
    # in poisson exampleScreenshot from 2024-09-17 18-18-13
    real_mu_list, trust_mu_list, trust_plus_mu_list = [], [], []
    boosting_mu_list = []
    for mu in tqdm(par_space, desc = "Computing cutoffs only for mu: "):
        cut_idx = np.where(valid_thetas[:, 1] == mu)
        real_mu_nuisance_cutoff = np.unique(
            real_grid_cutoff_array[cut_idx])[0]
        trust_mu_nuisance_cutoff = np.unique(
            trust_grid_cutoff_array[cut_idx])[0]
        trust_plus_mu_nuisance_cutoff = np.unique(
            trust_grid_plus_cutoff_array[cut_idx])[0]
        boosting_mu_nuisance_cutoff = np.unique(
            boosting_cutoff_array[cut_idx])[0]
        
        

        boosting_mu_list.append(boosting_mu_nuisance_cutoff)
        real_mu_list.append(real_mu_nuisance_cutoff)
        trust_mu_list.append(trust_mu_nuisance_cutoff)
        trust_plus_mu_list.append(trust_plus_mu_nuisance_cutoff)

    # trust coverage, trus plus coverage, trust cutoff, nuisance cutoff, 
    # trust plus cutoff, real cutoff for mu, trust cutoff for mu and
    # trust plus cutoff for mu
        return [coverage_trust, 
        coverage_trust_plus,
        trust_grid_cutoff_array, 
        real_grid_cutoff_array,
        trust_grid_plus_cutoff_array,
        np.array(real_mu_list),
        np.array(trust_mu_list),
        np.array(trust_plus_mu_list),
        stat_list,
        # mu_list,
        # problem_cutoff,
        coverage_boosting, #9
        boosting_cutoff_array, #10
        coverage_asymp, #11
        coverage_naive, #12
        naive_cutoff_array, #13
        # coverage_naive, #14
        ]

nuisance_glm_list = coverage_nuisance_glm(
    cutoff_beta_TRUST, 
    cutoff_beta_TRUST_plus,
    cutoff_beta_boosting,
    cutoff_beta_naive,
    valid_thetas, 
    par_space = beta_nuis_space,
    par_idx = np.array([1]),
    glm_class = glm_class,
    n_lambda = 100,
    )

trust_plus_coverage = nuisance_glm_list[1]
trust_coverage = nuisance_glm_list[0]
boosting_coverage = nuisance_glm_list[9]
real_cutoffs = nuisance_glm_list[3]
asymp_coverage = nuisance_glm_list[11]
naive_coverage = nuisance_glm_list[12]

diff_list = oracle_dist_glm(
    trust_coverage,
    trust_plus_coverage,
    boosting_coverage,
    asymp_coverage,
    naive_coverage,
    real_cutoffs,
    valid_thetas,
    glm_class = glm_class,
    new_seed = 90,
    n_rep = 15,
    par_idx = np.array([1]),
    n_lambda = 100,
)
