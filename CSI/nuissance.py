import numpy as np
import itertools
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


def TRUST_nuisance_cutoffs(
    trust_obj,
    nuissance_idx,
    par_values,
    trust_quantiles,
):
    """
    Computes the nuisance cutoffs for a given TRUST object.
    Parameters:
    -----------
    trust_obj : object
        An object containing the CART model and other relevant information.
    nuissance_idx : int or array-like
        Index or indices of the nuisance parameters.
    par_values : array-like
        Array of parameter values for which the cutoffs are to be computed.
    trust_quantiles : array-like
        Array of quantiles corresponding to the TRUST object.
    Returns:
    --------
    cutoff_nuis : numpy.ndarray
        Array of computed nuisance cutoffs.
    Notes:
    ------
    This function computes the nuisance cutoffs by analyzing the thresholds
    of the CART model in the TRUST object. It handles both single and multiple
    nuisance parameters and computes the cutoffs accordingly. If the nuisance
    parameter does not affect the TRUST object, it returns the quantiles directly.
    """

    # returning all thresholds
    feature_array = np.array(trust_obj.cart.tree_.feature)
    threshold_array = np.array(trust_obj.cart.tree_.threshold)

    # cutoff array
    cutoff_nuis = np.zeros(par_values.shape[0])

    # returning all index
    size = par_values.shape[1] + 1

    # idx list
    idx_array = np.arange(0, size)

    # selecting threshold_array with feature array equal to nuissance idx
    if isinstance(nuissance_idx, int):
        par_idx = idx_array[np.where(idx_array != nuissance_idx)]
        par_reorder = np.argsort(np.concatenate((par_idx, nuissance_idx), axis=None))

        idxs = np.where(feature_array == nuissance_idx)
        thres = threshold_array[idxs]

        if thres.shape[0] != 0:
            # choosing eps by pairwise distances
            dists = pairwise_distances(thres.reshape(-1, 1))
            upper_triangular_indices = np.triu_indices(dists.shape[0], k=1)
            eps = np.min(dists[upper_triangular_indices]) * 1 / 3

            nuis_values = np.concatenate((thres - eps, thres + eps), axis=None)

            i = 0
            for par in par_values:
                par_array = np.tile(par, reps=(nuis_values.shape[0], 1))
                new_par = np.column_stack((par_array, nuis_values))

                # reordering columns
                new_par = new_par[:, par_reorder]

                # computing cutoffs for new par
                idxs = trust_obj.cart.apply(new_par)
                list_locart_quantiles = np.array([trust_quantiles[idx] for idx in idxs])

                # returning minimal value
                cutoff_nuis[i] = np.max(list_locart_quantiles)
                i += 1
        # if thres is empty, then TRUST does not depend on nuisance
        else:
            i = 0
            for par in par_values:
                new_par = np.column_stack((par, 0))
                idxs = trust_obj.cart.apply(new_par)
                list_locart_quantiles = np.array([trust_quantiles[idx] for idx in idxs])

                cutoff_nuis[i] = list_locart_quantiles[0]
                i += 1

    else:
        # finding index of parameter of interest
        par_idx = np.setdiff1d(idx_array, nuissance_idx)
        par_reorder = np.argsort(np.concatenate((par_idx, nuissance_idx), axis=None))

        # finding indexes for nuisance threshold
        idxs_thres = np.where(np.isin(feature_array, nuissance_idx))
        thres = threshold_array[idxs_thres]

        # if there are any nuisance thresholds
        if thres.shape[0] != 0:
            feature_nuis = feature_array[idxs_thres]

            # looping through all analysed features
            feature_idx = np.unique(feature_nuis)
            used_idx_list = []
            nuis_list = []

            # looping through features
            for feature in feature_idx:
                idx = np.where(feature_nuis == feature)
                thres_sel = thres[idx]

                used_idx_list.append(np.where(nuissance_idx == feature)[0][0])

                if thres_sel.shape[0] > 1:
                    # choosing eps in this case
                    dists = pairwise_distances(thres_sel.reshape(-1, 1))
                    upper_triangular_indices = np.triu_indices(dists.shape[0], k=1)
                    eps = np.min(dists[upper_triangular_indices]) * 1 / 3
                else:
                    eps = 0.05

                nuis_values = np.concatenate(
                    (thres_sel - eps, thres_sel + eps), axis=None
                )
                nuis_list.append(nuis_values)

            # array of used indexes
            used_idx_array = np.array(used_idx_list)

            # checking if any nuisance feature was not used as threshold
            # and fixing a value to it
            idx_diff = np.setdiff1d(nuissance_idx, feature_idx)
            unused_idx_list = []
            if idx_diff.shape[0] != 0:
                for i in range(idx_diff.shape[0]):
                    unused_idx_list.append(np.where(nuissance_idx == idx_diff[i])[0][0])
                    nuis_list.append(np.array([0]))
                # array of unused indexes
                unused_idx_array = np.array(unused_idx_list)

                # reordering all combinations
                all_nuis_combinations = np.c_[list(itertools.product(*nuis_list))]

                # correcting order
                nuisance_reorder = np.argsort(
                    np.concatenate((used_idx_array, unused_idx_array), axis=None)
                )

                nuis_values = all_nuis_combinations[:, nuisance_reorder]
            else:
                nuis_values = np.c_[list(itertools.product(*nuis_list))]

            i = 0
            for par in par_values:
                par_array = np.tile(par, reps=(nuis_values.shape[0], 1))
                new_par = np.column_stack((par_array, nuis_values))

                # reordering columns
                new_par = new_par[:, par_reorder]

                # computing cutoffs for new par
                idxs = trust_obj.cart.apply(new_par)
                list_locart_quantiles = np.array([trust_quantiles[idx] for idx in idxs])

                # returning minimal value
                cutoff_nuis[i] = np.max(list_locart_quantiles)
                i += 1

        # conditioning if the threshold does not depend on other parameters
        else:
            # make vector of zeroes for the other parameters
            zeroes = np.zeros((1, nuissance_idx.shape[0]))
            i = 0
            for par in par_values:
                if par.ndim == 0:
                    par_col = par.reshape(1, 1)
                else:
                    par_col = par.reshape(1, -1)

                new_par = np.column_stack((par_col, zeroes))
                # reordering columns
                new_par = new_par[:, par_reorder]

                idxs = trust_obj.cart.apply(new_par)
                list_locart_quantiles = np.array([trust_quantiles[idx] for idx in idxs])

                cutoff_nuis[i] = list_locart_quantiles[0]
                i += 1
    return cutoff_nuis

def TRUST_plus_nuisance_cutoff(
    trust_plus_obj,
    nuissance_idx,
    par_values,
    K=100,
    strategy="all_cutoffs",
    total_h_cutoffs=50,
    compute_CI = False,
    alpha_CI = 0.05,
    by_batch = True,
    use_jax = True,
):
    """
    Computes nuisance cutoffs for a given TRUST+ object based on the specified strategy.

    Parameters:
    -----------
    trust_plus_obj : object
        An instance of the TRUST+ class containing the random forest model and other necessary attributes.
    nuissance_idx : int or list of int
        Index or indices of the nuisance features for which cutoffs are to be computed.
    par_values : numpy.ndarray
        Array of parameter values for which nuisance cutoffs are to be computed.
    K : int, optional
        Number of cutoffs to compute for each parameter value (default is 100).
    strategy : str, optional
        Strategy to use for computing cutoffs. Options are "all_cutoffs", "one_tree", and "horizontal_cutoffs" (default is "all_cutoffs").
    total_h_cutoffs : int, optional
        Total number of horizontal cutoffs to compute when using the "horizontal_cutoffs" strategy (default is 50).

    Returns:
    --------
    cutoff_nuis : numpy.ndarray
        Array of computed nuisance cutoffs for each parameter value.
    max_list : list
        List of parameter values corresponding to the maximum cutoffs.

    Notes:
    ------
    - The function supports three strategies for computing cutoffs:
        1. "all_cutoffs": Computes all thresholds for each tree in the random forest.
        2. "one_tree": Computes thresholds from a single nuisance tree.
        3. "horizontal_cutoffs": Computes a specified number of horizontal cutoffs across all trees.
    - The function uses pairwise distances to determine epsilon values for nuisance candidates.
    - The function prints intermediate results for debugging purposes.
    """
    n_trees = len(trust_plus_obj.RF.estimators_)
    threshold_list = []
    feature_list = []

    if compute_CI:
        lower_bound_nuis, upper_bound_nuis = (
                np.zeros(par_values.shape[0]), 
                np.zeros(par_values.shape[0])
            )

    if strategy == "all_cutoffs":
        # returning all thresholds for each tree
        for tree in trust_plus_obj.RF.estimators_:
            feature_array = np.array(tree.tree_.feature)
            threshold_array = np.array(tree.tree_.threshold)

            idxs = np.where(feature_array == nuissance_idx)
            thres = threshold_array[idxs]
            feature_nuis = feature_array[idxs]

            threshold_list.extend(list(thres))
            feature_list.extend(list(feature_nuis))
        thres = np.array(threshold_list)

    elif strategy == "one_tree":
        feature_array = np.array(trust_plus_obj.nuisance_tree.tree_.feature)
        threshold_array = np.array(trust_plus_obj.nuisance_tree.tree_.threshold)
        idxs = np.where(feature_array == nuissance_idx)
        thres = threshold_array[idxs]

    elif strategy == "horizontal_cutoffs":
        total_cutoffs = 0
        tree_position = 0
        # making list of features and thresholds
        tree_feature_list, tree_thres_list = [], []
        feature_list, thres_list = [], []
        # creating feature and tree threshold list
        for tree in trust_plus_obj.RF.estimators_:
            feature_array = np.array(tree.tree_.feature)
            threshold_array = np.array(tree.tree_.threshold)

            tree_feature_list.append(feature_array)
            tree_thres_list.append(threshold_array)

        # returning only the first thresholds in trees
        while total_cutoffs < total_h_cutoffs:
            for i in range(n_trees):
                curr_feature = tree_feature_list[i][tree_position]
                if isinstance(nuissance_idx, int):
                    if curr_feature == nuissance_idx:
                        thres_list.append(tree_thres_list[i][tree_position])
                        feature_list.append(curr_feature)
                        total_cutoffs += 1
                        if total_cutoffs == total_h_cutoffs:
                            break
                else:
                    if curr_feature in nuissance_idx:
                        thres_list.append(tree_thres_list[i][tree_position])
                        feature_list.append(curr_feature)
                        total_cutoffs += 1
                        if total_cutoffs == total_h_cutoffs:
                            break
            tree_position += 1
        thres = np.array(thres_list)
        feature_nuis = np.array(feature_list)

    # cutoff array
    cutoff_nuis = np.zeros(par_values.shape[0])

    # returning all index
    size = par_values.shape[1] + 1
    # idx list
    idx_array = np.arange(0, size)

    # selecting threshold_array with feature array equal to nuissance idx
    if isinstance(nuissance_idx, int):
        par_idx = idx_array[np.where(idx_array != nuissance_idx)]
        par_reorder = np.argsort(np.concatenate((par_idx, nuissance_idx), axis=None))

        # choosing eps by pairwise distances
        dists = pairwise_distances(thres.reshape(-1, 1))
        upper_triangular_indices = np.triu_indices(dists.shape[0], k=1)

        nuis_values = np.concatenate((thres - eps, thres + eps), axis=None)
        print(f"Using {nuis_values.shape[0]} nuisance candidates")

        par_idx = idx_array[np.where(idx_array != nuissance_idx)]
        par_reorder = np.concatenate((par_idx, nuissance_idx), axis=None)

        i = 0
        max_list = []
        for par in tqdm(
            par_values, desc="Computing nuisance cutoffs for each parameter value"
        ):
            
            par_array = np.tile(par, reps=(nuis_values.shape[0], 1))
            new_par = np.column_stack((par_array, nuis_values))

            # reordering columns
            new_par = new_par[:, par_reorder]

            if compute_CI:
                par_CI = new_par[max_idx, :].reshape(1, -1)
                _, lower_bound, upper_bound = (
                    trust_plus_obj.compute_cutoffs(
                    par_CI, 
                    K = K, 
                    compute_CI = compute_CI,
                    alpha_CI = alpha_CI,
                    by_batch = by_batch,
                    use_jax = use_jax,
                    )
                )
                lower_bound_nuis[i] = lower_bound[0]
                upper_bound_nuis[i] = upper_bound[0]

            # computing cutoffs for new par
            cutoff_vector = trust_plus_obj.compute_cutoffs(
                new_par, 
                K=K,
                by_batch = by_batch,
                use_jax = use_jax,
                )

            # index of max
            max_idx = np.argmax(cutoff_vector)
            max_list.append(new_par[max_idx, :])

            # returning minimal value
            cutoff_nuis[i] = np.max(cutoff_vector)
            
            i += 1
    else:
        # reordering parameters order
        par_idx = np.setdiff1d(idx_array, nuissance_idx)
        par_reorder = np.argsort(np.concatenate((par_idx, nuissance_idx), axis=None))

        # looping through all used nuisance features
        feature_idx = np.unique(feature_nuis)
        print(f"Using {feature_idx.shape[0]} nuisance features")

        nuis_list = []
        used_idx_list = []
        for feature in feature_idx:
            idxs = np.where(feature_nuis == feature)
            used_idx_list.append(np.where(nuissance_idx == feature)[0][0])
            thres_sel = thres[idxs]

            dists = pairwise_distances(thres_sel.reshape(-1, 1))
            upper_triangular_indices = np.triu_indices(dists.shape[0], k=1)

            # Check if we have at least two points to calculate a distance
            if dists[upper_triangular_indices].size > 0:
                eps = np.min(dists[upper_triangular_indices]) * 1 / 3
            else:
                eps = 0.05  # Default epsilon if only one threshold exists

            nuis_values = np.concatenate((thres_sel - eps, thres_sel + eps), axis=None)
            nuis_list.append(nuis_values)

        used_idx_array = np.array(used_idx_list)

        # checking if any feature didnt appear and fixing a value to it
        idx_diff = np.setdiff1d(nuissance_idx, feature_idx)
        unused_idx_list = []
        if idx_diff.shape[0] != 0:
            for i in range(idx_diff.shape[0]):
                nuis_list.append(np.array([0]))
                unused_idx_list.append(np.where(nuissance_idx == idx_diff[i])[0][0])
                nuis_list.append(np.array([0]))
                unused_idx_array = np.array(unused_idx_list)

            # reordering all combinations
            all_nuis_combinations = np.c_[list(itertools.product(*nuis_list))]
            nuisance_reorder = np.argsort(
                np.concatenate((used_idx_array, unused_idx_array), axis=None)
            )
            all_nuis_combinations = all_nuis_combinations[:, nuisance_reorder]
        else:
            all_nuis_combinations = np.c_[list(itertools.product(*nuis_list))]

        print(all_nuis_combinations[np.arange(0, 10), :])

        print(f"Total number of combinations: {all_nuis_combinations.shape[0]}")
        i = 0
        max_list = []
        for par in tqdm(
            par_values, desc="Computing nuisance cutoffs for each parameter value"
        ):
            par_array = np.tile(par, reps=(all_nuis_combinations.shape[0], 1))
            new_par = np.column_stack((par_array, all_nuis_combinations))

            # reordering columns
            new_par = new_par[:, par_reorder]

            # computing cutoffs for new par
            cutoff_vector = trust_plus_obj.compute_cutoffs(
                new_par, 
                K=K,
                by_batch = by_batch,
                use_jax = use_jax,)

            # index of max
            max_idx = np.argmax(cutoff_vector)
            max_list.append(new_par[max_idx, :])

            if compute_CI:
                par_CI = new_par[max_idx, :].reshape(1, -1)
                _, lower_bound, upper_bound = (
                    trust_plus_obj.compute_cutoffs(
                    par_CI, 
                    K = K, 
                    compute_CI = compute_CI,
                    alpha_CI = alpha_CI,
                    by_batch = by_batch,
                    use_jax = use_jax,
                    )
                )
                lower_bound_nuis[i] = lower_bound[0]
                upper_bound_nuis[i] = upper_bound[0]

            if i == 0:
                print(cutoff_vector)
                print(new_par[np.arange(0, 5)])

            # returning minimal value
            cutoff_nuis[i] = np.max(cutoff_vector)
            i += 1

    if compute_CI:
        return cutoff_nuis, max_list, lower_bound_nuis, upper_bound_nuis
    else:
        return cutoff_nuis, max_list