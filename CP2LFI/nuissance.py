import numpy as np
import itertools
from tqdm import tqdm


def TRUST_nuisance_cutoffs(
    trust_obj, nuissance_idx, par_values, trust_quantiles, eps=0.05
):
    """
    Compute the nuissance adapted cutoff values for TRUST.

    Args:
        trust_obj (object): The trust object.
        nuissance_idx (int): The index of the nuissance feature.
        par_values (numpy.ndarray): The parameter values.
        trust_quantiles (numpy.ndarray): The trust quantiles.
        eps (float, optional): The epsilon value. Defaults to 0.05.

    Returns:
        numpy.ndarray: The cutoff values for the nuissance feature.
    """

    # returning all thresholds
    feature_array = np.array(trust_obj.cart.tree_.feature)
    threshold_array = np.array(trust_obj.cart.tree_.threshold)

    # cutoff array
    cutoff_nuis = np.zeros(par_values.shape[0])

    # selecting threshold_array with feature array equal to nuissance idx
    if isinstance(nuissance_idx, int):
        idxs = np.where(feature_array == nuissance_idx)
        thres = threshold_array[idxs]
        nuis_values = np.concatenate((thres - eps, thres + eps), axis=None)

        # returning all index
        size = par_values.shape[1] + 1
        # idx list
        idx_array = np.arange(0, size)
        par_idx = idx_array[np.where(idx_array != nuissance_idx)]
        par_reorder = np.concatenate((par_idx, nuissance_idx), axis=None)

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
    else:
        idxs = np.where(feature_array == nuissance_idx)
        thres = threshold_array[idxs]
        # TODO obtaining all combinations using itertools
        nuis_values = np.concatenate((thres - eps, thres + eps), axis=None)

    return cutoff_nuis


def TRUST_plus_nuisance_cutoff(
    trust_plus_obj,
    nuissance_idx,
    par_values,
    K=100,
    eps=0.05,
):
    """
    Compute the nuisance adapted cutoffs for TRUST++ algorithm.

    Args:
        trust_plus_obj (object): An object containing the TRUST+ algorithm.
        nuissance_idx (int): The index of the nuissance feature.
        par_values (ndarray): An array of parameter values.
        K (int, optional): The number of trees to use in the random forest. Defaults to 100.
        eps (float, optional): The epsilon value used to compute the threshold values. Defaults to 0.05.

    Returns:
        ndarray: An array of cutoff values for each parameter value.
    """

    threshold_list = []
    feature_list = []
    # returning all thresholds for each tree
    for tree in trust_plus_obj.RF.estimators_:
        feature_array = np.array(tree.tree_.feature)
        threshold_array = np.array(tree.tree_.threshold)

        idxs = np.where(feature_array == nuissance_idx)
        thres = threshold_array[idxs]
        feature_nuis = feature_array[idxs]

        threshold_list.extend(list(thres))
        feature_list.extend(list(feature_nuis))

    # cutoff array
    cutoff_nuis = np.zeros(par_values.shape[0])

    # selecting threshold_array with feature array equal to nuissance idx
    if isinstance(nuissance_idx, int):
        thres = np.array(threshold_list)
        nuis_values = np.concatenate((thres - eps, thres + eps), axis=None)

        # returning all index
        size = par_values.shape[1] + 1
        # idx list
        idx_array = np.arange(0, size)
        par_idx = idx_array[np.where(idx_array != nuissance_idx)]
        par_reorder = np.concatenate((par_idx, nuissance_idx), axis=None)

        i = 0
        for par in tqdm(
            par_values, desc="Computing nuisance cutoffs for each parameter value"
        ):
            par_array = np.tile(par, reps=(nuis_values.shape[0], 1))
            new_par = np.column_stack((par_array, nuis_values))

            # reordering columns
            new_par = new_par[:, par_reorder]

            # computing cutoffs for new par
            cutoff_vector = trust_plus_obj.compute_cutoffs(new_par, K=K)

            # returning minimal value
            cutoff_nuis[i] = np.max(cutoff_vector)
            i += 1
    else:
        idxs = np.where(feature_array == nuissance_idx)
        thres = threshold_array[idxs]
        # TODO obtaining all combinations using itertools
        nuis_values = np.concatenate((thres - eps, thres + eps), axis=None)

    return cutoff_nuis
