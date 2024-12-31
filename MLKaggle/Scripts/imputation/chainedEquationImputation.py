from sklearn.model_selection import RepeatedStratifiedKFold
import miceforest as mf
import numpy as np
import itertools
from MLKaggle.Scripts.imputation.defineOnehotEncoding import apply_one_hot_encoding, apply_encoding
from MLKaggle.Scripts.imputation.scoreFunctions import normalized_rmsd_scoro_func


def data_prep(x_dat, na_perc, random_state):

    """
    Prepares the dataset by introducing missingness and applying one-hot encoding for categorical features.

    Parameters:
    - x_dat (DataFrame): The original dataset.
    - na_perc (float): Percentage of missingness to introduce. If None, no missingness is added.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - DataFrame: The dataset with introduced missingness.
    - DataFrame: The one-hot encoded version of the dataset with missingness.
    - Encoder object: The encoder used for one-hot encoding.
    """

    # Identify categorical columns
    categorical_columns = [col for col in x_dat.columns if x_dat[col].dtype == 'object' or
                           x_dat[col].dtype.name == 'category']

    # Select non-categorical columns
    num_cols = [col for col in x_dat.columns if col not in categorical_columns]

    # Introduce missingness
    if na_perc is not None:
        x_dat_spread_na = mf.ampute_data(x_dat, perc=na_perc, random_state=random_state)
    else:
        x_dat_spread_na = x_dat

    # One-Hot Encoding for categorical features
    encoded_x_dat_spread_na, encoder = apply_one_hot_encoding(x_dat_spread_na, categorical_columns, num_cols)

    encoded_x_dat_spread_na = encoded_x_dat_spread_na.dropna(axis=1, how='all')

    return x_dat_spread_na, encoded_x_dat_spread_na, encoder


def Chained_equation_with_lightgbm_greed_search(x_dat, y, na_perc=0.25, n_splits=5, n_repeats=10, random_state=42,
                                                on_hot_encoding_rule=True):

    """
    Performs hyperparameter tuning for chained equation imputation using LightGBM with cross-validation.

    Parameters:
    - x_dat (DataFrame): The dataset to impute.
    - y (Series): Target labels for stratified cross-validation.
    - na_perc (float, optional): Percentage of missingness to introduce. Default is 0.25.
    - n_splits (int, optional): Number of folds for cross-validation. Default is 5.
    - n_repeats (int, optional): Number of times cross-validation is repeated. Default is 10.
    - random_state (int, optional): Random seed for reproducibility. Default is 42.
    - on_hot_encoding_rule (bool, optional): Whether to apply one-hot encoding. Default is True.

    Returns:
    - dict: Best parameters for LightGBM.
    - float: Best normalized RMSD score.
    - Encoder object: The encoder used for one-hot encoding (if applied).
    """

    if on_hot_encoding_rule:
        x_dat_spread_na, encoded_x_dat_spread_na, encoder = data_prep(x_dat, na_perc, random_state)

    else:
        # Introduce missingness
        x_dat_spread_na = mf.ampute_data(x_dat, perc=na_perc, random_state=random_state)
        encoded_x_dat_spread_na = x_dat_spread_na.xopy()

    parameters_space = {
        'iterations': [5, 10, 50],
        'num_leaves': [15, 31, 63],
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [-1, 8]
    }

    # Get parameter names and their values
    param_names = parameters_space.keys()
    param_values = parameters_space.values()

    # Generate all combinations of parameter values
    param_combinations = list(itertools.product(*param_values))

    # Convert each combination into a dictionary
    param_dicts = [dict(zip(param_names, values)) for values in param_combinations]

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    score_params = []

    for params in param_dicts:

        score_sel_param = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(encoded_x_dat_spread_na, y)):

            x_train_dat = encoded_x_dat_spread_na.loc[train_idx]
            x_test_dat = encoded_x_dat_spread_na.loc[test_idx]

            x_test_dat_before_spread_na = x_dat.loc[test_idx]
            x_test_dat_after_spread_na = x_dat_spread_na.loc[test_idx]

            x_train_dat_reset_idx = x_train_dat.reset_index(drop=True)
            x_test_dat_reset_idx = x_test_dat.reset_index(drop=True)

            # Create kernel.
            kds = mf.ImputationKernel(x_train_dat_reset_idx, random_state=random_state)

            # Customizing LightGBM Parameters
            kds.mice(**params)

            x_test_dat_imputed = kds.impute_new_data(new_data=x_test_dat_reset_idx).complete_data()

            # add index
            x_test_dat_imputed.index = x_test_dat.index

            # now check score function
            avg_normalized_rmsd = \
                normalized_rmsd_scoro_func(x_test_dat_imputed, x_test_dat_after_spread_na,
                                           x_test_dat_before_spread_na, on_hot_encoding_rule, encoder=encoder)

            score_sel_param.append(avg_normalized_rmsd)

        score_params.append(np.mean(score_sel_param))

    # find best param
    best_normalized_rmsd = np.min(score_params)
    best_param = param_dicts[np.argmin(score_params)]

    return best_param, best_normalized_rmsd, encoder


def Chained_equation_with_lightgbm(x_dat, params, random_state=42, on_hot_encoding_rule=True, encoder=None):

    """
    Performs chained equation imputation on a dataset using LightGBM with specified parameters.

    Parameters:
    - x_dat (DataFrame): The dataset to impute.
    - params (dict): Parameters for LightGBM to customize the imputation process.
    - random_state (int, optional): Random seed for reproducibility. Default is 42.
    - on_hot_encoding_rule (bool, optional): Whether to apply one-hot encoding. Default is True.
    - encoder (Encoder object, optional): Encoder for one-hot encoding (if already applied).

    Returns:
    - ImputationKernel: The kernel containing imputed data and associated methods.
    """

    if on_hot_encoding_rule:

        encoded_x_dat_na_handling = apply_encoding(x_dat, encoder)

        encoded_x_dat_na_handling = encoded_x_dat_na_handling.dropna(axis=1, how='all')

    else:
        encoded_x_dat_na_handling = x_dat

    encoded_x_dat_na_handling_reset_idx = encoded_x_dat_na_handling.reset_index(drop=True)

    # Create kernel.
    kds = mf.ImputationKernel(encoded_x_dat_na_handling_reset_idx, random_state=random_state)

    # Customizing LightGBM Parameters
    kds.mice(**params)

    return kds
