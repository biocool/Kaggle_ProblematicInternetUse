import numpy as np
from MLKaggle.Scripts.imputation.defineOnehotEncoding import apply_encoding


def normalized_rmsd_scoro_func(x_dat_imputed, x_dat_spread_na, x_dat_before_spread_na, on_hot_encoding_rule,
                               encoder=None):

    """
    Calculates the normalized Root Mean Square Deviation (RMSD) for imputed data,
    comparing it to the original dataset(before introducing missingness).

    Parameters:
    - x_dat_imputed (DataFrame): The dataset after imputation.
    - x_dat_spread_na (DataFrame): The dataset with missing values introduced.
    - x_dat_before_spread_na (DataFrame): The original dataset before introducing missing values.
    - on_hot_encoding_rule (bool): Whether one-hot encoding was applied to categorical columns.
    - encoder (OneHotEncoder, optional): A pre-fitted encoder for transforming categorical columns. Required if
      `on_hot_encoding_rule` is True.

    Returns:
    - float: The mean normalized RMSD across all columns.

    Workflow:
    1. For each column in the original dataset:
        - Identify rows where missingness was introduced.
        - Compare actual (original) values and predicted (imputed) values for these rows.
    2. For categorical columns:
        - Compute RMSD in one-hot encoded space and normalize by the maximum possible distance.
    3. For numerical columns:
        - Compute RMSD and normalize by the range of actual values.
    4. Return the mean normalized RMSD across all processed columns.
    """

    if on_hot_encoding_rule:
        encoded_x_dat_before_spread_na = apply_encoding(x_dat_before_spread_na, encoder)

        encoded_x_dat_before_spread_na = \
            encoded_x_dat_before_spread_na.dropna(axis=1, how='all')

    normalized_rmsd_list = []

    for raw_col in x_dat_before_spread_na.columns:

        cond1 = ~ x_dat_before_spread_na[raw_col].isna()
        cond2 = x_dat_spread_na[raw_col].isna()
        sel_instances = x_dat_before_spread_na.loc[cond1 & cond2]

        if sel_instances.shape[0] > 0:

            sel_x_dat_imputed = x_dat_imputed.loc[sel_instances.index.values]

            if raw_col in encoder.feature_names_in_:

                encoded_indices = np.where(encoder.feature_names_in_ == raw_col)[0]
                rel_col_in_imputed = encoder.get_feature_names_out()[encoded_indices]
                rel_col_in_imputed = [col for col in rel_col_in_imputed if not col.endswith('_nan')]

                # categorical col
                # Ensure inputs are numpy arrays
                actual = \
                    encoded_x_dat_before_spread_na.loc[sel_instances.index.values, rel_col_in_imputed].values
                predicted = sel_x_dat_imputed[rel_col_in_imputed].values

                # Compute the squared differences
                squared_diff = np.sum((actual - predicted) ** 2, axis=1)

                # Calculate RMSD
                rmsd = np.sqrt(np.mean(squared_diff))

                # Calculate the normalization factor (max possible distance in one-hot space)
                max_distance = np.sqrt(2)

                # Normalized RMSD
                normalized_rmsd = rmsd / max_distance

                if str(normalized_rmsd) == 'nan':

                    print(rmsd)
                    print(squared_diff)
                    print(actual)
                    print(rel_col_in_imputed)
                    print(predicted)
                    breakpoint()

                normalized_rmsd_list.append(normalized_rmsd)

            else:

                rel_col_in_imputed = raw_col

                # numerical col
                all_instances = x_dat_before_spread_na[rel_col_in_imputed].values
                actual = x_dat_before_spread_na.loc[sel_instances.index.values, rel_col_in_imputed].values
                predicted = sel_x_dat_imputed[rel_col_in_imputed].values

                # Compute the squared differences
                squared_diff = (actual - predicted) ** 2

                # Calculate RMSD
                rmsd = np.sqrt(np.mean(squared_diff))

                # Calculate normalization factor (range of actual values)
                value_range = np.nanmax(all_instances) - np.nanmin(all_instances)

                # Normalized RMSD
                normalized_rmsd = rmsd / value_range

                if str(normalized_rmsd) == 'nan':
                    print(value_range)
                    print(all_instances.tolist())
                    breakpoint()
                normalized_rmsd_list.append(normalized_rmsd)

    return np.mean(normalized_rmsd_list)
