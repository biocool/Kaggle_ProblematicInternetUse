from MLKaggle.Scripts.helper import data_splitting, find_features_high_frequency_na
from MLKaggle.Scripts.imputation.chainedEquationImputation import (Chained_equation_with_lightgbm_greed_search,
                                                                   Chained_equation_with_lightgbm)


def handle_feature_space_missing_values(x_dat, y_dat, desc_df, random_state=42, output_path=None):

    """
    Handles missing values in the feature space using chained equations with LightGBM

    Parameters:
    - x_dat (DataFrame): Feature dataset with potential missing values.
    - y_dat (DataFrame or Series): Target variable used for splitting and imputation.
    - desc_df (DataFrame): Feature description metadata, used for additional context.
    - random_state (int, optional): Seed for reproducibility. Default is 42.
    - output_path (str, optional): Path to save intermediate results (e.g., splits).

    Returns:
    - kds (ImputationKernel): The imputation kernel containing the fitted imputation model.
    - encoded_imputed_x_dat (DataFrame): The imputed and encoded dataset.
    - encoder (OneHotEncoder): Encoder used for categorical feature transformations.
    - safe_cols (list): List of columns retained after filtering out those with high missingness.
    """

    x_train_for_imp, x_val_for_imp, y_train_for_imp, y_val_for_imp = \
        data_splitting(x_dat, y_dat, test_size=0.2, out_dir=output_path, mode='imputation')

    high_na_columns, safe_cols = find_features_high_frequency_na(x_train_for_imp, na_size=0.8)

    # remove high_na_columns
    x_train_for_imp = x_train_for_imp[safe_cols]
    x_val_for_imp = x_val_for_imp[safe_cols]

    x_dat = x_dat[safe_cols]

    x_train_for_imp_reset_index = x_train_for_imp.reset_index(drop=True)
    y_train_for_imp_reset_index = y_train_for_imp.reset_index(drop=True)

    best_param, best_normalized_rmsd, encoder = \
        Chained_equation_with_lightgbm_greed_search(x_train_for_imp_reset_index, y_train_for_imp_reset_index,
                                                    na_perc=0.25, n_splits=5, n_repeats=1,
                                                    random_state=random_state, on_hot_encoding_rule=True)

    kds = Chained_equation_with_lightgbm(x_dat, best_param, random_state=42, on_hot_encoding_rule=True, encoder=encoder)

    encoded_imputed_x_dat = kds.complete_data()

    return kds, encoded_imputed_x_dat, encoder, safe_cols
