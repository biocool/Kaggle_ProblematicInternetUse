import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def correct_na_categorical_cols(encoded_df, encoder, encoded_feature_names):

    """
    Corrects missing values in categorical columns after one-hot encoding by propagating NaN
    values to the related one-hot encoded features.

    Parameters:
    - encoded_df (DataFrame): The DataFrame with one-hot encoded categorical columns.
    - encoder (OneHotEncoder): The encoder used to transform the data.
    - encoded_feature_names (array-like): The names of the one-hot encoded features.

    Returns:
    - DataFrame: The updated DataFrame with NaN values propagated to one-hot encoded features.
    """

    cols_with_na = [col for col in encoded_feature_names if col.endswith('_nan')]

    for col_with_na in cols_with_na:
        col_name = col_with_na[:-4]
        encoded_indices = np.where(encoder.feature_names_in_ == col_name)[0]
        related_col = encoder.get_feature_names_out()[encoded_indices]

        sel_rows = encoded_df.loc[encoded_df[col_with_na] == 1]

        if sel_rows.shape[0] > 0:
            encoded_df.loc[sel_rows.index, related_col] = np.nan

    return encoded_df


def apply_one_hot_encoding(feature_space_df, categorical_columns, non_categorical_columns):

    """
    Applies one-hot encoding to categorical features and combines the encoded features with
    non-categorical features, handling missing values appropriately.

    Parameters:
    - feature_space_df (DataFrame): The original dataset with both categorical and non-categorical features.
    - categorical_columns (list): List of column names corresponding to categorical features.
    - non_categorical_columns (list): List of column names corresponding to non-categorical features.

    Returns:
    - DataFrame: The dataset with one-hot encoded categorical features and original non-categorical features.
    - OneHotEncoder: The encoder used for one-hot encoding.
    """

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit the encoder on training data
    encoder.fit(feature_space_df[categorical_columns])

    # Transform both train and test sets
    feature_space_encoded = encoder.transform(feature_space_df[categorical_columns])

    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

    feature_space_encoded_df = pd.DataFrame(
        feature_space_encoded,
        columns=encoded_feature_names,  # Use the encoded feature names from the OneHotEncoder
        index=feature_space_df.index  # Use the same index as the original DataFrame
    )

    feature_space_encoded_df_na_handling = (
        correct_na_categorical_cols(feature_space_encoded_df.copy(), encoder, encoded_feature_names))

    # Combine encoded categorical columns with non-categorical columns
    feature_space_final_na_handling = pd.concat([feature_space_df[non_categorical_columns],
                                                 feature_space_encoded_df_na_handling], axis=1)

    return feature_space_final_na_handling, encoder


def apply_encoding(dat, encoder):

    """
    Applies a pre-fitted one-hot encoder to a new dataset, handling missing values and ensuring consistency
    with the original dataset used for encoding.

    Parameters:
    - dat (DataFrame): The new dataset to be transformed.
    - encoder (OneHotEncoder): A pre-fitted encoder to transform the categorical features.

    Returns:
    - DataFrame: The transformed dataset with one-hot encoded categorical features and
      original non-categorical features.
    """

    categorical_columns = [col for col in dat.columns if dat[col].dtype == 'object'
                           or dat[col].dtype.name == 'category']

    num_cols = [col for col in dat.columns if col not in categorical_columns]

    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

    # Transform both train and test sets
    dat_encoded = encoder.transform(dat[categorical_columns])

    dat_encoded_df = pd.DataFrame(
        dat_encoded,
        columns=encoded_feature_names,  # Use the encoded feature names from the OneHotEncoder
        index=dat.index  # Use the same index as the original DataFrame
    )

    dat_encoded_df_na_handling = correct_na_categorical_cols(dat_encoded_df.copy(), encoder, encoded_feature_names)

    cols_not_in_feature_space = [col for col in encoder.feature_names_in_ if col not in categorical_columns]

    if len(cols_not_in_feature_space) > 0:
        extra_encoded_feature_names = [
            name for name, original_col in zip(encoder.get_feature_names_out(), encoder.feature_names_in_)
            if original_col in cols_not_in_feature_space]
        cols_without_na = [col for col in extra_encoded_feature_names if not col.endswith('_nan')]

        dat_encoded_df_na_handling[cols_without_na] = 0

    # Combine encoded categorical columns with non-categorical columns
    dat_final_na_handling = pd.concat([dat[num_cols], dat_encoded_df_na_handling], axis=1)

    return dat_final_na_handling
