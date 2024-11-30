import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder


def handle_feature_space_missing_values(df, n_neighbors=5, output_path=None, mode=None):
    """
    Handle missing values in the feature space using KNNImputer with ordinal encoding for categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_neighbors (int): Number of neighbors to use for KNN imputation.
        output_path (str): Directory to save the output file (optional).
        mode (str): Mode name for output file (optional).

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    # Identify categorical columns
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']

    # Define the order of seasons
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']

    # Initialize OrdinalEncoder with defined order
    encoder = OrdinalEncoder(categories=[season_order] * len(categorical_columns),
                             handle_unknown='use_encoded_value', unknown_value=np.nan)

    # Encode categorical columns
    df_encoded = df.copy()
    df_encoded[categorical_columns] = encoder.fit_transform(df[categorical_columns])

    # Impute missing values
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df_encoded.columns)

    # Decode categorical columns back to their original values
    df_imputed[categorical_columns] = encoder.inverse_transform(df_imputed[categorical_columns].round().astype(int))

    # Debugging breakpoint if shapes don't match
    if df_imputed.shape != df.shape:
        breakpoint()

    # Save the output to CSV if output_path is provided
    if output_path is not None:
        df_imputed.to_csv(f"{output_path}/{mode}.handle.missing.values.feature.space.csv", index=False)

    return df_imputed


def handle_label_missing_values(df, n_neighbors=5, output_path=None, mode=None, label_col='sii'):
    """
    Handle missing values in the feature space using KNNImputer with ordinal encoding for categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_neighbors (int): Number of neighbors to use for KNN imputation.
        output_path (str): Directory to save the output file (optional).
        mode (str): Mode name for output file (optional).

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """

    # Impute missing values
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

    # Debugging breakpoint if shapes don't match
    if df_imputed.shape != df.shape:
        breakpoint()

    # Save the output to CSV if output_path is provided
    if output_path is not None:
        df_imputed.to_csv(f"{output_path}/{mode}.handle.missing.values.label.csv", index=False)

    return df_imputed
