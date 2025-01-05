import pandas as pd
import numpy as np
from MLKaggle.Scripts.dataManipulation import diff_train_test_feature_space
from sklearn.model_selection import train_test_split
import os


def remove_non_shared_features(train_dat, test_dat):

    """
    Removes features present in the training dataset but absent in the test dataset.

    Parameters:
    - train_dat (DataFrame): The training dataset.
    - test_dat (DataFrame): The test dataset.

    Returns:
    - DataFrame: The training dataset with non-shared features removed.
    """

    diff_df = diff_train_test_feature_space(train_dat, test_dat, output_path=None)

    cols_should_remove = diff_df['Columns in Train data not in Test'].values

    sel_cols = [v for v in train_dat.columns if v not in cols_should_remove]

    train_dat = train_dat[sel_cols]

    return train_dat


def data_splitting(X, y, test_size, mode, out_dir):

    """
    Splits the data into training and test sets, performs stratification based on the target variable,
    and saves the split data to a file.

    Parameters:
    - X (DataFrame): Features dataset.
    - y (Series): Target variable.
    - test_size (float): Proportion of the dataset to include in the test split.
    - mode (str): Name of the mode (e.g., 'train', 'test') for saving the data.
    - out_dir (str): Directory path to save the split data.

    Returns:
    - tuple: (X_train, X_test, y_train, y_test) split datasets.
    """

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,  # Stratify based on the target variable
        random_state=42  # Set random state for reproducibility
    )

    # Prepare data dictionary
    data_dict = {
        "X_train": X_train.values,
        "X_test": X_test.values,
        "y_train": y_train.values,
        "y_test": y_test.values
    }

    # Directory to save the file
    save_dir = out_dir + "/split_data/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the dictionary as a single .npy file
    save_path = os.path.join(save_dir + mode + ".split_data.npy")
    np.save(save_path, data_dict)

    return X_train, X_test, y_train, y_test


def find_highly_correlated_cols(df, threshold=0.85, output_path=None, mode=None):

    """
    Identifies and removes highly correlated columns based on a given threshold.

    Parameters:
    - df (DataFrame): The input dataset.
    - threshold (float): Correlation threshold to consider for removing features. Default is 0.85.
    - output_path (str, optional): Path to save the correlation results as a CSV file.
    - mode (str, optional): Label (e.g., 'train', 'test') to include in the saved file's name.

    Returns:
    - DataFrame: Dataset with highly correlated features removed.
    """

    # Identify categorical columns
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']

    # Convert categorical columns to one-hot encoding
    df_one_hot = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Calculate correlation matrix
    correlation_matrix = df_one_hot.corr()

    # Use the upper triangle of the correlation matrix to find highly correlated pairs
    corr_matrix_upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    # Find pairs with high absolute correlation
    high_correlation_pairs = corr_matrix_upper.stack().reset_index()
    high_correlation_pairs.columns = ['Column 1', 'Column 2', 'Correlation']
    high_correlation_pairs = high_correlation_pairs[high_correlation_pairs['Correlation'].abs() > threshold]

    # Calculate average correlation for each column
    avg_correlation = correlation_matrix.abs().mean(axis=0)

    # Track columns to keep
    columns_to_drop = set()

    for _, row in high_correlation_pairs.iterrows():
        col1 = row['Column 1']
        col2 = row['Column 2']

        # Compare average correlations and mark the column with higher avg correlation for removal
        if col1 not in columns_to_drop and col2 not in columns_to_drop:
            if avg_correlation[col1] > avg_correlation[col2]:
                columns_to_drop.add(col1)
            else:
                columns_to_drop.add(col2)

    # Optionally save results
    if output_path is not None and mode is not None:
        high_correlation_pairs.to_csv(output_path + '/' + mode + '.correlation.csv', index=False)

    if len(set(columns_to_drop)):
        print(set(columns_to_drop))
        df_reduced = df.drop(columns=list(set(columns_to_drop)))

        return df_reduced
    else:
        return df


def separate_feature_space_label(df, label_col=None):

    """
    Separates the feature space and the label column from the dataset.

    Parameters:
    - df (DataFrame): The input dataset with features and labels.
    - label_col (str, optional): Column name for the label variable.

    Returns:
    - tuple:
        - DataFrame: Feature space with categorical columns appropriately converted.
        - DataFrame: Label column as a separate DataFrame (if `label_col` is specified).
    """

    df.index = df['id'].values
    df = df.drop('id', axis=1)

    feature_space_cols = [col for col in df if col not in ['id', label_col]]

    feature_space_df = df[feature_space_cols].copy()

    categorical_features_encoded_numerical = ['Basic_Demos-Sex', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone',
                                              'FGC-FGC_GSD_Zone', 'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone',
                                              'FGC-FGC_SRR_Zone', 'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num',
                                              'BIA-BIA_Frame_num', 'PreInt_EduHx-computerinternet_hoursday']

    intersect_cols = set(feature_space_cols).intersection(set(categorical_features_encoded_numerical))

    # Convert intersecting columns to categorical
    feature_space_df[list(intersect_cols)] = feature_space_df[list(intersect_cols)].astype('category')

    if label_col is not None:

        y_df = df[[label_col]].astype(int)

        return feature_space_df, y_df

    else:

        return feature_space_df


def find_features_high_frequency_na(df, na_size=0.8):

    """
    Identifies columns with a high frequency of missing values.

    Parameters:
    - df (DataFrame): The input dataset.
    - na_size (float): Threshold for the fraction of missing values to consider a column as high NA. Default is 0.8.

    Returns:
    - tuple:
        - list: List of columns with high NA frequency.
        - list: List of columns safe to retain (not highly NA).
    """

    # Calculate the fraction of missing values per column
    na_fraction = df.isna().mean()

    # Find columns with NaN frequency > 0.8
    high_na_columns = na_fraction[na_fraction > na_size].index.tolist()

    safe_cols = [v for v in df.columns if v not in high_na_columns]

    return high_na_columns, safe_cols
