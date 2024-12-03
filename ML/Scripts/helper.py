import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from ML.Scripts.dataManipulation import diff_train_test_feature_space
from sklearn.model_selection import train_test_split
import os


def remove_non_shared_features(train_dat, test_dat):

    diff_df = diff_train_test_feature_space(train_dat, test_dat, output_path=None)

    cols_should_remove = diff_df['Columns in Train data not in Test'].values

    sel_cols = [v for v in train_dat.columns if v not in cols_should_remove]

    train_dat = train_dat[sel_cols]

    return train_dat


def data_splitting(X, y, out_dir, test_size):

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
    save_path = os.path.join(save_dir, "split_data.npy")
    np.save(save_path, data_dict)

    return X_train, X_test, y_train, y_test


def find_highly_correlated_cols(df, threshold=0.85, output_path=None, mode=None):
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

    high_correlation_pairs = high_correlation_pairs.sort_values(by='Correlation', key=abs, ascending=False)

    if output_path is not None:
        high_correlation_pairs.to_csv(output_path + '/' + mode + '.correlation.csv', index=False)


def separate_feature_space_label(df, label_col=None):

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

        y_df = df[[label_col]]

        return feature_space_df, y_df

    else:

        return feature_space_df

