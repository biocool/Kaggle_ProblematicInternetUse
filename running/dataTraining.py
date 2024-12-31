from MLKaggle.Scripts.dataImputation import handle_feature_space_missing_values
from MLKaggle.Scripts.helper import separate_feature_space_label, remove_non_shared_features
from MLKaggle.Scripts.dataManipulation import check_range_values
import pandas as pd
from MLKaggle.Scripts.pipeline.modelTraining import apply_pipeline
from MLKaggle.Scripts.imputation.defineOnehotEncoding import apply_encoding
from MLKaggle.Scripts.helper import data_splitting

if __name__ == '__main__':

    # Main script for data preprocessing, imputation, and model training.

    train_data_path = '../Data/child-mind-institute-problematic-internet-use/train.csv'
    test_data_path = '../Data/child-mind-institute-problematic-internet-use/test.csv'
    desc_file_path = '../Data/child-mind-institute-problematic-internet-use/data_dictionary.csv'

    output_path = '../Data/data_cleaning'
    clf_list = ['RandomForestClassifier', 'DecisionTreeClassifier', 'ExtraTreeClassifier',
                'GradientBoostingClassifier', 'XGBoostTree', 'XGBoostLinear', 'CatBoostClassifier']

    clf_list = ['RandomForestClassifier']

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    desc_df = pd.read_csv(desc_file_path)

    # remove non-shared features
    train_df = remove_non_shared_features(train_df, test_df)

    # filter instances with na label
    train_df = train_df.loc[~ train_df['sii'].isna()]

    # Drop rows with more than 80% NaN values
    threshold = int(0.2 * train_df.shape[1])
    train_df = train_df.dropna(thresh=threshold)

    x_total_train_dat, y_total_train_dat = separate_feature_space_label(train_df, label_col='sii')

    check_range_values(x_total_train_dat, y_df=y_total_train_dat,
                       feature_explanation_df=desc_df, label_col='sii',
                       output_path=output_path, mode='train')

    # data splitting
    X_train, X_test, Y_train, Y_test = data_splitting(x_total_train_dat, y_total_train_dat, test_size=0.2,
                                                      out_dir=output_path, mode='train.test')

    kds, encoded_imputed_x_dat, encoder, safe_cols = \
        handle_feature_space_missing_values(X_train, Y_train, desc_df, random_state=42, output_path=output_path)

    X_train = X_train[safe_cols]
    X_test = X_test[safe_cols]
    test_df = test_df[safe_cols]

    # Identify categorical columns
    categorical_columns = [col for col in X_train.columns if X_train[col].dtype == 'object' or
                           X_train[col].dtype.name == 'category']

    # Select non-categorical columns
    num_cols = [col for col in X_train.columns if col not in categorical_columns]

    # apply encoding
    encoded_X_test_na_handling = apply_encoding(X_test, encoder)

    encoded_X_test_na_handling = encoded_X_test_na_handling.dropna(axis=1, how='all')

    encoded_X_test_na_handling_reset_idx = encoded_X_test_na_handling.reset_index(drop=True)

    encoded_X_test_na_handling_reset_idx = \
        kds.impute_new_data(new_data=encoded_X_test_na_handling_reset_idx).complete_data()

    apply_pipeline(encoded_imputed_x_dat, encoded_X_test_na_handling, Y_train, Y_test, clf_list,
                   output_path, num_cols=num_cols, n_classes=4,
                   n_jobs=-1, random_state=42, mode='Train')
