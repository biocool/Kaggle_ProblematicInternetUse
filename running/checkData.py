from ML.Scripts.dataManipulation import check_range_values, diff_train_test_feature_space
from ML.Scripts.helper import separate_feature_space_label, remove_non_shared_features
import pandas as pd

if __name__ == '__main__':

    train_data_path = '../Data/child-mind-institute-problematic-internet-use/train.csv'
    test_data_path = '../Data/child-mind-institute-problematic-internet-use/test.csv'
    feature_explanation = '../Data/child-mind-institute-problematic-internet-use/data_dictionary.csv'

    output_path = '../Data/summerise_data'

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    feature_explanation_df = pd.read_csv(feature_explanation)

    x_train, y_train = separate_feature_space_label(train_df, label_col='sii')
    x_test = separate_feature_space_label(test_df, label_col=None)

    # compare train & test
    diff_df = diff_train_test_feature_space(x_train, x_test, output_path=output_path)

    # remove non-shared features
    x_train = remove_non_shared_features(x_train, x_test)

    # check values
    check_range_values(x_train, y_df=y_train, feature_explanation_df=feature_explanation_df, label_col='sii',
                       output_path=output_path, mode='train')
    check_range_values(x_test, y_df=None, feature_explanation_df=feature_explanation_df, label_col=None,
                       output_path=output_path, mode='test')


