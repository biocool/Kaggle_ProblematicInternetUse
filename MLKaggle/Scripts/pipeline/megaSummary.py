import pandas as pd


def mega_summary_results(fitted_model_dict):

    """

    Generates a summary of model performance metrics for each classifier and saves it as a CSV file.

    Parameters:
    - fitted_model_dict (dict): A dictionary containing performance metrics and other details
      for each classifier, keyed by classifier name.

    """

    for classifier_name in fitted_model_dict.keys():

        # Replace 'Models_' with 'MegaSummary_' in the model path
        full_file_path_for_writing = classifier_name + '.MegaSummary.csv'

        # Initialize rows for train and validation data
        train_row = [classifier_name]
        # Append performance metrics
        train_row.extend(fitted_model_dict[classifier_name]['performance train'])
        train_row.extend(fitted_model_dict[classifier_name]['performance test'])
        train_row.append(fitted_model_dict[classifier_name]['Train mean across folds'])
        train_row.append(fitted_model_dict[classifier_name]['Test mean across folds'])
        train_row.append(fitted_model_dict[classifier_name]['Train std across folds'])
        train_row.append(fitted_model_dict[classifier_name]['Test std across folds'])
        train_row.append(fitted_model_dict[classifier_name]['mean time to fit the model (min)'])
        train_row.append(fitted_model_dict[classifier_name]['best params'])

        rows = [train_row]

        # Create a dataframe from the rows and write it to a .csv file
        perf_measures_df = pd.DataFrame(rows, columns=["Classifier", "performance train", "performance test",
                                                       "Train mean across folds", "Test mean across folds",
                                                       "Train std across folds", "Test std across folds",
                                                       "mean time to fit the model (min)", 'best params'])

        perf_measures_df.to_csv(full_file_path_for_writing, index=False)
