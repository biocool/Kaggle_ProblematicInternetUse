import numpy as np


def extract_grid_search_results(grid_search):

    """
    Extracts key metrics from a GridSearchCV object, including the mean and standard deviation
    of training and test scores for the best parameters, and the average fit time.

    Parameters:
    - grid_search (GridSearchCV): The trained GridSearchCV object.

    Returns:
    - tuple: Contains:
        - float: Mean training score of the best parameters (rounded to 2 decimals).
        - float: Mean test score of the best parameters (rounded to the nearest integer).
        - float: Standard deviation of training scores for the best parameters (rounded to 2 decimals).
        - float: Standard deviation of test scores for the best parameters (rounded to 2 decimals).
        - float: Average fit time in minutes.
    """

    # Get the mean training and test scores for the best parameters
    best_index = grid_search.best_index_
    best_param_mean_train_score = np.round(grid_search.cv_results_['mean_train_score'][best_index], 2)
    best_param_mean_test_score = np.round(grid_search.cv_results_['mean_test_score'][best_index])

    # Get the std values for the training and test scores of the best parameters
    best_param_std_train_score = np.round(grid_search.cv_results_['std_train_score'][best_index], 2)
    best_param_std_test_score = np.round(grid_search.cv_results_['std_test_score'][best_index], 2)

    # Convert from seconds to minutes
    mean_fit_times_minutes = np.mean(grid_search.cv_results_['mean_fit_time']) / 60

    return best_param_mean_train_score, best_param_mean_test_score, best_param_std_train_score, \
               best_param_std_test_score, mean_fit_times_minutes
