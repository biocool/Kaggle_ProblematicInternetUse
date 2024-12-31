from MLKaggle.Scripts.pipeline.gridSearchPipeline import grid_search_pipeline, quadratic_weighted_kappa
from MLKaggle.Scripts.pipeline.extractGridSearchResults import extract_grid_search_results
from MLKaggle.Scripts.pipeline.megaSummary import mega_summary_results
import traceback


def apply_pipeline(X_train, X_test, y_train, y_test, clf_list, output_path, num_cols, n_classes=4,
                   n_jobs=-1, random_state=42, mode='Train'):

    """
    Applies a pipeline with grid search for multiple classifiers, evaluates their performance, and
    saves results to a dictionary for further analysis.

    Parameters:
    - X_train (DataFrame): Training features.
    - X_test (DataFrame): Test features.
    - y_train (DataFrame or Series): Training target labels.
    - y_test (DataFrame or Series): Test target labels.
    - clf_list (list): List of classifier names to be included in the pipeline.
    - output_path (str): Path to save the results (currently unused).
    - num_cols (list): List of numerical columns for preprocessing.
    - n_classes (int, optional): Number of classes for the quadratic weighted kappa scorer. Default is 4.
    - n_jobs (int, optional): Number of jobs for parallel computation. Default is -1 (use all cores).
    - random_state (int, optional): Random seed for reproducibility. Default is 42.
    - mode (str, optional): Indicates the pipeline's execution mode (e.g., 'Train'). Default is 'Train'.

    Returns:
    - None: Results are saved in `clf_model_dict` and processed by `mega_summary_results`.

    Workflow:
    1. Builds a list of GridSearchCV objects for classifiers using `grid_search_pipeline`.
    2. Iterates through classifiers and fits the grid search on the training data.
    3. Evaluates performance using:
       - Mean and standard deviation of training and test scores.
       - Quadratic weighted kappa (QWK) for training and test predictions.
    4. Saves results in `clf_model_dict`, including:
       - Best parameters and predictions.
       - QWK scores for training and test sets.
       - Cross-validation statistics (mean and std scores).
       - Model fit time in minutes.
    """

    # building the gridsearch
    grid_search_list = grid_search_pipeline(clf_list, num_cols, n_jobs=n_jobs, random_state=random_state,
                                            n_classes=n_classes)
    clf_model_dict = {}

    for i, grid_search in enumerate(grid_search_list):

        try:

            pipeline = grid_search.estimator
            # last key in pipline is clf name
            clf_name, classifier = pipeline.steps[-1]

            # fitting the grid_search
            grid_search.fit(X_train, y_train.values.ravel())

            # Get the mean training and test scores for the best parameters
            best_param_mean_train_score, best_param_mean_test_score, best_param_std_train_score, \
                best_param_std_test_score, mean_fit_times_minutes = \
                extract_grid_search_results(grid_search=grid_search)

            y_pred_train = grid_search.predict(X_train)
            y_pred_test = grid_search.predict(X_test)

            qwk_test_score_train = quadratic_weighted_kappa(y_train.values.ravel(), y_pred_train, n_classes)
            qwk_test_score_test = quadratic_weighted_kappa(y_test.values.ravel(), y_pred_test, n_classes)

            clf_model_dict[clf_name] = {'fitted_model': grid_search,
                                        'mode': mode,
                                        'performance train': [qwk_test_score_train],
                                        'performance test': [qwk_test_score_test],
                                        'best params': [str(grid_search.best_params_)],
                                        'predict train': y_pred_train,
                                        'predict test': y_pred_test,
                                        'Train mean across folds': best_param_mean_train_score,
                                        'Test mean across folds': best_param_mean_test_score,
                                        'Train std across folds': best_param_std_train_score,
                                        'Test std across folds': best_param_std_test_score,
                                        'mean time to fit the model (min)': mean_fit_times_minutes}

            mega_summary_results(clf_model_dict)

        except Exception as e:
            tb = traceback.format_exc()
            print("Couldn't run gridseach: ")
            print(grid_search)
            print("Error is:")
            print(e)
            print(tb)
