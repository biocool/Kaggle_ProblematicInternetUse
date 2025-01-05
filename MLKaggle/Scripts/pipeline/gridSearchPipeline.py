from MLKaggle.Scripts.pipeline.classifiers import define_classifiers
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import make_scorer
import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, n_classes):
    O = np.zeros((n_classes, n_classes), dtype=np.float64)

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)

    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    hist_true = np.histogram(y_true, bins=np.arange(n_classes+1))[0]
    hist_pred = np.histogram(y_pred, bins=np.arange(n_classes+1))[0]

    E = np.outer(hist_true, hist_pred) / np.sum(hist_true)

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)
    kappa = 1 - (numerator / denominator)

    return kappa


def make_qwk_scorer(n_classes=4):
    def qwk_fixed_n_classes(y_true, y_pred):
        return quadratic_weighted_kappa(y_true, y_pred, n_classes=n_classes)

    return make_scorer(qwk_fixed_n_classes, greater_is_better=True)


def grid_search_pipeline(clf_list, num_cols, n_jobs, random_state, n_classes=4):
    """
    Constructs and returns a list of GridSearchCV objects for different classifiers, with preprocessing,
    custom scoring, and cross-validation setup.

    Parameters:
    - clf_list (list): List of classifier names to include in the pipeline.
    - num_cols (list): List of numerical column names for preprocessing.
    - n_jobs (int): Number of jobs to run in parallel for GridSearchCV.
    - random_state (int): Random seed for reproducibility.
    - n_classes (int, optional): Number of classes for the quadratic weighted kappa scorer. Default is 4.

    Returns:
    - list: A list of GridSearchCV objects, one for each classifier.

    Workflow:
    - Preprocessing:
        - Applies `RobustScaler` to numerical columns.
        - Keeps non-numerical columns as is using `make_column_transformer`.
    - Pipelines:
        - Combines preprocessing and classifier steps using `make_pipeline`.
    - Cross-Validation:
        - Uses `RepeatedStratifiedKFold` for robust evaluation with 5 splits and 5 repeats.
    - Scoring:
        - Utilizes a custom quadratic weighted kappa (QWK) scorer tailored to the specified number of classes.
    - Grid Search:
        - Creates a `GridSearchCV` object for each classifier with its hyperparameter grid.
    """

    # It is the same for all the classifiers.

    classifiers_name_list, classifiers_list, param_grid_list = \
        define_classifiers(clf_list, n_jobs=n_jobs, random_state=random_state)

    # It is the same for all the classifiers.
    robust_scaler_transformer = (RobustScaler(), num_cols)

    preprocessor = make_column_transformer(robust_scaler_transformer,
                                           remainder='passthrough')

    # a list of pipeline objects (each contains the preprocessing and classification steps).
    clf_pipeline_list = [make_pipeline(preprocessor, clf) for clf in classifiers_list]

    # n_splits = number of folds
    # n_repeats: Number of times cross-validator needs to be repeated
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)

    qwk_scorer = make_qwk_scorer(n_classes)

    grid_search_list = [
        GridSearchCV(pipeline, param_grid=param_grid, scoring=qwk_scorer,
                     refit=True, verbose=2, cv=cv, return_train_score=True)
        for pipeline, param_grid in zip(clf_pipeline_list, param_grid_list)
    ]

    return grid_search_list
