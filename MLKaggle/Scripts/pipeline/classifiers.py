# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier


def classifier_parameters(n_jobs, random_state):

    """
    Defines a dictionary of classifiers with their corresponding objects and hyperparameter grids.

    Parameters:
    - n_jobs (int): Number of jobs to run in parallel for training.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - dict: A dictionary where keys are classifier names and values are lists containing the classifier object
      and a dictionary of hyperparameter grids.
    """

    classifiers_dict = {'RandomForestClassifier':
        [
            RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, class_weight='balanced'),
            {
                'randomforestclassifier__n_estimators': [100],
                'randomforestclassifier__max_depth': [2],
                'randomforestclassifier__min_samples_split': [2],
                'randomforestclassifier__max_samples': [0.7]
            }
        ],
        'DecisionTreeClassifier':
            [DecisionTreeClassifier(random_state=random_state, class_weight='balanced'),
             {
                 # https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
                 'decisiontreeclassifier__criterion': ['gini', 'entropy'],
                 'decisiontreeclassifier__max_depth': [2, 4, 8],
                 'decisiontreeclassifier__min_samples_split': [2, 5],
                 'decisiontreeclassifier__min_samples_leaf': [1, 2],

             }
             ],
        'ExtraTreeClassifier':
            [ExtraTreeClassifier(random_state=random_state, class_weight='balanced'),
             {
                 'extratreeclassifier__criterion': ['gini', 'entropy'],
                 'extratreeclassifier__max_depth': [2, 4, 8, 16],
                 'extratreeclassifier__min_samples_split': [2, 5],
                 'extratreeclassifier__min_samples_leaf': [1, 2],

             }
             ],
        'XGBoostTree':
            [XGBClassifier(seed=random_state),
             {
                 'xgbclassifier__booster': ['gbtree'],
                 'xgbclassifier__max_depth': [2, 4, 6, 8],
                 'xgbclassifier__learning_rate': [0.1, 0.01, 0.05],
                 'xgbclassifier__n_estimators': [100, 500, 1000],
             }
             ],
        'XGBoostLinear':
            [XGBClassifier(seed=random_state),
             {
                 'xgbclassifier__booster': ['gblinear'],
                 'xgbclassifier__learning_rate': [0.1, 0.01, 0.05],
                 'xgbclassifier__n_estimators': [100, 500, 1000],
                 'xgbclassifier__lambda': [0.1, 1, 5, 10],
                 'xgbclassifier__alpha': [0, 0.1, 0.01]
             }
             ],
        'CatBoostClassifier':
            [CatBoostClassifier(random_seed=random_state, auto_class_weights='Balanced'),
             {
                 'catboostclassifier__depth': [4, 6],
                 'catboostclassifier__learning_rate': [0.01, 0.05],
                 'catboostclassifier__iterations': [100, 200],
                 'catboostclassifier__l2_leaf_reg': [3, 5],
                 'catboostclassifier__random_strength': [1, 5, 10]
             }
        ]
    }
    return classifiers_dict


def get_all_clfs(n_jobs, random_state):

    """
    Retrieves the dictionary of classifiers and their hyperparameter grids.

    Parameters:
    - n_jobs (int): Number of jobs to run in parallel for training.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - dict: A dictionary of classifiers and their hyperparameter grids.
    """

    classifiers_dict = classifier_parameters(n_jobs, random_state)

    return classifiers_dict


def define_classifiers(selected_classifiers_name_list, n_jobs, random_state):

    """
    Filters and retrieves the classifiers and their hyperparameter grids based on a list of selected classifier names.

    Parameters:
    - selected_classifiers_name_list (list): List of classifier names to include.
    - n_jobs (int): Number of jobs to run in parallel for training.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - tuple: A tuple containing:
        - list: Names of the selected classifiers.
        - list: Classifier objects for the selected classifiers.
        - list: Hyperparameter grids for the selected classifiers.
    """

    classifiers_dict = classifier_parameters(n_jobs, random_state)
    # a list of parameters for the above classifiers to be used in grid search
    classifiers_list = []
    param_grid_list = []
    for classifier_name in selected_classifiers_name_list:
        if classifier_name in classifiers_dict.keys():
            classifiers_list.append(classifiers_dict[classifier_name][0])
            param_grid_list.append(classifiers_dict[classifier_name][1])
        else:
            print(classifier_name + 'is not a defined classifier name')

    return selected_classifiers_name_list, classifiers_list, param_grid_list
