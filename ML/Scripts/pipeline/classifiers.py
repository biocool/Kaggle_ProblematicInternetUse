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
    This function generates a dictionary containing a set of classifiers with their corresponding
    hyperparameters for model tuning.

    @param n_jobs int The number of CPU cores to use for the execution of the classifiers.
                      '-1' means using all processors.
    @param random_state int

    Returns:
        dict: A dictionary where the keys are the classifier names and the values are lists consisting
              of classifier objects and their associated hyperparameters.
    """

    # note: SN of RadiusNeighborsClassifier was very low in 3 different cancers.

    classifiers_dict = {'RandomForestClassifier':
        [
            RandomForestClassifier(warm_start=True, random_state=random_state, n_jobs=n_jobs),
            {
                # , 500, 1000
                'randomforestclassifier__n_estimators': [100],
                'randomforestclassifier__max_depth': [2, 4, 8, 16],
                'randomforestclassifier__min_samples_split': [2, 5],
                'randomforestclassifier__max_samples': [0.7, 0.8]
            }
        ],
        'LogisticRegression':
            [
                LogisticRegression(warm_start=True, random_state=random_state, class_weight='balanced',
                                   n_jobs=n_jobs),
                {
                    # https://levelup.gitconnected.com/a-comprehensive-analysis-of-hyperparameter-optimization-in-logistic-regression-models-521564c1bfc0#:~:text=Logistic%20Regression%20Hyperparameters,sklearn%20documentation)%20%5B4%5D.
                    # I did not use the None penalty function because the number of samples is not large
                    # enough and we might encounter overfitting.
                    # Based on runs on three cancer types, the best model was 'max_iter' 100 and
                    # not other values (500 or 1000 or 2000).
                    # If you set 'n_jobs' to -1, it means that you want to use all processors.
                    # But when the solver is 'liblinear', this won't have any effect because 'liblinear'
                    # doesn't support parallelism.
                    'logisticregression__penalty': ['l1', 'l2'],
                    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'logisticregression__solver': ['lbfgs', 'liblinear']
                }
            ],
        'GaussianProcessClassifier':
            [GaussianProcessClassifier(random_state=random_state, n_jobs=n_jobs),
             {
                 'gaussianprocessclassifier__warm_start': [True, False]
             }
             ],
        'QuadraticDiscriminantAnalysis':
            [QuadraticDiscriminantAnalysis(store_covariance=True),
             {
                 'quadraticdiscriminantanalysis__reg_param': [0, 0.1, 0.01, 0.001, 0.5],
             }
             ],
        'LinearDiscriminantAnalysis':
            [LinearDiscriminantAnalysis(),
             {
                 'lineardiscriminantanalysis__solver': ['lsqr', 'eigen', 'svd'],
                 'lineardiscriminantanalysis__shrinkage': ['auto', None],
                 'lineardiscriminantanalysis__store_covariance': [False, True]

             }
             ],
        'DecisionTreeClassifier':
            [DecisionTreeClassifier(random_state=random_state),
             {
                 # https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
                 'decisiontreeclassifier__criterion': ['gini', 'entropy'],
                 'decisiontreeclassifier__max_depth': [2, 4, 8],
                 'decisiontreeclassifier__min_samples_split': [2, 5],
                 'decisiontreeclassifier__min_samples_leaf': [1, 2],

             }
             ],
        'ExtraTreeClassifier':
            [ExtraTreeClassifier(random_state=random_state),
             {
                 'extratreeclassifier__criterion': ['gini', 'entropy'],
                 'extratreeclassifier__max_depth': [2, 4, 8, 16],
                 'extratreeclassifier__min_samples_split': [2, 5],
                 'extratreeclassifier__min_samples_leaf': [1, 2],

             }
             ],
        'GradientBoostingClassifier':
            [GradientBoostingClassifier(random_state=random_state),
             {
                 'gradientboostingclassifier__n_estimators': [100, 500, 1000],
                 'gradientboostingclassifier__max_depth': [3, 4, 5],
                 'gradientboostingclassifier__min_samples_split': [2, 5],
                 'gradientboostingclassifier__min_samples_leaf': [1, 2, 3],
                 'gradientboostingclassifier__max_features': ['sqrt', 'log2']
             }
             ],
        'C-Support Vector Classifier':
            [SVC(random_state=random_state, probability=True),
             {
                 'svc__C': [0.5, 0.8, 1],
                 'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                 'svc__gamma': ['scale', 'auto']
             }
             ],
        'Nu-Support Vector Classifier':
            [NuSVC(random_state=random_state, probability=True),
             {
                 'nusvc__nu': [0.5, 0.8, 0.1],
                 'nusvc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                 'nusvc__gamma': ['scale', 'auto']
             }
             ],
        'XGBoostTree':
            [XGBClassifier(seed=random_state),
             {
                 'xgbclassifier__booster': ['gbtree'],
                 'xgbclassifier__max_depth': [2, 4, 6, 8],
                 'xgbclassifier__learning_rate': [0.1, 0.01, 0.05],
                 'xgbclassifier__n_estimators': [100, 200, 500, 1000],
             }
             ],
        'XGBoostLinear':
            [XGBClassifier(seed=random_state),
             {
                 'xgbclassifier__booster': ['gblinear'],
                 'xgbclassifier__learning_rate': [0.1, 0.01, 0.05],
                 'xgbclassifier__n_estimators': [100, 200, 500, 1000],
                 'xgbclassifier__lambda': [0.1, 1, 5, 10],
                 'xgbclassifier__alpha': [0, 0.1, 0.01]
             }
             ],
        'CatBoostClassifier':
            [CatBoostClassifier(random_seed=random_state),
             {
                 'catboostclassifier__depth': [4, 6],
                 'catboostclassifier__learning_rate': [0.01, 0.05],
                 'catboostclassifier__iterations': [100, 200],
                 'catboostclassifier__l2_leaf_reg': [3, 5],
                 'catboostclassifier__random_strength': [1, 5, 10]
             }
             ],
        'MLPClassifier':
            [MLPClassifier(random_state=random_state, early_stopping=True,
                           validation_fraction=0.2),
             {
                 'mlpclassifier__hidden_layer_sizes': [(20, 20), (30, 30), (40, 40), (20, 20, 20),
                                                       (30, 30, 30), (40, 40, 40), (30, 20, 10)],
                 'mlpclassifier__activation': ['tanh', 'relu'],
                 'mlpclassifier__solver': ['sgd', 'adam'],
                 'mlpclassifier__alpha': [0.0001, 0.05],
                 'mlpclassifier__learning_rate': ['constant', 'adaptive'],
             }
             ],
    }
    return classifiers_dict


def get_all_clfs(n_jobs, random_state):
    classifiers_dict = classifier_parameters(n_jobs, random_state)
    return classifiers_dict


def define_classifiers(selected_classifiers_name_list, n_jobs, random_state):
    """
    This function defines the classifiers that will be used based on the list of classifier names provided.

    @param selected_classifiers_name_list list A list of classifier names to be used.
    @param n_jobs int The number of CPU cores to use. Default is -1, which means using all processors.
    @param random_state int

    Returns:
        tuple: A tuple containing three lists:
                - selected classifier names
                - classifier objects
                - parameter grid for grid search
    """

    classifiers_dict = classifier_parameters(n_jobs, random_state)
    # a list of parameters for the above classifiers to be used in grid search
    classifiers_list = []
    param_grid_list = []
    for classifier_name in selected_classifiers_name_list:
        if classifier_name in classifiers_dict.keys():
            classifiers_list.append(classifiers_dict[classifier_name][1])
            param_grid_list.append(classifiers_dict[classifier_name][2])
        else:
            print(classifier_name + 'is not a defined classifier name')

    return selected_classifiers_name_list, classifiers_list, param_grid_list
