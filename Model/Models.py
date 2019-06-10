from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection, tree, svm
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.base import clone
import statistics as st
import numpy as np
import pandas as pd

import time

import warnings

warnings.simplefilter('ignore')


class Model:
    name: str
    model: object
    params: dict
    accuracy_list: list
    error_list: list
    model_accuracy: float
    model_error: float
    model_time: float
    X: object
    y: object
    precision: float
    specificity: float
    recall: float
    f1_score: float
    confusion_matrix: list
    tn: int
    fp: int
    fn: int
    tp: int

    def __init__(self, name, model, params, acc_list, err_list):
        self.name = name
        self.model = model
        self.params = params
        self.model_accuracy = 0.0
        self.model_error = 0.0
        self.accuracy_list = acc_list
        self.error_list = err_list

    def get_name(self):
        return self.name

    def get_params(self):
        return self.params.keys()

    def add_accuracy_list(self, val):
        self.accuracy_list.append(val)

    def get_accuracy_list(self):

        return self.accuracy_list

    def add_error_list(self, val):
        self.error_list.append(val)

    def get_error_list(self):
        return self.error_list

    def set_accuracy(self):

        self.model_accuracy = round(st.mean(self.accuracy_list), 2)

    def get_accuracy(self):

        return self.model_accuracy

    def is_initialized(self):
        if self.model_accuracy == 0:
            return False
        else:
            return True

    def set_error(self):
        self.model_error = round(st.mean(self.error_list), 2)

    def get_error(self):
        return self.model_error

    def set_time(self, val):
        self.model_time = val

    def get_time(self):
        return round(self.model_time, 4)

    def load_current_features_target(self, X, y):
        self.X = X
        self.y = y

    def set_score(self, collection_of_indices):

        if not (self.accuracy_list == [] or self.error_list == []):
            self.error_list = []
            self.accuracy_list = []

        start = time.time()

        for i in range(len(collection_of_indices)):
            # splitting up data set
            X_train = self.X.iloc[collection_of_indices[i][0]]
            y_train = self.y.iloc[collection_of_indices[i][0]]
            X_test = self.X.iloc[collection_of_indices[i][1]]
            y_test = self.y.iloc[collection_of_indices[i][1]]

            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test) * 100
            error = 100 * (self.model.predict(X_test) != y_test.iloc[:, 0].tolist()).sum().astype(float) / len(collection_of_indices[i][1])

            self.add_accuracy_list(accuracy)
            self.add_error_list(error)
        # set_acc returns index of value that is used, so it can get the train nad test data for the
        # the specific acc
        self.set_accuracy()
        self.set_error()
        end = time.time()
        self.set_time(end - start)

    def get_predict_prob(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)

        return self.model.predict_proba(X_test)[:, 1]

    def get_predict_prob_new(self,X_test):
        to_array=np.reshape(X_test, (1, -1))
        return self.model.predict_proba(to_array)

    def grid_fit(self):

        model1 = GridSearchCV(self.model, param_grid=self.params, n_jobs=-1, cv=5, error_score=0.0)
        model1.fit(self.X, self.y.values.ravel())

        self.model.set_params(**model1.best_params_)

    def random_search_fit(self):
        model1 = RandomizedSearchCV(self.model, param_distributions=self.params,
                                    n_iter=20, cv=5, iid=False, error_score=0.0)
        model1.fit(self.X, self.y.values.ravel())
        self.model.set_params(**model1.best_params_)

    def set_precision_specificity_recall_f1(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_true = y_test.iloc[:, 0].values

        metrics = precision_recall_fscore_support(y_true, y_pred, average='macro')
        self.precision = round(metrics[0], 4) * 100
        self.recall = round(metrics[1], 4) * 100
        self.f1_score = round(metrics[2], 4) * 100

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true, y_pred).ravel()

        self.specificity = round(self.tn / (self.tn + self.tp) * 100, 4)

    def get_precision_specificity_recall_f1(self):
        return self.precision, self.specificity, self.recall, self.f1_score

    def get_confusion_matrix(self):
        return self.tn, self.fp, self.fn, self.tp

    def forward_selection(self):
        M = self.X.shape

        ## Crossvalidation
        # Create crossvalidation partition for evaluation
        K = 5
        CV = model_selection.KFold(n_splits=K, shuffle=True)

        # Initializing variables
        features = []
        empty = np.zeros(M)
        error_list = []

        labels = self.X.columns.values

        model = self.model
        # First fit and test empty
        k = 0

        for train_index, test_index in CV.split(self.X):
            # extract training and test set for current CV fold
            X_train = empty[train_index, :]
            y_train = self.y.iloc[train_index]
            X_test = empty[test_index, :]
            y_test = self.y.iloc[test_index]

            model.fit(X_train, y_train)

            f1 = f1_score(y_test, model.predict(X_test), average='micro')
            error_list.append(f1)

        best_feature_comb = st.mean(error_list)

        while len(labels) != 0:
            combos = np.zeros(len(labels))
            for train_index, test_index in CV.split(self.X):

                for i in range(len(labels)):
                    list_of_features = features + [labels[i]]

                    # extract training and test set for current CV fold
                    X_train = self.X[list_of_features].iloc[train_index]
                    y_train = self.y.iloc[train_index]
                    X_test = self.X[list_of_features].iloc[test_index]
                    y_test = self.y.iloc[test_index]

                    model.fit(X_train, y_train)
                    f1 = f1_score(y_test, model.predict(X_test), average='micro')
                    combos[i] += f1
            combos[:] = [(x / K) * 100 for x in combos]

            best_combo = combos.argmax()

            if combos[best_combo] >= best_feature_comb:
                best_feature_comb = combos[best_combo]
                features.append(labels[best_combo])
                labels = np.delete(labels, best_combo)
            else:
                break
        return features

    def get_parameters_to_tune(self):
        return self.params.keys()

    def get_parameters(self):

        return self.model.get_params()

    def fit_for_finale(self):
        self.model.fit(self.X, self.y)

    def set_parameter(self, key, value):
        old_value = self.model.get_params()[key]
        try:
            if "." in value:
                value = float(value)

            elif (value.isdigit()):
                value = int(value)


            elif value == "None":
                value = None

            elif value == "-1":
                value = -1

            testmodel=clone(self.model)
            testmodel.set_params(**{key: value})
            testmodel.fit(self.X, self.y)

            self.model.set_params(**{key: value})



            return True

        except ValueError:
            return False
        except TypeError:
            return False

    def get_value_parameter_and_type(self, param):
        return self.model.get_params()[param],type(self.model.get_params()[param])




class KNNModel(Model):
    def __init__(self):
        params = {'n_neighbors': [5, 6, 7, 8, 9, 10],
                  'leaf_size': [1, 2, 3, 5],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'n_jobs': [-1, None]}
        Model.__init__(self, "KNN", KNeighborsClassifier(), params, [], [])


class SVCModel(Model):
    def __init__(self):
        params = {'C': [6, 7, 8, 9, 10, 11, 12],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
        Model.__init__(self, "SVC", SVC(probability=True), params, [], [])


class DecTreeModel(Model):
    def __init__(self):
        params = {'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                  'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                  'random_state': [123]}

        Model.__init__(self, "Decision Tree", tree.DecisionTreeClassifier(), params, [], [])


class NBModel(Model):
    def __init__(self):
        params = {}
        Model.__init__(self, "Naive Bayes", GaussianNB(), params, [], [])


class GBCModel(Model):
    def __init__(self):
        params = {'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                  'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                  'max_depth': np.linspace(1, 32, 32, endpoint=True),
                  'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)}
        Model.__init__(self, 'Gradient Boosting Classifier', GradientBoostingClassifier(), params, [], [])


class RandomForestModel(Model):
    def __init__(self):
        params = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
                  'min_samples_leaf': [1, 2, 3],
                  'min_samples_split': [3, 4, 5, 6, 7],
                  'random_state': [123],
                  'n_jobs': [-1, None]}

        Model.__init__(self, 'Random Forest', RandomForestClassifier(), params, [], [])


class LRModel(Model):
    def __init__(self):
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2'], 'dual': [False, True]}

        Model.__init__(self, 'Logistic Regression', LogisticRegression(), params, [], [])


class NNModel(Model):
    def __init__(self):
        params = {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
            'activation': ['tanh', 'relu', 'identity', 'logistic'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': [0.00001, 0.05],
            'learning_rate': ['constant', 'invscaling', 'adaptive']}
        Model.__init__(self, 'Neural Network', MLPClassifier(), params, [], [])

