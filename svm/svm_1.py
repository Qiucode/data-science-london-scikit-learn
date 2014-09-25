"""

"""
"""

"""
from pprint import pprint
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np


from data.data import trainX, trainY


def naive_svm(x,y):
    """
    try a naive svm method!
    :return: numpy, test labels
    """
    preprocess(x)

    # grid search
    params = {'kernel': ['rbf'],
              'C': [0.5, 1, 2, 3, 4, 5],
              'gamma': np.arange(0.01, 0.02, 0.001)}
    svc = SVC()
    clf = GridSearchCV(svc, params, cv=5, n_jobs=-1)
    clf.fit(x, y)
    pprint(clf.grid_scores_)
    print(clf.best_params_, clf.best_score_)

    # create result
    best_clf = clf.best_estimator_

    return best_clf


def preprocess(x):
    """
    do some preprocessing on raw data
    :param x: numpy array
    :return:
    """
    # scale to standard distribution
    scale(x)

def feature_extraction(x,y):
    n_features = x.shape[-1]

    scores = {}
    # using p-value to evaluate features
    scores['p-value'], _ = f_classif(x, y)

    # using Logistic Regression to evaluate features
    scaleX = scale(x, copy=True)
    clf = LogisticRegression(penalty='l1').fit(scaleX, y)
    scores['LogReg'] = clf.coef_[0]

    # using Lasso to evaluate features
    clf = Lasso(0.005).fit(scaleX, y)
    scores['Lasso'] = clf.coef_

    # using LinearSVC
    clf = LinearSVC(penalty='l1', dual=False).fit(scaleX, y)
    scores['svc'] = clf.coef_[0]

    # using ensemble tree
    clf = ExtraTreesClassifier().fit(x, y)
    scores['tree'] = clf.feature_importances_
    feature_list = {}
    for tittle, score in scores.items():
        this_list = score.argsort()
        feature_list[tittle] = this_list[30:40]

    return feature_list



if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
                                    trainX, trainY, test_size = 0.2, random_state = 0)
    feat_list = feature_extraction(X_train,Y_train)
    score={}
    for tittle, index in feat_list.items():
        clf = naive_svm(X_train[:,index], Y_train)
        y = clf.predict(X_test[:, index])
        score[tittle] =1 - np.sum(np.abs(Y_test - y)) / Y_test.shape[0]
    print(score)

