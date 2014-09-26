"""

"""
from data.data import trainX, trainY, testX, write_result
from sklearn import cross_validation
# from sklearn import svm
# import random
from pprint import pprint

from sklearn.svm import SVC
import numpy as np
from sklearn.grid_search import GridSearchCV


def naive_svm():
    """
    try a naive svm method!
    :return: numpy, test labels
    """
    # grid search
    params = {'kernel': ['rbf'],
              'C': [0.5, 1, 2, 3, 4, 5],
              'gamma': np.arange(0.01, 0.02, 0.001)}
    svc = SVC()
    clf = GridSearchCV(svc, params, cv=5, n_jobs=-1)
    clf.fit(trainX, trainY)
    pprint(clf.grid_scores_)
    print(clf.best_params_, clf.best_score_)

    # create result
    best_clf = clf.best_estimator_
    best_clf.fit(trainX, trainY)
    y = best_clf.predict(testX)
    return y



# def evaluate(gt,pt):
#     """
#     :param gt: groundtruth
#     :param pt: predict
#     :return: precision
#     """
#     answer = gt-pt
#     precision = abs(answer).sum
#     return  precision

# def split():
#     """
#     use to divide taindata into two parts
#     :return: train_data,test_data
#     """
#     index = list(range(0,trainX.shape[0]))
#     print(index)
#     index_1 = random.sample(index,int((trainX.shape[0])/2))
#     index_2 = []
#     for i in index:
#         if i not in index_1:
#             index_2.append(i)
#     train_data = trainX[index_1,:]
#     train_label = trainY[index_1]
#     test_data = trainX[index_2,:]
#     test_label = trainY[index_2]
#     return train_data,train_label,test_data,test_label

# def multiple_feature_methods():
#     pca = PCA(n_components = 1)
#     selection  = SelectKBest(k = 1)
#     combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#     X_features = combined_features.fit(trainX, trainY).transform(trainX)
#
#     svc = SVC(kernel = "linear")
#     clf = SVC(kernel = "rbf")
#
#     pipline = Pipeline([("fearture", combined_features),("svm", clf)])
#
#     scores = cross_validation.cross_val_score(
#         pipline, trainX, trainY, cv = 5)
#     print(scores)


    # y = naive_svm()
    # [train_data,train_label,test_data,test_label] = split()

    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    #                                 trainX, trainY, test_size = 0.5, random_state = 0)

    # clf = svm.SVC(kernel = 'rbf', C = 1).fit(X_train, Y_train)
    # clf = svm.SVC(kernel = 'rbf')

    # estimators = [('reduce_dim', PCA()), ('SVM', SVC())]
    # clf = Pipeline(estimators)
    # scores = cross_validation.cross_val_score(
    #     clf, trainX, trainY, cv = 5)
    # print(scores)
    # print(clf.score(X_test, Y_test))
    # TODO 处理程序
    # write_result(y, "naive svm, first edition")
    # multiple_feature_methods()

    # print(clf)
#
# def preprocess(x):
#     """
#     do some preprocessing on raw data
#     :param x: numpy array
#     :return:
#     """
#     # scale to standard distribution
#     scale(x)


if __name__ == '__main__':
    results = naive_svm()

