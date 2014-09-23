"""

"""
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from data.data import trainX, trainY, testX, write_result
import numpy as np
from sklearn import cross_validation
# from sklearn import svm
# import random

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def naive_svm():
    """
    try a naive svm method!
    :return: numpy, test labels
    """

    # TODO 特征预处理
    # preprocess(trainX)

    clf = LinearSVC()
    clf.fit(trainX, trainY)
    y = clf.predict(testX)
    return y

def preprocess(data):
    """
    data preprocessing
    :param data:
    :return:
    """
    data_normalized = preprocessing.normalize(data,norm = 'l2')

    return data_normalized

def evaluate(gt,pt):
    """
    :param gt: groundtruth
    :param pt: predict
    :return: precision
    """
    answer = gt-pt
    precision = abs(answer).sum
    return  precision

def split():
    """
    use to divide taindata into two parts
    :return: train_data,test_data
    """
    index = list(range(0,trainX.shape[0]))
    print(index)
    index_1 = random.sample(index,int((trainX.shape[0])/2))
    index_2 = []
    for i in index:
        if i not in index_1:
            index_2.append(i)
    train_data = trainX[index_1,:]
    train_label = trainY[index_1]
    test_data = trainX[index_2,:]
    test_label = trainY[index_2]
    return train_data,train_label,test_data,test_label


if __name__ == '__main__':
    # y = naive_svm()
    # [train_data,train_label,test_data,test_label] = split()

    # X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    #                                 trainX, trainY, test_size = 0.5, random_state = 0)

    # clf = svm.SVC(kernel = 'rbf', C = 1).fit(X_train, Y_train)
    # clf = svm.SVC(kernel = 'rbf')

    estimators = [('reduce_dim', PCA()), ('SVM', SVC())]
    clf = Pipeline(estimators)
    scores = cross_validation.cross_val_score(
        clf, trainX, trainY, cv = 5)
    print(scores)
    # print(clf.score(X_test, Y_test))
    # TODO 处理程序
    # write_result(y, "naive svm, first edition")


    # print(clf)

