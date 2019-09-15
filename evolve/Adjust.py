# -*- coding: utf-8 -*-
"""
计算个体的适应度

@author: youhaolin
@email: cador.ai@aliyun.com
"""

from sklearn import tree
from sklearn import linear_model
import numpy as np


def get_adjust(std_error, y, indiv_data, handle):
    """计算适应度，通过外部定义的handle来处理，同时适用于分类和回归问题"""
    X = indiv_data
    cur_error = handle(X, y)
    return std_error - cur_error if std_error > cur_error else 0


def evaluation_classify(X, y):
    """建立分类问题的评估方法"""
    clf = tree.DecisionTreeClassifier(random_state=0)
    errors = []
    for i in range(X.shape[0]):
        index = [e for e in range(X.shape[0])]
        index.remove(i)
        x_train = X.iloc[index, :]
        x_test = X.iloc[[i], :]
        y_train = y[index]
        y_test = y[i]
        clf.fit(x_train, y_train)
        if clf.predict(x_test)[0] != y_test:
            errors.append(1)
        else:
            errors.append(0)
    return np.sum(errors)/len(errors)


def evaluation_regression(X, y):
    """建立回归问题的评估方法"""
    reg = linear_model.LinearRegression()
    errors = 0
    for i in range(X.shape[0]):
        index = [e for e in range(X.shape[0])]
        index.remove(i)
        x_train = X.iloc[index, :]
        x_test = X.iloc[[i], :]
        y_train = y[index]
        y_test = y[i]
        reg.fit(x_train, y_train)
        errors = errors + (y_test - reg.predict(x_test)[0])**2
    return errors
