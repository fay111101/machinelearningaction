#!/usr/bin/python
# -*- coding: UTF-8 -*-


"""
    XXX
    ~~~~~~~~~~
    XXX is a module for XXX
    :copyright: (c) Copyright 2017 by Xiuhong Fei.
    :license: iscas, see LICENSE for more details.
"""
import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# print(df.head())
# 定义特征(X)和标签(y)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
# scikits-learn中的cross_validation.train_test_split来创建训练和测试样本
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# 定义K近邻分类器
clf = neighbors.KNeighborsClassifier()
# 训练分类器
clf.fit(X_train, y_train)
# 测试
accuracy = clf.score(X_test, y_test)
print('sklearn accuracy',accuracy)

