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
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
import pandas as pd
import random
from collections import Counter

style.use('fivethirtyeight')


def test_data():
    # 然后创建数据集：
    dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_features = [5, 7]
    # dataset(数据集)只是一个Python字典，其中的键看作类，后面的值看做这个类相关的数据点，new_features是我们想要测试的数据。
    # 我们可以做一个快速的图表:
    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color=i)
    plt.scatter(new_features[0], new_features[1], s=10)
    plt.show()


def test_data1():
    dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_features = [5, 7]
    for i in dataset:
        for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color=i)
    plt.scatter(new_features[0], new_features[1], s=100)

    result = k_nearest_neighbors(dataset, new_features)
    plt.scatter(new_features[0], new_features[1], s=100, color=result)
    plt.show()


def k_nearest_neighbors(data, predict, k=3):
    """
    不使用sklearn库的KNN，自定义KNN
    :param data:
    :param predict:
    :param k:
    :return:
    """
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = sqrt((features[0] - predict[0]) ** 2 + (features[1] - predict[1]) ** 2)
            distances.append([euclidean_distance, group])
        #       i[0]为距离，i[1]为类别，我们需要的是类别
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

def train_data():
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # 一些数据点虽然是数字，但是字符串数据类型，我们将整个dataframe转换为float
    full_data = df.astype(float).values.tolist()
    # 接下来，我们将对数据进行shuffle，然后将其拆分:
    random.shuffle(full_data)
    test_size = 0.2
    # 2代表良性肿瘤，4代表恶性肿瘤
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]
    # 现在我们有了和测试集相同的字典，其中的键是类，值是属性。
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0
    # 我k值选择了5，因为这是Scikit学习KNeighborsClassifier的默认值。
    for group in test_set:
        for data in test_set[group]:
            vote = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print('Accuracy:', correct / total)


if __name__=='__main__':
    train_data()