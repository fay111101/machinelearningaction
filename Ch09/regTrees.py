'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
import numpy as np
import pickle


#
#
# def loadDataSet(fileName):  # general function to parse tab -delimited floats
#     dataMat = []  # assume last column is target value
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         fltLine = map(float, curLine)  # map all elements to float()
#         dataMat.append(fltLine)
#     return dataMat
#
#
# def binSplitDataSet(dataSet, feature, value):
#     """
#     该函数用于根据给定的特征和特征值将数据集划分成两个部分
#     :param dataSet:
#     :param feature: 给定的特征
#     :param value: 给定的特征值
#     :return:
#     """
#     mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
#     mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
#     return mat0, mat1
#
#
# def regLeaf(dataSet):  # returns the value used for each leaf
#     """
#     叶节点生成函数
#     当chooseBestSplit函数确定不再对数据进行切分时，将调用该regTree函数来得到叶节点的模型
#     在回归树中，该模型其实是目标变量的均值
#     :param dataSet:
#     :return: 数据集列表最后一列特征值的均值作为叶节点返回
#     """
#     return mean(dataSet[:, -1])
#
#
# def regErr(dataSet):
#     """
#     误差计算函数
#
#     在给定数据集上计算目标变量的平方误差
#     :param dataSet:
#     :return:
#     """
#     # 计算数据集最后一列特征值的均方差*数据集样本数，得到总方差返回
#     return var(dataSet[:, -1]) * shape(dataSet)[0]
#
#
# def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
#     """
#     选择最佳切分特征和最佳特征取值函数
#
#     该函数用于找到数据集切分的最佳位置
#     对于每个特征：
#         对于每个特征值：
#             将数据集切分成两份
#             计算切分的误差
#             如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差，返回最佳切分的特征和阈值
#
#     :param dataSet: 数据集
#     :param leafType: 生成叶节点的类型，默认为回归树类型
#     :param errType: 计算误差的类型，默认为总方差类型
#     :param ops: 用户指定的参数，默认tolS=1.0，tolN=4
#     :return:返回特征编号和切分特征
#     """
#     # 容忍误差下降值1，最少切分样本数4
#     tolS = ops[0]
#     tolN = ops[1]
#     # if all the target variables are the same value: quit and return value
#     # 数据集最后一列所有的值都相同
#     if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
#         # 1）最优特征返回none，将该数据集最后一列计算均值作为叶节点值返回
#         return None, leafType(dataSet)
#     # 数据集的行与列
#     m, n = shape(dataSet)
#     # the choice of the best feature is driven by Reduction in RSS error from mean
#     # 计算未切分前数据集的误差
#     S = errType(dataSet)
#     # 初始化最小误差；最佳切分特征索引；最佳切分特征值，inf为numpy包下的无穷
#     bestS = inf;
#     bestIndex = 0;
#     bestValue = 0
#     # 遍历数据集所有的特征，除最后一列目标变量值
#     for featIndex in range(n - 1):
#         # 遍历该特征的每一个可能取值
#         for splitVal in set(dataSet[:, featIndex]):
#             # 以该特征，特征值作为参数对数据集进行切分为左右子集
#             mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
#             # 如何左分支子集样本数小于tolN或者右分支子集样本数小于tolN，跳出本次循环
#             if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
#             # 计算切分后的误差，即均方差和
#             newS = errType(mat0) + errType(mat1)
#             # 保留最小误差及对应的特征及特征值
#             if newS < bestS:
#                 bestIndex = featIndex
#                 bestValue = splitVal
#                 bestS = newS
#     # if the decrease (S-bestS) is less than a threshold don't do the split
#     # 如果切分后比切分前误差下降值未达到tolS
#     if (S - bestS) < tolS:
#         # 2）不需切分，直接返回目标变量均值作为叶节点
#         return None, leafType(dataSet)  # exit cond 2
#     # 检查最佳特征及特征值是否满足不切分条件
#     mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
#     # 3）如果某个子集的大小小于用户定义的参数tolN，那么也不切分
#     if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
#         return None, leafType(dataSet)
#     # 返回最佳切分特征及最佳切分特征取值
#     return bestIndex, bestValue  # returns the best feature to split on
#     # and the value used for that split
#
#
# def create_cart_tree(dataSet, leafType=regLeaf, errType=regErr,
#                ops=(1, 4)):  # assume dataSet is NumPy Mat so we can array filtering
#     """
#     CART算法构建函数
#     找到最佳的切分特征：
#         如果该节点不能再分，将该节点存为叶节点
#         执行二元切分
#         在左子树递归调用createTree（）方法
#         在右子树递归调用createTree（）方法
#     :param dataSet:
#     :param leafType: 生成叶节点的类型 1 回归树：叶节点为常数值 2 模型树：叶节点为线性模型
#     :param errType: 计算误差的类型 1 回归错误类型：总方差=均方差*样本数 2 模型错误类型：预测误差(y-yHat)平方的累加和
#     :param ops: 用户指定的参数，包含tolS：容忍误差的降低程度 tolN：切分的最少样本数
#     :return:
#     """
#     # 选取最佳分割特征和特征值
#     feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # choose the best split
#     # 如果特征为none，直接返回叶节点值
#     if feat == None:
#         return val  # if the splitting hit a stop condition return val
#     # 树的类型是字典类型
#     retTree = {}
#     # 树字典的一个元素是切分的最佳特征
#     retTree['spInd'] = feat
#     # 第二个元素是最佳特征对应的最佳切分特征值
#     retTree['spVal'] = val
#     # 根据特征索引及特征值对数据集进行二元拆分，并返回拆分的两个数据子集
#     lSet, rSet = binSplitDataSet(dataSet, feat, val)
#     # 第三个元素是树的左分支，通过lSet子集递归生成左子树
#     retTree['left'] = create_cart_tree(lSet, leafType, errType, ops)
#     # 第四个元素是树的右分支，通过rSet子集递归生成右子树
#     retTree['right'] = create_cart_tree(rSet, leafType, errType, ops)
#     # 返回生成的数字典
#     return retTree
#
#
# # 后减枝方法
#
# def isTree(obj):
#     """
#     根据目标数据的存储类型是否为字典型，是返回true，否则返回false
#     :param obj:
#     :return:
#     """
#     return (type(obj).__name__ == 'dict')
#
#
# def getMean(tree):
#     """
#     获取均值函数
#
#     递归函数，它从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值
#     :param tree:
#     :return:
#     """
#     # 树字典的右分支为字典类型：递归获得右子树的均值
#     if isTree(tree['right']): tree['right'] = getMean(tree['right'])
#     # 递归直至找到两个叶节点，求二者的均值返回
#     if isTree(tree['left']): tree['left'] = getMean(tree['left'])
#     return (tree['left'] + tree['right']) / 2.0
#
#
# def prune(tree, testData):
#     """
#
#     CART生成算法如下：
#
# 输入：训练数据集DD，停止计算的条件：
# 输出：CART决策树。
#
#     根据训练数据集，从根结点开始，递归地对每个结点进行以下操作，构建二叉决策树：
#         设结点的训练数据集为DD，计算现有特征对该数据集的Gini系数。此时，对每一个特征AA，
#         对其可能取的每个值aa，根据样本点对A=aA=a的测试为“是”或 “否”将DD分割成D1D1和D2D2两部分，
#         计算A=aA=a时的Gini系数。在所有可能的特征AA以及它们所有可能的切分点aa中，选择Gini系数最小
#         的特征及其对应的切分点作为最优特征与最优切分点。依最优特征与最优切分点，从现结点生成两个子
#         结点，将训练数据集依特征分配到两个子结点中去。
#         对两个子结点递归地调用步骤l~2，直至满足停止条件。
#
#     算法停止计算的条件是结点中的样本个数小于预定阈值，或样本集的Gini系数小于预定阈值（样本基本属于同一类），
#     或者没有更多特征。
#
#     后剪枝函数伪代码：
#      基于已经存在的树切分测试数据：
#         如果存在任一子集是一棵树，则在该子集递归剪枝过程
#         计算将当前两个叶节点合并后的误差
#         计算不合并的误差
#         如果合并会降低误差的话，就将叶节点合并
#     :param tree: 树字典
#     :param testData: 用于剪枝的测试集
#     :return:
#     """
#     # 测试集为空，直接对树相邻叶子结点进行求均值操作
#     if shape(testData)[0] == 0:
#         return getMean(tree)  # if we have no test data collapse the tree
#     # 左右分支中有非叶子结点类型
#     if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
#         # 利用当前树的最佳切分点和特征值对测试集进行树构建过程
#         lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
#     # 左分支非叶子结点，递归利用测试数据的左子集对做分支剪枝
#     if isTree(tree['left']):
#         tree['left'] = prune(tree['left'], lSet)
#     # 同理，右分支非叶子结点，递归利用测试数据的右子集对做分支剪枝
#     if isTree(tree['right']):
#         tree['right'] = prune(tree['right'], rSet)
#     # if they are now both leafs, see if we can merge them
#     # 左右分支都是叶节点
#     if not isTree(tree['left']) and not isTree(tree['right']):
#         # 利用该子树对应的切分点对测试数据进行切分(树构建)
#         lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
#         # 如果这两个叶节点不合并，计算误差，即（实际值-预测值）的平方和
#         errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
#                        sum(power(rSet[:, -1] - tree['right'], 2))
#         # 求两个叶结点值的均值
#         treeMean = (tree['left'] + tree['right']) / 2.0
#         # 如果两个叶节点合并，计算合并后误差,即(真实值-合并后值）平方和
#         errorMerge = sum(power(testData[:, -1] - treeMean, 2))
#         # 合并后误差小于合并前误差
#         if errorMerge < errorNoMerge:
#             # 和并两个叶节点，返回合并后节点值
#             print("merging")
#             return treeMean
#         # 否则不合并，返回该子树
#         else:
#             return tree
#     # 不合并，直接返回树
#     else:
#         return tree
#
#
# def linearSolve(dataSet):  # helper function used in two places
#     """
#     模型树叶节点生成函数
#     :param dataSet:
#     :return:
#     """
#     m, n = shape(dataSet)
#     X = mat(ones((m, n)));
#     Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
#     X[:, 1:n] = dataSet[:, 0:n - 1];
#     Y = dataSet[:, -1]  # and strip out Y
#     xTx = X.T * X
#     if linalg.det(xTx) == 0.0:
#         raise NameError('This matrix is singular, cannot do inverse,\n\
#         try increasing the second value of ops')
#     ws = xTx.I * (X.T * Y)
#     return ws, X, Y
#
#
# def modelLeaf(dataSet):  # create linear model and return coeficients
#     """
#
#     :param dataSet:
#     :return:
#     """
#     ws, X, Y = linearSolve(dataSet)
#     return ws
#
#
# def modelErr(dataSet):
#     ws, X, Y = linearSolve(dataSet)
#     yHat = X * ws
#     return sum(power(Y - yHat, 2))
#
# def regressData(filename):
#     fr=open(filename)
#     return pickle.load(fr)
#
# #------------CART预测子函数------------#
#
# # 用树回归进行预测代码
# def regTreeEval(model, inDat):
#     """
#
#     :param model:
#     :param inDat: 为采样数为1的特征行向量
#     :return:
#     """
#     # 回归树的叶节点为float型常量
#     return float(model)
#
#
# def modelTreeEval(model, inDat):
#     """
#     #模型树的叶节点浮点型参数的线性方程
#     :param model:
#     :param inDat:为采样数为1的特征行向量
#     :return:
#     """
#     n = shape(inDat)[1]
#     X = mat(ones((1, n + 1)))
#     X[:, 1:n + 1] = inDat
#     return float(X * model)
#
#
# def treeForeCast(tree, inData, modelEval=regTreeEval):
#     """
#     #树预测
#     :param tree: 树回归模型
#     :param inData: 输入数据
#     :param modelEval: 叶节点生成类型，需指定，默认回归树类型
#     :return:
#     """
#     # 如果当前树为叶节点，生成叶节点
#     if not isTree(tree): return modelEval(tree, inData)
#     # 非叶节点，对该子树对应的切分点对输入数据进行切分
#     if inData[tree['spInd']] > tree['spVal']:
#         # 该树的左分支为非叶节点类型
#         if isTree(tree['left']):
#             # 递归调用treeForeCast函数继续树预测过程，直至找到叶节点
#             return treeForeCast(tree['left'], inData, modelEval)
#         else:
#             # 左分支为叶节点，生成叶节点
#             return modelEval(tree['left'], inData)
#     # 小于切分点值的右分支
#     else:
#         # 非叶节点类型
#         if isTree(tree['right']):
#             # 继续递归treeForeCast函数寻找叶节点
#             return treeForeCast(tree['right'], inData, modelEval)
#         else:
#             # 叶节点，生成叶节点类型
#             return modelEval(tree['right'], inData)
#
#
# def createForeCast(tree, testData, modelEval=regTreeEval):
#     """
#     #创建预测树
#     :param tree:
#     :param testData:
#     :param modelEval:
#     :return:
#     """
#     # 测试集样本数
#     m = len(testData)
#     # 初始化行向量各维度值为1
#     yHat = mat(zeros((m, 1)))
#     # 遍历每个样本
#     for i in range(m):
#         # 利用树预测函数对测试集进行树构建过程，并计算模型预测值
#         yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
#     # 返回预测值
#     return yHat
def loadDataSet(filename):
    '''
    输入：文件的全路径
    功能：将输入数据保存在datamat中
    输出：datamat
    '''
    fr = open(filename)
    datamat = []
    for line in fr.readlines():
        cutLine = line.strip().split('\t')
        # map()函数起到映射的作用，将curLine中的元素变为float类型
        floatLine = list(map(float, cutLine))
        datamat.append(floatLine)
    return datamat


def binarySplitDataSet(dataset, feature, value):
    '''
    输入：数据集，数据集中某一特征列，该特征列中的某个取值
    功能：将数据集按特征列的某一取值换分为左右两个子数据集
    输出：左右子数据集
    '''
    matLeft = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
    matRight = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
    return matLeft, matRight


# --------------回归树所需子函数---------------#

def regressLeaf(dataset):
    '''
    输入：数据集
    功能：求数据集输出列的均值
    输出：对应数据集的叶节点
    '''
    return np.mean(dataset[:, -1])


def regressErr(dataset):
    '''
    输入：数据集(numpy.mat类型)
    功能：求数据集划分左右子数据集的误差平方和之和
    输出: 数据集划分后的误差平方和
    '''
    # 由于回归树中用输出的均值作为叶节点，所以在这里求误差平方和实质上就是方差
    return np.var(dataset[:, -1]) * np.shape(dataset)[0]


def regressData(filename):
    fr = open(filename)
    return pickle.load(fr)


# --------------回归树子函数  END  --------------#

def chooseBestSplit(dataset, leafType=regressLeaf, errType=regressErr, threshold=(1, 4)):  # 函数做为参数，挺有意思
    thresholdErr = threshold[0];
    thresholdSamples = threshold[1]
    # 当数据中输出值都相等时，feature = None,value = 输出值的均值（叶节点）
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataset)
    m, n = np.shape(dataset)
    Err = errType(dataset)
    bestErr = np.inf;
    bestFeatureIndex = 0;
    bestFeatureValue = 0
    for featureindex in range(n - 1):
        for featurevalue in dataset[:, featureindex]:
            matLeft, matRight = binarySplitDataSet(dataset, featureindex, featurevalue)
            if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
                continue
            temErr = errType(matLeft) + errType(matRight)
            if temErr < bestErr:
                bestErr = temErr
                bestFeatureIndex = featureindex
                bestFeatureValue = featurevalue
    # 检验在所选出的最优划分特征及其取值下，误差平方和与未划分时的差是否小于阈值，若是，则不适合划分
    if (Err - bestErr) < thresholdErr:
        return None, leafType(dataset)
    matLeft, matRight = binarySplitDataSet(dataset, bestFeatureIndex, bestFeatureValue)
    # 检验在所选出的最优划分特征及其取值下，划分的左右数据集的样本数是否小于阈值，若是，则不适合划分
    if (np.shape(matLeft)[0] < thresholdSamples) or (np.shape(matRight)[0] < thresholdSamples):
        return None, leafType(dataset)
    return bestFeatureIndex, bestFeatureValue


def createCARTtree(dataset, leafType=regressLeaf, errType=regressErr, threshold=(1, 4)):
    '''
    输入：数据集dataset，叶子节点形式leafType：regressLeaf（回归树）、modelLeaf（模型树）
         损失函数errType:误差平方和也分为regressLeaf和modelLeaf、用户自定义阈值参数：
         误差减少的阈值和子样本集应包含的最少样本个数
    功能：建立回归树或模型树
    输出：以字典嵌套数据形式返回子回归树或子模型树或叶结点
    '''
    feature, value = chooseBestSplit(dataset, leafType, errType, threshold)
    # 当不满足阈值或某一子数据集下输出全相等时，返回叶节点
    if feature == None:
        return value
    returnTree = {}
    returnTree['bestSplitFeature'] = feature
    returnTree['bestSplitFeatValue'] = value
    leftSet, rightSet = binarySplitDataSet(dataset, feature, value)
    returnTree['left'] = createCARTtree(leftSet, leafType, errType, threshold)
    returnTree['right'] = createCARTtree(rightSet, leafType, errType, threshold)
    return returnTree


# ----------回归树剪枝函数----------#
def isTree(obj):  # 主要是为了判断当前节点是否是叶节点
    return (type(obj).__name__ == 'dict')


def getMean(tree):  # 树就是嵌套字典
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    剪枝函数
    :param tree: 需要剪枝的树
    :param testData: 测试数据
    :return:
    """
    if np.shape(testData)[0] == 0:
        return getMean(tree)  # 存在测试集中没有训练集中数据的情况

    if isTree(tree['left']) or isTree(tree['right']):
        leftTestData, rightTestData = binarySplitDataSet(testData, tree['bestSplitFeature'], tree['bestSplitFeatValue'])
    # 递归调用prune函数对左右子树,注意与左右子树对应的左右子测试数据集
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], leftTestData)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rightTestData)
    # 当递归搜索到左右子树均为叶节点时，计算测试数据集的误差平方和
    if not isTree(tree['left']) and not isTree(tree['right']):
        leftTestData, rightTestData = binarySplitDataSet(testData, tree['bestSplitFeature'], tree['bestSplitFeatValue'])
        errorNOmerge = sum(np.power(leftTestData[:, -1] - tree['left'], 2)) + sum(
            np.power(rightTestData[:, -1] - tree['right'], 2))
        errorMerge = sum(np.power(testData[:, 1] - getMean(tree), 2))
        if errorMerge < errorNOmerge:
            print('Merging')
            return getMean(tree)
        else:
            return tree
    else:
        return tree


# ---------回归树剪枝END-----------#

# -----------模型树子函数-----------#
def linearSolve(dataset):
    m, n = np.shape(dataset)
    X = np.mat(np.ones((m, n)));
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataset[:, 0:(n - 1)]
    Y = dataset[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of threshold')
        ws = xTx.I * (X.T * Y)
        return ws, X, Y


def modelLeaf(dataset):
    ws, X, Y = linearSolve(dataset)
    return ws


def modelErr(dataset):
    ws, X, Y = linearSolve(dataset)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))


# ------------模型树子函数END-------#

# ------------CART预测子函数------------#

def regressEvaluation(tree, inputData):
    # 只有当tree为叶节点时，才会输出
    return float(tree)


def modelTreeEvaluation(model, inputData):
    # inoutData为采样数为1的特征行向量
    n = np.shape(inputData)
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inputData
    return float(X * model)


def treeForeCast(tree, inputData, modelEval=regressEvaluation):
    if not isTree(tree):
        return modelEval(tree, inputData)
    print(tree['bestSplitFeature'],type(tree['bestSplitFeature']))
    print(inputData)
    # print(inputData[0])
    if inputData[tree['bestSplitFeature']] <= tree['bestSplitFeatValue']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inputData, modelEval)
        else:
            return modelEval(tree['left'], inputData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inputData, modelEval)
        else:
            return modelEval(tree['right'], inputData)


def createForeCast(tree, testData, modelEval=regressEvaluation):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat = treeForeCast(tree, testData[i], modelEval)
    return yHat


# -----------CART预测子函数 END------------#


if __name__ == '__main__':
    trainfilename = 'bikeSpeedVsIq_test.txt'
    testfilename = 'bikeSpeedVsIq_train.txt'
    # trainDataset = regressData(trainfilename)
    # testDataset = regressData(testfilename)
    trainDataset1 = loadDataSet(trainfilename)
    testDataset1 = loadDataSet(testfilename)
    trainDataset = np.mat(trainDataset1)
    testDataset = np.mat(testDataset1)

    cartTree = createCARTtree(trainDataset)
    pruneTree = prune(cartTree, testDataset)
    # treePlotter.createPlot(cartTree)
    y = createForeCast(cartTree, np.mat([0.3]), modelEval=regressEvaluation)
