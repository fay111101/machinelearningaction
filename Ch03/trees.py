'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
"""
创建分支的伪代码：

检测数据集中的每个子项是否属于同一分类：
    if so return 类标签;
    else
      寻找划分数据集的最好特征
      划分数据集
      创建分支结点
            for 每个分支结点
                 调用函数createBranch并增加返回结点到分支结点中//递归调用createBranch（）
      return 分支结点
"""
def createDataSet():
    """
    创建样本数据，最后一列代表标签
    :return:
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    # 为所有可能分类创建的字典
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        # 分类标签
        currentLabel = featVec[-1]
        # 为所有可能分类创建的字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算所有类别可能值包含的信息期望值
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #表示该分类的频率，由大数定理可知 为頻数
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value:需要返回的特征值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 按照axis轴将样本数据切分
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    """

    选择最好的数据集划分方式

        如果待分类的事物可能会出现多个结果x，则第i个结果xi发生的概率为p(xi),
        1）可以由此计算出xi的信息熵为l(xi)=p(xi)log(1/p(xi))=-p(xi)log(p(xi))
        2）对于所有可能出现的结果，事物所包含的信息期望值（信息熵）就为：H=-Σp(xi)log(p(xi))，i属于所有可能的结果
        信息增益H(D,A)=原始数据集的信息熵H(D)-特征A对数据集进行划分后信息熵H(D/A)
        其中H(D/A)=∑|Aj|/|D|*H(Aj)，j属于A的k种取值之一，|Aj|和|D|分别表示，特征A第j种取值的样本数占所有取值样本总数的比例，以及数据集的样本总数
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        # 获取第i维特征值
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # 对特征值去重
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            # 以特征i划分数据集得到的数据子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 得到该子集占总体的概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算特征取得该值时的信息熵*特征取得该值的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    """
    多数表决方法决定叶子节点的分类
    :param classList:
    :return:
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    """
    李航ID3算法构造决策树算法：


    :param dataSet: 训练数据集
    :param labels: 特征集 代表特征的标签列表 如包含不浮出水面能否可以生存、是否有脚蹼、
    :return:返回字典嵌套表示的决策树 {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    """
    # 获取类别标签
    classList = [example[-1] for example in dataSet]
    # 1）当所有类别都是同一类别时，递归停止
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 当数据集的属性全部遍历完，仍不满足递归终止条件1），即类标签依然不是唯一的
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包含的所有可能值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 对于每种可能的取值，构建该取值下的决策树
    for value in uniqueVals:
        # 复制类特征标签列表 由于Python 是按照引用传递对象的
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    """
    完成决策树的构造后，采用决策树实现具体应用
    :param inputTree: 构建好的决策树
    :param featLabels: 特征标签列表
    :param testVec: 测试实例
    :return:
    """
    # 注意python2.x和3.x区别，2.x可写成firstStr=inputTree.keys()[0]
    # 而不支持3.x
    # 找到树的第一个分类特征，或者说根节点'no surfacing'
    firstStr = list(inputTree.keys())[0]
    # 从树中得到该分类特征的分支，有0和1
    secondDict = inputTree[firstStr]
    # 根据分类特征的索引找到对应的标称型数据值
    # 'no surfacing'对应的索引为0
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        # 测试实例的第0个特征取值等于第key个子节点
        if testVec[featIndex] == key:
            # type()函数判断该子节点是否为字典类型
            if type(secondDict[key]).__name__ == 'dict':
                # 子节点为字典类型，则从该分支树开始继续遍历分类
                classLabel = classify(secondDict[key], featLabels, testVec)
            # 如果是叶子节点，则返回节点取值
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    """
    #决策树的存储：python的pickle模块序列化决策树对象，使决策树保存在磁盘中
    #在需要时读取即可，数据集很大时，可以节省构造树的时间
    #pickle模块存储决策树
    :param inputTree:
    :param filename:
    :return:
    """
    import pickle
    #创建一个可以'写'的文本文件
    #这里，如果按树中写的'w',将会报错write() argument must be str,not bytes
    #所以这里改为二进制写入'wb'
    # fw = open(filename,'wb')
    fw = open(filename,'w')
    # pickle的dump函数将决策树写入文件中
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    """
    #取决策树操作
    :param filename:
    :return:
    """
    import pickle
    # 对应于二进制方式写入数据，'rb'采用二进制形式读出数据
    # fr = open(filename, 'rb')
    fr = open(filename)
    return pickle.load(fr)



if __name__=='__main__':
    dataSet, labels=createDataSet()
    print(dataSet)
    print(labels)
    splitDataSet(dataSet,0,1)
    chooseBestFeatureToSplit(dataSet)
    myTree=createTree(dataSet,labels)
    print(myTree)
    classify(myTree,labels,[1,0])
    storeTree(myTree,'classifiserStorage.txt')
    grabTree('classifiserStorage.txt')
