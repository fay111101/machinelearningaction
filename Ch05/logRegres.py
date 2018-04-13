'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
from numpy import mat

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    梯度上升的伪代码：
    每个回归系数初始化为1
    重复R次
        计算整个数据集的梯度
        使用alpha×gradient更新回归系数的向量
    返回回归系数
    :param dataMatIn: 2维Numpy数组，行代表每个训练样本，列代表每个不同的特征
    :param classLabels:
    :return:
    """
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    # 目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def plotBestFit(weights):
    """
    画出决策边界
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    # 设置步长为0.01
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    # 循环m次，每次选取数据集一个样本更新参数
    for i in range(m):
        # 计算当前样本的sigmoid函数值
        h = sigmoid(sum(dataMatrix[i]*weights))
        # 计算当前样本的残差(代替梯度)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    # 原因是python3中range不返回数组对象，而是返回range对象
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    """
    分类决策函数
    :param inX:
    :param weights:
    :return:
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    """
    #logistic回归预测算法
    :return:
    """
    # 打开训练数据集
    frTrain = open('horseColicTraining.txt')
    # 打开测试数据集
    frTest = open('horseColicTest.txt')
    # 新建两个空列表，用于保存训练数据集和标签
    trainingSet = []; trainingLabels = []
    # 读取训练集文档的每一行
    for line in frTrain.readlines():
        # 对当前行进行特征分割
        currLine = line.strip().split('\t')
        # 新建列表存储每个样本的特征向量
        lineArr =[]
        for i in range(21):
            # 将该样本的特征存入lineArr列表
            lineArr.append(float(currLine[i]))
        # 将该样本的特征向量添加到数据集列表
        trainingSet.append(lineArr)
        # 将该样本标签存入标签列表
        trainingLabels.append(float(currLine[21]))

    # 调用随机梯度上升法更新logistic回归的权值参数
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    # 统计测试数据集预测错误样本数量和样本总数
    errorCount = 0; numTestVec = 0.0
    # 遍历测试数据集的每个样本
    for line in frTest.readlines():
        # 样本总数加1
        numTestVec += 1.0
        # 对当前行进行处理，分割出各个特征及样本标签
        currLine = line.strip().split('\t')
        # 新建特征向量
        lineArr =[]
        # 将各个特征构成特征向量
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 利用分类预测函数对该样本进行预测，并与样本标签进行比较
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__=='__main__':
    dataArr,labelMat=loadDataSet()
    weights=gradAscent(dataArr,labelMat)
    print(gradAscent(dataArr,labelMat))
    # 得到一维矩阵
    plotBestFit(weights.getA())

    weights1=stocGradAscent0(array(dataArr),labelMat)
    plotBestFit(weights1)

    weights2=stocGradAscent1(array(dataArr),labelMat)
    plotBestFit(weights2)
