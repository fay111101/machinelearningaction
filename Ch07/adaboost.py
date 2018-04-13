'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    """
    创建一个简单数据集
    :return:
    """
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    """
    自适应加载数据
    :param fileName:
    :return:
    """
    # 获取特征数目(包括最后一类标签)
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    # 创建数据集矩阵，标签向量
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    """
    单层决策树的阈值过滤函数

    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq:阈值的模式，1,当threshIneq==lt时，将<=归类为-1,否则将>归类为-1
    :return:
    """
    # 对数据集每一列的各个特征进行阈值过滤
    retArray = ones((shape(dataMatrix)[0],1))
    # 阈值的模式，将小于某一阈值的特征归类为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    # 将大于某一阈值的特征归类为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    """
    #构建单层分类器
    #单层分类器是基于最小加权分类错误率的树桩

    #将最小错误率minError设为+∞
    #对数据集中的每个特征(第一层特征)：
        #对每个步长(第二层特征)：
            #对每个不等号(第三层特征)：
                #建立一颗单层决策树并利用加权数据集对它进行测试
                #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    #返回最佳单层决策树
    :param dataArr:
    :param classLabels:
    :param D:
    :return:返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # 步长或区间总数 最优决策树信息 最优单层决策树预测结果
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    # 最小错误率初始化为+∞
    minError = inf #init error sum, to +infinity
    # 遍历每一列的特征值
    for i in range(n):#loop over all dimensions
        # 找出列中特征值的最小值和最大值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        # 求取步长大小或者说区间间隔
        stepSize = (rangeMax-rangeMin)/numSteps
        # 遍历各个步长区间
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            # 两种阈值过滤模式
            for inequal in ['lt', 'gt']: #go over less than and greater than
                # 阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                threshVal = (rangeMin + float(j) * stepSize)
                # 选定阈值后，调用阈值过滤函数分类预测
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                # 初始化错误向量
                errArr = mat(ones((m,1)))
                # 将错误向量中分类正确项置0
                errArr[predictedVals == labelMat] = 0
                # 计算"加权"的错误率
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 如果当前错误率小于当前最小错误率，将当前错误率作为最小错误率
                # 存储相关信息
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
    #完整AdaBoost算法实现
    #算法实现伪代码
    #对每次迭代：
        #利用buildStump()函数找到最佳的单层决策树
        #将最佳单层决策树加入到单层决策树数组
        #计算alpha
        #计算新的权重向量D
        #更新累计类别估计值
        #如果错误率为等于0.0，退出循环
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    """
    # 弱分类器相关信息列表
    weakClassArr = []
    # 获取数据集行数
    m = shape(dataArr)[0]
    # 初始化权重向量的每一项值相等
    D = mat(ones((m,1))/m)   #init D to all equal
    # 累计估计值向量
    aggClassEst = mat(zeros((m,1)))
    # 循环迭代次数
    for i in range(numIt):
        # 根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        # 打印权重向量
        #print ("D:",D.T)
        # 求单层决策树的系数alpha
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        # 存储决策树的系数alpha到字典
        bestStump['alpha'] = alpha
        # 将该决策树存入列表
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        # 打印决策树的预测结果
        #print ("classEst: ",classEst.T)
        # 预测正确为exp(-alpha),预测错误为exp(alpha)
        # 即增大分类错误样本的权重，减少分类正确的数据点权重
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        # 更新权值向量
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        # 累加当前单层决策树的加权预测值
        aggClassEst += alpha*classEst
        #print ("aggClassEst: ",aggClassEst.T)
        # 求出分类错的样本个数
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        # 计算错误率
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        # 错误率为0.0退出循环
        if errorRate == 0.0: break
    # 返回弱分类器的组合列表
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    """

    :param datToClass: 测试数据点
    :param classifierArr: 构建好的最终分类器
    :return:
    """
    # 构建数据向量或矩阵
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    # 获取矩阵行数
    m = shape(dataMatrix)[0]
    # 初始化最终分类器
    aggClassEst = mat(zeros((m,1)))
    # 遍历分类器列表中的每一个弱分类器
    for i in range(len(classifierArr)):
        # 每一个弱分类器对测试数据进行预测分类
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        # 对各个分类器的预测结果进行加权累加
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    # 通过sign函数根据结果大于或小于0预测出+1或-1
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)


if __name__=='__main__':
    trainfilename='horseColicTraining2.txt'
    # dataMat,classLabels=loadDataSet(trainfilename)
    dataMat,classLabels=loadSimpData()
    classifier_array=adaBoostTrainDS(dataMat,classLabels,30)
    print(classifier_array)
    print(type(classifier_array))
    adaClassify([0,0],classifier_array[0])