'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    """
    #标准线性回归算法
    #ws=(X.T*X).I*(X.T*Y)
    :param xArr:
    :param yArr:
    :return:
    """
    # 将列表形式的数据转为numpy矩阵形式
    xMat = mat(xArr); yMat = mat(yArr).T
    # 将列表形式的数据转为numpy矩阵形式
    xTx = xMat.T*xMat
    # numpy线性代数库linalg
    # 调用linalg.det()计算矩阵行列式
    # 计算矩阵行列式是否为0,用以判断矩阵是否可逆
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    # 如果可逆，根据公式计算回归系数
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    """
    # 局部加权线性回归lwlr
    每个测试点赋予权重系数
    回归系数的解 w*=(xTWx)-1(xTWy)
    :param testPoint: 测试点
    :param xArr: 样本数据矩阵
    :param yArr: 样本对应的原始值
    :param k: 用户定义的参数，决定权重的大小，默认1.0
    :return:
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]
        # 根据偏差利用高斯核函数赋予该样本相应的权重
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    # 将权重矩阵应用到公式中
    xTx = xMat.T * (weights * xMat)
    # 计算行列式值是否为0，即确定是否可逆
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    # 根据公式计算回归系数
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    """
    测试集进行预测
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    m = shape(testArr)[0]
    # 测试集预测结果保存在yHat列表中
    yHat = zeros(m)
    # 测试集预测结果保存在yHat列表中
    for i in range(m):
        # 计算预测值
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    """
    计算平方误差的和
    :param yArr:
    :param yHatArr:
    :return:
    """
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    """
    岭回归即是在矩阵xTx上加入一个λI从而使得矩阵非奇异，进而能对矩阵xTx+λI求逆
    w*=(xTx+λI)-1(xTy)
    :param xMat: 样本数据
    :param yMat: 样本对应的原始值
    :param lam: 惩罚项系数lamda，默认值为0.2
    :return:
    """
    # 计算矩阵内积
    xTx = xMat.T*xMat
    # 添加惩罚项，使矩阵xTx变换后可逆
    denom = xTx + eye(shape(xMat)[1])*lam
    # 判断行列式值是否为0，确定是否可逆
    if linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    # 计算回归系数
    ws = (denom.I) * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    """
    特征需要标准化处理，使所有特征具有相同重要性
    :param xArr:
    :param yArr:
    :return:得到30个不同的λ所对应的回归系数
    """
    xMat = mat(xArr); yMat=mat(yArr).T
    # 计算均值
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    # 计算各个特征的方差
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    # 特征-均值/方差
    xMat = (xMat - xMeans)/xVar
    # 在30个不同的lamda下进行测试
    numTestPts = 30
    # 30次的结果保存在wMat中
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        # 计算对应lamda回归系数，lamda以指数形式变换
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    前向逐步回归算法伪代码
    数据标准化，使其分布满足均值为0,和方差为1
    在每轮的迭代中：
        设置当前最小的误差为正无穷
        对每个特征：
            增大或减小：
                改变一个系数得到一个新的w
                计算新w下的误差
                如果误差小于当前最小的误差：设置最小误差等于当前误差
                将当前的w设置为最优的w
        将本次迭代得到的预测误差最小的w存入矩阵中
    返回多次迭代下的回归系数组成的矩阵
    :param xArr:
    :param yArr:
    :param eps: 每次迭代需要调整的步长
    :param numIt:
    :return:
    """
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    # 将特征标准化处理为均值为0，方差为1
    xMat = regularize(xMat)
    m,n=shape(xMat)
    # 将每次迭代中得到的回归系数存入矩阵
    #returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print (ws.T)
        # 初始化最小误差为正无穷
        lowestError = inf; 
        for j in range(n):
            # 对每个特征的系数执行增加和减少eps*sign操作
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                # 变化后计算相应预测值
                yTest = xMat*wsTest
                # 保存最小的误差以及对应的回归系数
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        #returnMat[i,:]=ws.T
    #return returnMat

#def scrapePage(inFile,outFile,yr,numPce,origPrc):
#    from BeautifulSoup import BeautifulSoup
#    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
#    soup = BeautifulSoup(fr.read())
#    i=1
#    currentRow = soup.findAll('table', r="%d" % i)
#    while(len(currentRow)!=0):
#        title = currentRow[0].findAll('a')[1].text
#        lwrTitle = title.lower()
#        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#            newFlag = 1.0
#        else:
#            newFlag = 0.0
#        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#        if len(soldUnicde)==0:
#            print "item #%d did not sell" % i
#        else:
#            soldPrice = currentRow[0].findAll('td')[4]
#            priceStr = soldPrice.text
#            priceStr = priceStr.replace('$','') #strips out $
#            priceStr = priceStr.replace(',','') #strips out ,
#            if len(soldPrice)>1:
#                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
#            print "%s\t%d\t%s" % (priceStr,newFlag,title)
#            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
#        i += 1
#        currentRow = soup.findAll('table', r="%d" % i)
#    fw.close()

from time import sleep
import json
import urllib


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, setNum)
    pg = urllib.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    """
    交叉验证
    :param xArr:
    :param yArr:
    :param numVal:
    :return:
    """

    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = [];
        trainY = []
        testX = [];
        testY = []
        random.shuffle(indexList)
        # 将数据分为训练集和测试集
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX);
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr);
    yMat = mat(yArr).T
    meanX = mean(xMat, 0);
    varX = var(xMat, 0)
    # 数据还原
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))


if __name__=='__main__':
    abx,aby=loadDataSet('ex0.txt')
    #--------------------标准线性回归----------------------------

    ws=standRegres(abx,aby)
    xMat=mat(abx)
    yMat=mat(aby)
    yHat=xMat*ws
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # 散点图
    # a = np.array([[1,2], [3,4]])
    # a.flatten()
    # array([1, 2, 3, 4])
    # a.flatten('F') #按竖的方向降
    # array([1, 3, 2, 4])a是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组
    # from numpy import *
    #  a = mat(a)
    #  y = a.flatten()
    #  y
    # matrix([[1, 3, 2, 4, 3, 5]])
    #  y = a.flatten().A
    #  y
    # array([[1, 3, 2, 4, 3, 5]])
    #  shape(y)
    # (1, 6)
    # y = a.flatten().A[0]
    # shape(y)
    # (6,)
    # y
    # array([1, 3, 2, 4, 3, 5])
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    # 绘制直线之前要对直线上的点进行排序否则将错乱
    yHat=xCopy*ws
    # yHat=xMat*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()
    yHat=xMat*ws
    # 利用numpy提供的相关系数计算公式来计算预测值和真实值之间的相关性
    print(corrcoef(yHat.T,yMat))
    # --------------------局部加权线性回归----------------------------
    print(aby[0])
    print(lwlr(abx[0],abx,aby,1.0))
    # 得到数据集中所有的估计
    yHat=lwlrTest(abx,abx,aby,1.0)


    # --------------------Ridge回归----------------------------
    abX,abY=loadDataSet('abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)
    fig2=plt.figure()
    ax=fig2.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

    # --------------------前向逐步回归----------------------------
    stageWise(abX,abY,0.01,200)
