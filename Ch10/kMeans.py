from numpy import *

"""
K-means算法的流程

首先，随机确定k个初始点的质心；
然后将数据集中的每一个点分配到一个簇中，
即为每一个点找到距其最近的质心，
并将其分配给该质心所对应的簇；该步完成后，每一个簇的质心更新为该簇所有点的平均值

K-means算法伪代码

创建k个点作为起始质心，可以随机选择(位于数据边界内)
　　当任意一个点的簇分配结果发生改变时
　　　　对数据集中每一个点
　　　　　　　　对每个质心
　　　　　　　　　　计算质心与数据点之间的距离
　　　　　　　　将数据点分配到距其最近的簇
　　　　对每一个簇，计算簇中所有点的均值并将均值作为质心

误差度量
使用后处理来提高聚类性能
度量聚类效果的指标SSE(Sum of Squared Error 误差平方和）
"""

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # Python3此处变成map对象
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    """
    数据向量计算欧式距离
    :param vecA:
    :param vecB:
    :return:
    """
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    """
    随机初始化K个质心(质心满足数据边界之内)
    :param dataSet:
    :param k:
    :return:
    """
    # 得到数据样本的维度
    n = shape(dataSet)[1]
    # 初始化为一个(k,n)的矩阵
    centroids = mat(zeros((k,n)))#create centroid mat
    # 遍历数据集的每一维度
    for j in range(n):#create random cluster centers, within bounds of each dimension
        # 得到该列数据的最小值
        minJ = min(dataSet[:,j])
        maxJ=max(dataSet[:,j])
        # print(minJ)
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(maxJ- minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    # 返回初始化得到的k个质心向量
    return centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    k-均值聚类算法

    :param dataSet: 聚类数据集
    :param k: 用户指定的k个类
    :param distMeas: 距离计算方法，默认欧氏距离distEclud()
    :param createCent: 获得k个质心的方法，默认随机获取randCent()
    :return:
    """
    # 获取数据集样本数
    m = shape(dataSet)[0]
    # 初始化一个(m,2)的矩阵
    # 簇分配结果矩阵 一列记录簇索引值，一列存储误差（当前点到质心的距离）
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为false
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):#for each data point assign it to the closest centroid
            # 初始化最小距离最正无穷；最小距离对应索引为-1
            minDist = inf; minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)
                    minDist = distJI; minIndex = j
             # 当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            # 更新当前变化样本的聚类结果和平方误差 SSE误差平方和是代表聚类效果的指标
            clusterAssment[i,:] = minIndex,minDist**2
        # 打印k-均值聚类的质心
        # print (centroids)
        # 遍历每一个质心
        for cent in range(k):#recalculate centroids
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment
"""
二分K-means算法
首先将所有点作为一个簇，然后将簇一分为二。
之后选择其中一个簇继续进行划分，选择哪一个簇取决于对其进行划分是否能够最大程度的降低SSE的值。
上述划分过程不断重复，直至划分的簇的数目达到用户指定的值为止。

二分K-means算法的伪代码
将所有点看成一个簇
当簇数目小于k时
    对于每一个簇
        计算总误差
        在给定的簇上面进行k-均值聚类（k=2）
        计算将该簇一分为二之后的总误差
    选择使得总误差最小的簇进行划分
"""

def biKmeans(dataSet, k, distMeans=distEclud):
    """
    二分K-均值聚类算法
    :param dataSet: 待聚类数据集
    :param k: 用户指定的聚类个数
    :param distMeas: 用户指定的距离计算方法，默认为欧式距离计算
    :return:
    """
    # 获得数据集的样本数
    m = shape(dataSet)[0]
    # 初始化一个元素均值0的(m,2)矩阵
    clusterAssment = mat(zeros((m,2)))
    # 获取数据集每一列数据的均值，组成一个长为列数的列表
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    # 当前聚类列表将数据集聚为一类
    centList =[centroid0] #create a list with one centroid
    # 遍历每个数据集样本
    for j in range(m):#calc initial Error
        # 计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1] = distMeans(mat(centroid0), dataSet[j,:])**2
    # 循环，直至二分k-均值达到k类为止
    while (len(centList) < k):
        # 将当前最小平方误差置为正无穷
        lowestSSE = inf
        # 遍历当前每个聚类
        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            print('cluster',clusterAssment[:,0].A)
            print('nonzero',nonzero(clusterAssment[:,0].A==i))
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            # 对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            # 计算该类划分后两个类的误差平方和
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            # 打印这两项误差值
            # print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                # 第i类作为本次划分类
                bestCentToSplit = i
                # 第i类划分后得到的两个质心向量
                bestNewCents = centroidMat
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss = splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowestSSE = sseSplit + sseNotSplit
        # 数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
        # 当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        # 同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        # 连续不出现空缺
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        # 打印本次执行2-均值聚类算法的类
        # print ('the bestCentToSplit is: ',bestCentToSplit)
        # 打印被划分的类的数据个数
        # print ('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表中的变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        # 添加新的类的质心向量
        centList.append(bestNewCents[1,:].tolist()[0])
        # 更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    # 返回聚类结果
    return mat(centList), clusterAssment

from urllib import request
from urllib import parse
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = parse.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print (yahooApi)
    c=request.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print ("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__=='__main__':
    # dataMat=mat(loadDataSet('testSet.txt'))
    # myCentroids,clustAssing=kMeans(dataMat,4)

    dataMat3=mat(loadDataSet('testSet2.txt'))
    centList,myNewAssments=biKmeans(dataMat3,3)