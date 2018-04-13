'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *


def loadDataSet():
    # 词条切分后的文档集合，列表每一行代表一个文档
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 由人工标注的每篇文档的类标签
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    #创建词汇表
    :param dataSet:
    :return:
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 创建并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    # 根据词汇表，讲句子转化为向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    http://blog.csdn.net/tanhongguang1/article/details/45016421
    p(ci|w)=p(w|ci)*p(ci)/p(w)
    #训练算法，从词向量计算概率p(w0|ci)...及p(ci)
    伪代码：
    计算每个类别文档的数目
    计算每个类别占总文档数目的比例
    对每一篇文档：
    　　对每一个类别：
    　　　　如果词条出现在文档中->增加该词条的计数值#统计每个类别中出现的词条的次数
    　　　　增加所有词条的计数值#统计每个类别的文档中出现的词条总数
    　　对每个类别：
    　　　　将各个词条出现的次数除以类别中出现的总词条数目得到条件概率
    返回每个类别各个词条的条件概率和每个类别所占的比例

    :param trainMatrix: 由每篇文档的词条向量组成的文档矩阵
    :param trainCategory: 每篇文档的类标签组成的向量
    :return:返回各类对应特征的条件概率向量和各类的先验概率
    """
    # 获取文档矩阵中文档的数目
    numTrainDocs = len(trainMatrix)
    # 获取词条向量的长度
    numWords = len(trainMatrix[0])
    # 所有文档中属于类1所占的比例p(c=1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 计算频数初始化为1  #即拉普拉斯平滑
    # 创建一个长度为词条向量等长的列表
    p0Num = ones(numWords);
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0;
    p1Denom = 2.0  # change to 2.0
    # 遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        # 如果该词条向量对应的标签为1
        if trainCategory[i] == 1:
            # 统计所有类别为1的词条向量中各个词条出现的次数
            # 统计类别为1的词条向量中出现的所有词条的总数
            # 即统计类1所有文档中出现单词的数目
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 避免乘积过小产生下溢
    # 利用NumPy数组计算p(wi|c1)
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    #朴素贝叶斯分类函数
    :param vec2Classify: 待测试分类的词条向量
    :param p0Vec: 类别0所有文档中各个词条出现的频数p(wi|c0)
    :param p1Vec: 类别1所有文档中各个词条出现的频数p(wi|c1)
    :param pClass1: 类别为1的文档占文档总数比例
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    """
    #分类测试整体函数
    :return:
    """
    #由数据集获取文档矩阵和类标签向量
    listOPosts,listClasses=loadDataSet()
    #统计所有文档中出现的词条，存入词条列表
    myVocabList=createVocabList(listOPosts)
    #创建新的列表
    trainMat=[]
    for postinDoc in listOPosts:
        #将每篇文档利用words2Vec函数转为词条向量，存入文档矩阵中
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))\
    #将文档矩阵和类标签向量转为NumPy的数组形式，方便接下来的概率计算
    #调用训练函数，得到相应概率值
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    #测试文档
    testEntry=['love','my','dalmation']
    #将测试文档转为词条向量，并转为NumPy数组的形式
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    #利用贝叶斯分类函数对测试文档进行分类并打印
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    #第二个测试文档
    testEntry1=['stupid','garbage']
    #同样转为词条向量，并转为NumPy数组的形式
    thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',classifyNB(thisDoc1,p0V,p1V,pAb))

#贝叶斯算法实例：过滤垃圾邮件

def textParse(bigString):  # input is big string, #output is word list
    """
    #处理数据长字符串
    #1 对长字符串进行分割，分隔符为除单词和数字之外的任意符号串
    #2 将分割后的字符串中所有的大些字母变成小写lower(),并且只
    #保留单词长度大于3的单词
    :param bigString:
    :return:
    """
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        # 打开并读取指定目录下的本文中的长字符串，并进行处理返回
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        # 将得到的字符串列表添加到docList
        docList.append(wordList)
        # 将字符串列表中的元素添加到fullTest
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 将所有邮件中出现的字符串构建成字符串列表
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50);
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__=='__main__':
    testingNB()