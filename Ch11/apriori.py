'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
@author: fay
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    """
    构建第一个候选项集的列表C1
    :param dataSet:
    :return: 大小为1的所有候选项的集合
    """

    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 让每个物品项作为集合
                C1.append([item])
                
    C1.sort()
    # 使用frozenset可以将集合作为字典的键使用
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict    

def scanD(D, Ck, minSupport):
    """
    1.生成候选项集的伪代码（用于从C1生成L1）

     对数据集中的每条交易记录tran：
        对每个候选项集can：
            检查一下can是否是tran的子集：
            如果是，则增加can的计数值
            对每个候选项集：
            如果其支持度不低于最小值，则保留该项集
            返回所有频繁项集列表
    2. 相关概念

    支持度：一个项集的支持度为数据集中包含该项集的记录所占的比例
    可信度：针对一条关联规则来定义的， e.g.一条规则P->H的可信度定义为support(P|H)/support(P) |表示集合的或操作

    :param dataSet:数据集
    :param Ck:候选项集列表
    :param minSupport:感兴趣项集的最小支持度
    :return:返回一个包含支持度的字典
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                # 如果不包含can候选集则以can为键初始化为列表中的值
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    # 交易总数
    numItems = float(len(D))
    # 满足最小支持度的集合
    retList = []
    # 用于存储最频繁项集的支持度数据
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k): #creates Ck
    """
    用于生成候选项集Ck
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return: 返回创建的候选项集Ck
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 当前k-2项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    """
    Apriori算法的伪代码
    当集合中项的个数大于0时
        构建一个k个项组成的候选项集的列表
        检查数据以确认每个项集都是频繁的
        保留频繁项集并构建k+1项组成的候选项集的列表
    :param dataSet:数据集
    :param minSupport:最小支持度
    :return:候选项集的列表
    """
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        # 扫描数据集，从Ck得到Lk
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    """
    性质：
    如果某条规则并不满足最小可信度的要求，那么该规则的所有子集也不会满足最小可信度要求
    e.g.
    假设规则 0,1,2->3并不满足最小可信度的要求，那么可知任何左部为{0,1,2}子集的规则也不会满足最小可信度要求
    :param L: 频繁项集列表
    :param supportData: 包含那些频繁项集支持数据的字典
    :param minConf: 最小可信读阈值
    :return: 返回一个包含可信度的规则列表
    """
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    该函数用于计算规则的可信度以及找到满足最小可信度要求的规则
    :param freqSet:
    :param H:
    :param supportData:
    :param brl:
    :param minConf:
    :return: 返回一个满足最小可信度要求的规则列表
    """
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    用于生成候选集规则集合
    :param freqSet: 频繁项集
    :param H: 可以出现在规则右部的元素列表
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    """
    # H中频繁项集的大小m
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print (itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("confidence: %f" % ruleTup[2])
        print  ()     #print a blank line
        
            
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning
