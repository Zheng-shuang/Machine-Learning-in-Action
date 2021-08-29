from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))#使用frozenset类型,不可修改
                            #can use it as a key in a dict    

def scanD(D, Ck, minSupport):    # 数据集，候选项集列表，感兴趣项集的最小支持度
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):    # issubset() 方法用于判断集合的所有元素是否都包含在指定集合中，如果是则返回 True，否则返回 False。
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData               # 返回ck的频繁项集,和支持度

dataSet=loadDataSet()
C1=createC1(dataSet)
D=list(map(set,dataSet))
L1,suppData0=scanD(D,C1,0.5)
# print(L1)

# 创建候选项集CK
def aprioriGen(Lk, k):    # 频繁项集列表，项集元素个数
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: # 如果前k-2个元素相等
                retList.append(Lk[i] | Lk[j]) # 两个集合合成一个大小为k的集合
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

L,suppData=apriori(dataSet)
# print(L)
# print(aprioriGen(L[0],2))

# 生成满足最小可信度的关联规则
def generateRules(L, supportData, minConf=0.7):  # 最大频繁项集,支持度数据集,最低可信度
    bigRuleList = []
    for i in range(1, len(L)): # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         # 满足最小可信度的关联规则

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] 
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]   # 计算可信度
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): # 尝试进一步合并
        Hmp1 = aprioriGen(H, m+1)# create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):      # 至少需要两个集合才能合并
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# L,supportData=apriori(dataSet,minSupport=0.5)
# rules=generateRules(L,supportData,minConf=0.7)
# print("rules:",rules)

mushDatSet=[line.split() for line in open('mushroom.dat').readlines()]
L,supportData=apriori(mushDatSet,minSupport=0.3)
for item in L[1]:
    if item.intersection('2'):
        print(item)
