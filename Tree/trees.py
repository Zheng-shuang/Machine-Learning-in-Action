# 《机器学习实战》第三章
# 计算给定数据集的香农di
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算给定数据集的香农熵  
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}                 #字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0  #不存在，则加入到字典中
        labelCounts[currentLabel] += 1
    # print(labelCounts)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) 
    return shannonEnt

# myDat,labels=createDataSet()
# tt=calcShannonEnt(myDat)
# print(tt)

# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     # 剔除axis用来划分
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# tt=splitDataSet(myDat,1,0)
# print(tt)

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 最后一列用于标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        # 迭代所有特征
        featList = [example[i] for example in dataSet]  # 创建此特征的所有示例的列表，获取每列的值
        uniqueVals = set(featList)       # 获取set的唯一值,set() 函数创建一个无序不重复元素集
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  
        infoGain = baseEntropy - newEntropy     # 计算信息增益；即熵的减少
        # print(infoGain)
        if (infoGain > bestInfoGain):       # 将此与迄今为止的最佳收益进行比较
            bestInfoGain = infoGain         # 如果优于当前最佳，则设置为最佳
            bestFeature = i
    return bestFeature                      # 返回一个整数

# tt=chooseBestFeatureToSplit(myDat)
# print(tt)

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]               # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:              # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)     # 遍历完所有特征时返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}            # 存储树的所有信息
    del(labels[bestFeat])                  # #删除已经使用过的属性标签
    featValues = [example[bestFeat] for example in dataSet]    # 取dataSet数据集的第bestFeat列
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       # 复制所有的labels, 就不会打乱现在已有的lables
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree 

# myTree = createTree(myDat,labels)
# print(myTree)

# 测试算法：使用决策树执行分类
def classify(inputTree,featLabels,testVec):
    '''
      分类函数
      :inputTree:  决策树
      :featLabels:  特征标签
      :testVec:   测试向量
    '''
    firstStr = list(inputTree.keys())[0]       # #获取根节点,即取出inputTree的第一个关键词
    secondDict = inputTree[firstStr]           # 获取下一级分支，即取出第一个关键词的值
    featIndex = featLabels.index(firstStr)     # 查找当前列表中第一个匹配firstStr变量的元素的索引
    key = testVec[featIndex]                 # 获取测试样本中，与根节点特征对应的取值
    valueOfFeat = secondDict[key]            # 获取测试样本通过第一个特征分类器后的输出
    print(valueOfFeat)
    if isinstance(valueOfFeat, dict):        # 判断节点是否为字典来以此判断是否为叶节点
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat           # 如果到达叶子节点，则返回当前节点的分类标签
    return classLabel

# 导入画图，不导入会出错
import treePlotter

myDat,labels=createDataSet()
myTree=treePlotter.retrieveTree(0)
# print(myTree)
tt=classify(myTree,labels,[1,0])
# print(tt)

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

# storeTree(myTree,'classifierStorage.txt')
# print(grabTree('classifierStorage.txt'))

# 隐形眼镜数据集
fr=open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree=createTree(lenses,lensesLabels)
# print(lensesTree)
treePlotter.createPlot(lensesTree)