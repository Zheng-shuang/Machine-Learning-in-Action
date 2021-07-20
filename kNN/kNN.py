# 2.1 K-近邻算法（《机器学习实战》第二章）

# 输入：inX：与现有数据集比较的向量（1xN）
# 数据集：大小为m的已知向量数据集（NxM）
# 标签：数据集标签（1xM向量）
# k:用于比较的邻居数（应为奇数）
# 输出：最流行的类标签
from numpy import * #科学计算包

import operator #运算符模块
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.pyplot是一些命令行风格函数的集合

from os import listdir #列出给定目录的文件名

# 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])               # 4行2列，即group.shape为（4，2）
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group,labels=createDataSet()
# print(group)
# print(labels)

#  实施kNN分类算法
def classify0(inX, dataSet, labels, k):
    # 用于分类的输入向量是inX，输入的训练样本集是dataSet，标签向量是labels，k表示用于选择最近邻居的数目。 
    dataSetSize = dataSet.shape[0]                         # 查看矩阵或者数组的维数.c.shape[0] 为第一维的长度，c.shape[1]为第二维的长度,此处为4
    # (dataSetSize, 1)使数组重复完是四行一样的  而不是在1行中。#numpy.tile(A,reps) tile共有2个参数，A指待输入数组，reps则决定A重复的次数。整个函数用于重复数组A来构建新的数组。
    diffMat = tile(inX, (dataSetSize,1))-dataSet                 # tile函数作用：复制给定内容，并生成指定行列的矩阵.这里是将inX重复1次形成dataSetSize行1列的数组
    sqDiffMat = diffMat**2                           # 幂  （x1 - x2）的幂
    sqDistances = sqDiffMat.sum(axis=1)              # 每行相加,横着相加
    distances = sqDistances**0.5                     # 开根号
    sortedDistIndicies = distances.argsort()         # argsort是排序，将元素按照由小到大的顺序返回下标
    classCount={}                                    # dict字典数据类型，字典是Python中唯一内建的映射类型

    for i in range(k):       
        voteIlabel = labels[sortedDistIndicies[i]]
        # get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），
        # 这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)的意思是按照字典里的第1个排序，{A:1,B:2},要按照第1个（A和B是第0个），即‘1’‘2’排序。reverse=True是降序排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

# kNN=classify0([0,0],group,labels,3)
# print(kNN)

#  2.2 准备数据：从文本文件中解析数据
def file2matrix(filename):
    fr = open(filename)
    # 一次读取整个文本数据，并且自动将文件内容分析成一个行的列表，比readline（）快 ，后面的img2vector就是使用的readline（），因为要逐行逐个读取，可以对比一下
    tt=fr.readlines()
    numberOfLines = len(tt) 
    # 返回来一个给定形状和类型的用0填充的数组;文件有几行就是几行，设置为3列（可调）      
    returnMat = zeros((numberOfLines,3))        
    classLabelVector = []                         
    index = 0
    for line in tt:
        line = line.strip()                                  # 去掉回车符
        listFromLine = line.split('\t')                      # 分成了4列数据，得到了4个列表
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]                      
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

datingDataMat,datingLabels=file2matrix("datingTestSet2.txt")
# print(datingDataMat)
# print(datingLabels[0:20])

# 准备数据：归一化数值
def autoNorm(dataSet):
    # min(0)返回该矩阵中每一列的最小值
    # min(1)返回该矩阵中每一行的最小值
    # max(0)返回该矩阵中每一列的最大值
    # max(1)返回该矩阵中每一行的最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))            # (1000,3)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   # element wise divide
    return normDataSet, ranges, minVals

normMat,ranges,minVals=autoNorm(datingDataMat)
# print(normMat)
# print(ranges)
# print(minVals)

# 测试算法：作为完整程序验证分类器
def datingClassTest():
    hoRatio = 0.50      # hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       # 从文件加载数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]                                                 # 1000
    numTestVecs = int(m*hoRatio)                                         # 训练样本从第m * hoRatio 开始
    errorCount = 0.0
    for i in range(numTestVecs):
        # print(a[:,1])表示取第二维度的第二个元素（即第二列）,print(a[:,0])取所有第一维，第二维的第一个元素，即第一列,print(a[:,0:2])取第一维的所有，第二维的[0,2)区间
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # print("the classifier came back with:")
        # print(classifierResult)
        # print("the real answer is:")
        # print(datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is:")
    error=(errorCount/float(numTestVecs))
    return errorCount
    
# k=datingClassTest()
# print(k)

def classifyPerson() :
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLables = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    print ("You will probably like this person:", resultList[classifierResult - 1]) #索引从0开始，索引减去1才能索引到对应的resultList

# classifyPerson()

# 2.3 手写识别系统
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# testVector=img2vector('testDigits/0_13.txt')
# print(testVector[0,0:31])

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           # #获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     # 无后缀文件名
        classNumStr = int(fileStr.split('_')[0])   # 获取文件内的数字
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)   # 图片转换为向量
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        strr = "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        print(strr)
        if (classifierResult != classNumStr): errorCount += 1.0
    str1="\nthe total number of errors is: %d" % errorCount
    print(str1)
    str2="\nthe total error rate is: %f" % (errorCount/float(mTest))
    print(str2)

handwritingClassTest()