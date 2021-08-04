from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])      # 分别是X0，X1，X2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             # 100 * 3的矩阵
    labelMat = mat(classLabels).transpose() # 100 * 1的列向量
    m,n = shape(dataMatrix)        # 行数（样本数：100）、列数（特征数：3）
    alpha = 0.001           # 目标移动的步长
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              # 重矩阵运算
        h = sigmoid(dataMatrix*weights)     # 矩阵乘法：每个样本的特征值×系数，拔得出来的值作为sigmoid函数输入
        error = (labelMat - h)              # 计算每个样本的sigmoid输出与标签的差值
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights

dataArr,labelMat=loadDataSet()
# print(gradAscent(dataArr,labelMat))

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]       # 样本数：100
    xcord1 = []; ycord1 = []    # 标签为1的数据点的x坐标、y坐标
    xcord2 = []; ycord2 = []    # 标签为0的数据点的x坐标、y坐标
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # s参数是数据点的粗细，marker是标记（'s'是正方形，默认圆形）
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)                  # 直线的x坐标范围
    y = (-weights[0]-weights[1]*x)/weights[2]   # 直线的方程,由0=w0x0+w1x1+w2x2得出x1与x2的关系式（即分割线的方程，其中x0=1）
    ax.plot(x, y) 
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

# dataMat, labelMat = loadDataSet()
# weights=gradAscent(dataMat, labelMat)
# plotBestFit(weights.getA())              # getA()函数将Numpy.matrix型转为ndarray型

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
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
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():            # 处理每行数据，前21个特征赋值给 lineArr，添加到trainingSet；后1个标签添加到trainingLabels
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0   # 测试集的长度
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    a="the error rate of this test is: %f" % errorRate
    print(a)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    m="after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
    print(m)

multiTest()