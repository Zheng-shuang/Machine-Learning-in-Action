'''
集成分类器方法有：bagging(boosting aggregating，自举汇聚法)、随机森林(random forest)、boosting等
boosting也可细分为很多种，其中比较流行的一种是AdaBoost(adaptive boosting, 自适应boosting)
AdaBoost一般流程为：
1、收集数据
2、准备数据
3、分析数据
4、训练算法：AdaBoost的大部分时间用在训练上，分类器将多次在同一数据集上训练弱分类器
5、测试算法：计算分类的错误率
6、使用算法
以下是利用多个单层决策树和adaboost算法，在小数据上的运用实例
'''

from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# 通过阈值比较对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    [summary]:单层决策树分类函数,根据某一特征进行分类
    
    Arguments:
        dataMatrix  -- 数据矩阵
        dimen -- 选取第几列,对特征进行抽取
        threshVal  -- 阀值
        threshIneq  -- 比较关系(lt)
    
    Returns:
        retArray [numpy.ndarray]-- 分类结果
    """
    # 初始化retArray为1,m行1列全为1
    retArray = ones((shape(dataMatrix)[0],1))
    # 置于-1,进行分类
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0   # 该列的值<=threshVal的，全部置为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
# 构建单层决策树（决策树的简化版本），是一种弱分类器算法
def buildStump(dataArr,classLabels,D):
    """
    [summary]:找到数据集上最佳的单层决策树
    将最小错误率minError设为+∞   
    对数据集中的每一个特征（第一层循环）:
            对每个步长（第二层循环）:
            对每个不等号（第三层循环）:
                    建立一棵单层决策树并利用加权数据集对它进行测试
                    如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树
    
    Arguments:
        dataArr  -- 数据矩阵
        classLabels  -- 数据标签
        D -- 样本权重
    
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T       #.T就是对一个矩阵的转置
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf  # 最小误差初始化为正无穷大
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()         # 找到特征中最小的值和最大值
        stepSize = (rangeMax-rangeMin)/numSteps                              # 步长，按步长选择列的最佳分割值
        for j in range(-1,int(numSteps)+1):# 第二层循环：按一定步长，遍历当前特征的特征值
            for inequal in ['lt', 'gt']: # 大于和小于的情况， 第三层循环：在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)   #  根据阈值对数据进行分类，得到预测分类值
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)   # 结果为m行1列的二维，值为-1或者1
                # 初始化误差矩阵
                errArr = mat(ones((m,1)))
                # 分类正确的,赋值为0,其他依然为1
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  # 计算总误差乘以D，结果为一个值
                # nn="split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                # print(nn)
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i                 # 最佳分割维度
                    bestStump['thresh'] = threshVal      # 最佳分割值
                    bestStump['ineq'] = inequal          # 最佳分割方法：le/ge
    return bestStump,minError,bestClasEst

datMat,classLabels=loadSimpData()
D=mat(ones((5,1))/5)
bestStump,minError,bestClasEst=buildStump(datMat,classLabels,D)
# print("***********************")
# print(bestStump)
# print(minError)
# print(bestClasEst)

#INPUT:dataArr:训练集 classLabels:训练集的标签  numIt:弱分类器最多的个数
#OUPUT:weakClassArr:弱分类器的线性组合
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
    [summary]:
    对每次迭代：
        利用buildStump()函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树数组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率等于0.0，则退出循环
    
    Arguments:
        dataArr {[type]} -- 数据
        classLabels {[type]} -- 标签
    
    Keyword Arguments:
        numIt {int} -- 迭代次数 (default: {40})
    
    Returns:
        weakClassArr
        aggClassEst
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   # 初始权重1/m，概率分布向量,元素之和为1。D在迭代中增加错分数据的权重
    aggClassEst = mat(zeros((m,1)))       # 记录每个数据点的类别估计累计值
    for i in range(numIt):
        # 构建单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        # print("D:",D.T)
        # 根据公式计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))   # 1/2*In((1-error)/error),分类器的权重。
        bestStump['alpha'] = alpha           # 存储弱学习算法权重
        weakClassArr.append(bestStump)       # 弱分类器的列表，存储单层决策树           
        # print("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)  # 根据数学公式更改权重
        D = multiply(D,exp(expon))                              # 为下一次迭代计算新的D
        D = D/D.sum()    # 下一个分类的各样本的权重D(i+1)
        # 所有分类器的计算训练错误，如果为0，则提前退出循环（使用中断）
        aggClassEst += alpha*classEst
        # print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))   # sign()函数：如果数字为正数，则返回 1；如果数字为 0，则返回零 (0)；如果数字为负数，则返回 -1
        errorRate = aggErrors.sum()/m
        # print("total error: ",errorRate)
        if errorRate == 0.0: break    # 两种情况停止:(1)40个弱分类器的组合 (2)分类误差为0
    # return weakClassArr
    return weakClassArr,aggClassEst    # plotROC()函数

# print(adaBoostTrainDS(datMat,classLabels,9))


'''
参数：多个待分类样例 dataToClass，多个弱分类器 classifierArr
'''
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq']) 
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)

datArr,labelArr=loadSimpData()
classifierArr=adaBoostTrainDS(datArr,labelArr,30)
# print(adaClassify([0,0],classifierArr))

def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split('\t'))
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

# datArr,labelArr=loadDataSet('horseColicTraining2.txt')
# classifierArr=adaBoostTrainDS(datArr,labelArr,10)

# testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
# prediction10=adaClassify(testArr,classifierArr)
# errArr=mat(ones((67,1)))    # 有67行数据
# sum=errArr[prediction10!=mat(testLabelArr).T].sum()
# print(sum)

'''
参数：分类器的预测强度(即每个特征对应的类别累计估计值)，数据标签
'''
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) # 绘制光标的位置
    ySum = 0.0 # 用于计算AUC的值
    numPosClas = sum(array(classLabels)==1.0)  # 正例的数目,这里是178
    yStep = 1/float(numPosClas);      # 纵坐标表示实际正例中被正确识别的概率
    xStep = 1/float(len(classLabels)-numPosClas)  # 横坐标表示实际反例中被错误识别的概率
    sortedIndicies = predStrengths.argsort() # 获取排序索引：由小到大
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    print(sortedIndicies.tolist()[0])
    # 循环遍历所有值，在每个点绘制线段
    for index in sortedIndicies.tolist()[0]:    # tolist()作用：将矩阵（matrix）和数组（array）转化为列表。
        if classLabels[index] == 1.0: 
            delX = 0; delY = yStep    # delX是横坐标变化值，delY是纵坐标变化值
        else:
            delX = xStep; delY = 0
            ySum += cur[1]            # 把每一小段的y值相加，最后乘以xStep就是面积AUC
        # 从cur到（cur[0]-delX，cur[1]-delY）绘制线
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')  # 从右上到左下画线
        cur = (cur[0]-delX,cur[1]-delY)       # 画完线之后，当前点作为光标起点
    ax.plot([0,1],[0,1],'b--')           # 画对角线
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])         # 设置坐标轴范围
    plt.savefig('ROC.png')
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)

datArr,labelArr=loadDataSet('horseColicTraining2.txt')
classifierArr,aggClassEst=adaBoostTrainDS(datArr,labelArr,10)
plotROC(aggClassEst.T,labelArr)