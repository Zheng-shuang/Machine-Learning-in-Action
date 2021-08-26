from numpy import *

def loadDataSet(fileName):      
    dataMat = []               
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) 
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    binSplitDataSet(将数据集，按照feature列的value进行 二元切分)
        Description：在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
    Args:
        dataMat 数据集
        feature 待切分的特征列
        value 特征列要比较的值
    Returns:
        mat0 小于等于 value 的数据集在左边
        mat1 大于 value 的数据集在右边
    Raises:
    """
    # dataSet[:, feature] 取去每一行中，第1列的值(从0开始算)
    # nonzero(dataSet[:, feature] > value)  返回结果为true行的index下标
    dataSet=mat(dataSet)
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

testMat=mat(eye(4))
mat0,mat1=binSplitDataSet(testMat,1,0.5)
# print(mat0)
# print(mat1)

def regLeaf(dataSet):
    dataSet=mat(dataSet)
    return mean(dataSet[:,-1])

def regErr(dataSet):   # 误差估计函数
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """chooseBestSplit(用最佳方式切分数据集和生成相应的叶节点)
    Args:
        dataSet   加载的原始数据集
        leafType  建立叶子点的函数
        errType   误差计算函数(求总方差)
        ops       [容许误差下降值，切分的最少样本数]。
    Returns:
        bestIndex feature的index坐标
        bestValue 切分的最优值
    Raises:
    """
    # ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
    # 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
    # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分

    tolS = ops[0]; tolN = ops[1]

    # 如果结果集(最后一列为1个变量)，就返回退出
    # .T 对数据集进行转置
    # .tolist()[0] 转化为数组并取第0列
    # 如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:  
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    # 默认最后一个特征为最佳切分特征,计算其误差估计  
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0   # 分别为最佳误差,最佳特征切分的索引值,最佳特征值
    for featIndex in range(n-1):    # 遍历所有特征列,n-1是为了不计算最后一列
        # [0]表示这一列的[所有行]，不要[0]就是一个array[[所有行]]
        for splitVal in set((dataSet[:,featIndex].T.tolist())[0]):    # 遍历所有特征值
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN,则退出
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            # 计算误差估计
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小,则更新特征索引值和特征值 
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS: 
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
        函数功能:树构建函数 参数说明:
        dataSet：原始数据集 leafType：建立叶结点的函数  errType：误差计算函数
        ops：包含树构建所有其他参数的元组 返回:
        ifretTree：构建的回归树
    """
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果没有特征,则返回特征值
    if feat == None: return val
    # 回归树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

myDat=loadDataSet('ex00.txt')
myMat=mat(myDat)
# print(createTree(myMat))
tree=createTree(myMat)

def isTree(obj):
    """
    Desc:
        测试输入变量是否是一棵树,即是否是一个字典
    Args:
        obj -- 输入变量
    Returns:
        返回布尔类型的结果。如果 obj 是一个字典，返回true，否则返回 false
    """
    return (type(obj).__name__=='dict')

# 计算左右枝丫的均值
def getMean(tree):
    """
    Desc:
        从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
        对 tree 进行塌陷处理，即返回树平均值。
    Args:
        tree -- 输入的树
    Returns:
        返回 tree 节点的平均值
    """
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

# 检查是否适合合并分枝
def prune(tree, testData):
    """
    Desc:
        从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
    Args:
        tree -- 待剪枝的树
        testData -- 剪枝所需要的测试数据 testData 
    Returns:
        tree -- 剪枝完成的树
    """
    if shape(testData)[0] == 0: return getMean(tree) # 如果没有测试数据，则对树进行塌陷处理(即返回树的平均值)
    if (isTree(tree['right']) or isTree(tree['left'])):  # 判断分枝是否是dict字典，如果是就将测试数据集进行切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)      # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)   # 如果是右边分枝是字典，就传入右边的数据集和右边的分枝，进行递归
    # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    # 1. 如果正确 
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        # power(x, y)表示x的y次方
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
         # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree

myDat2=loadDataSet('ex2.txt')
myMat2=mat(myDat2)
myTree=createTree(myMat2,ops=(0,1))
myDatTest=loadDataSet('ex2test.txt')
# print(prune(myTree,myDatTest))

# 得到模型的ws系数：f(x) = x0 + x1*featrue1+ x3*featrue2 ...
# create linear model and return coeficients
def modelLeaf(dataSet):
    """
    Desc:
        当数据不再需要切分的时候，生成叶节点的模型。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回得到的 回归系数ws
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


# 计算线性模型的误差值
def modelErr(dataSet):
    """
    Desc:
        在给定数据集上计算误差。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    # print corrcoef(yHat, Y, rowvar=0)
    return sum(power(Y - yHat, 2))

def linearSolve(dataSet):
    """
    Desc:
        将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
    Args:
        dataSet -- 输入数据
    Returns:
        ws -- 执行线性回归的回归系数 
        X -- 格式化自变量X
        Y -- 格式化目标变量Y
    """
    m, n = shape(dataSet)
    # 产生一个关于1的矩阵
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    # X的0列为1，常数项，用于计算平衡误差
    X[:, 1: n] = dataSet[:, 0: n-1]
    Y = dataSet[:, -1]

    # 转置矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的逆不存在，会造成程序异常
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

myMat2=mat(loadDataSet('exp2.txt'))
print(createTree(myMat2,modelLeaf,modelErr),(1,10))