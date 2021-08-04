from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 在已选中一个alpha的情况下再随机选择一个不一样的alpha
# m 是可选下标长度，i，j皆为下标
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 调整大于H，或小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

'''
参数：数据集，类别标签，松弛变量C，容错率， 退出前最大的循环次数
C表示不同优化问题的权重，如果C很大，分类器会力图通过分离超平面将样例都正确区分，如果C很小又很难很好的分类，C值需要平衡两者
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b    # 预测类别，multiply：矩阵对应元素相乘
            Ei = fXi - float(labelMat[i])       # 预测误差 = 预测类别 - 实际类别，如果误差很大就要进行优化，如果在0-C之外，则已在边界，没有优化空间了，无需优化
            # 如果该数据向量可以被优化，则随机选择另一个数据向量，同时进行优化
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()     # 深拷贝，不会像复制只是引用，对象内容不会被更改
                # 计算上下界L和H,保证alpha在0-C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    '''
                    sum = alphas[i] + alphas[j]
                    如果 sum > C , 则 H = C，L = sum - C
                    如果 sum < C , 则 H = sum, L = 0
                    '''
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                # eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                # eta >= 0的情况比较少，并且优化过程计算复杂，所以此处做了简化处理，直接跳过了
                if eta >= 0: print("eta>=0"); continue
                # 获取alphas[j]的优化值
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 比对原值，看变化是否明显，如果优化并不明显则退出
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                # 对alphas[i]进行和alphas[j]量相同、方向相反的优化
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])                                                                       
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                aa="iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
                print(aa)
        '''
        如果所有向量都没有优化，则增加迭代数目，继续下一次循环
        当所有的向量都不再优化，并且达到最大迭代次数才退出，一旦有向量优化，iter都要归零
        所以iter存储的是在没有任何alpha改变的情况下遍历数据集的次数
        '''
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        bb="iteration number: %d" % iter
        print(bb)
    return b,alphas

dataArr,labelArr=loadDataSet('testSet.txt')
b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
print("****")
print(b)
for i in range(100):
    if alphas[i]>0.0:print(dataArr[i],labelArr[i])