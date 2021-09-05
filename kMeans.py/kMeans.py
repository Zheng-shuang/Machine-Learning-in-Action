from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):      
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) # 映射所有的元素为 float（浮点数）类型
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):            # 计算两个向量的欧式距离（可根据场景选择）
    return sqrt(sum(power(vecA - vecB, 2)))       # sqrt()计算一个非负实数的平方根

# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。
# 然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataSet, k):
    n = shape(dataSet)[1]        # 列的数量
    centroids = mat(zeros((k,n)))   # 创建k个质心矩阵
    for j in range(n):      # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)          # 范围 = 最大值 - 最小值
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))           # 随机生成k*1的[0,1)的数
    return centroids

datMat=mat(loadDataSet('testSet.txt'))
k=randCent(datMat,2)
# print(k)
# print(distEclud(datMat[0],datMat[1]))

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):    # distMeas: 计算数据到质心距离的函数,createCent: 创建质心函数
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))  # 第一列存放簇类index,第二列存放误差值
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):           # 遍历每一行数据，将数据划分到最近的质心
            minDist = inf; minIndex = -1
            for j in range(k):             # 计算第i个数据到每个质心的距离 
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:     # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2          # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        # print(centroids)
        for cent in range(k):        # 重新计算质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] # 获取某个簇类的所有点,.A表示矩阵转化为数组
            centroids[cent,:] = mean(ptsInClust, axis=0)  # 计算均值作为新的簇类中心
    return centroids, clusterAssment           # centroids：质心位置，clusterAssment：第一列是所属分类下标，第二列是点到质心距离

def plot(dataSet):
    """
    函数说明：绘制原数据集
    :param dataSet:
    :return:
    """
    x = dataSet[:, 0].tolist()
    y = dataSet[:, 1].tolist()
    plt.scatter(x, y)
    plt.show()

def plotKMeans(dataSet, clusterAssment, cenroids):
    """
    函数说明：绘制聚类后情况
    :param dataSet: 数据集
    :param clusterAssment: 聚类结果
    :param cenroids: 质心坐标
    :return:
    """
    m = np.shape(dataSet)[0]
    x0 = dataSet[np.nonzero(clusterAssment[:, 0] == 0), 0][0].tolist()
    y0 = dataSet[np.nonzero(clusterAssment[:, 0] == 0), 1][0].tolist()
    x1 = dataSet[np.nonzero(clusterAssment[:, 0] == 1), 0][0].tolist()
    y1 = dataSet[np.nonzero(clusterAssment[:, 0] == 1), 1][0].tolist()
    x2 = dataSet[np.nonzero(clusterAssment[:, 0] == 2), 0][0].tolist()
    y2 = dataSet[np.nonzero(clusterAssment[:, 0] == 2), 1][0].tolist()
    x3 = dataSet[np.nonzero(clusterAssment[:, 0] == 3), 0][0].tolist()
    y3 = dataSet[np.nonzero(clusterAssment[:, 0] == 3), 1][0].tolist()
    plt.scatter(x0, y0, color = 'red', marker='*')
    plt.scatter(x1, y1, color = 'yellow', marker='o')
    plt.scatter(x2, y2, color = 'blue', marker='s')
    plt.scatter(x3, y3, color = 'green', marker='^')
    for i in range(np.shape(cenroids)[0]):
        plt.scatter(cenroids[i, 0], cenroids[i, 1], color='k', marker='+', s=200)
    # plt.plot(cenroids[0,0], cenroids[0,1], 'k+', cenroids[1,0], cenroids[1,1], 'k+',cenroids[2,0],
    #          cenroids[2,1], 'k+',cenroids[3,0], cenroids[3,1], 'k+',)
    plt.show()

# dataSet = loadDataSet('testSet.txt')
# dataMat = np.mat(dataSet)
# plot(dataMat)
# cenroids, clusterAssment = kMeans(dataMat, 4)
# print(cenroids, clusterAssment)
# plotKMeans(dataMat, clusterAssment, cenroids)

# 二分 KMeans 聚类算法, 基于 kMeans 基础之上的优化，以避免陷入局部最小值
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))               # 保存每个数据点的簇分配结果和平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]              # # 质心初始化为所有数据点的均值,创建一个初始簇,tolist()作用：将矩阵（matrix）和数组（array）转化为列表。
    centList =[centroid0]  # 用来保存质心的列表,初始化只有 1 个质心的 list
    for j in range(m):  # 计算所有数据点到初始质心的距离平方误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):            # 当质心数量小于 k 时
        lowestSSE = inf
        for i in range(len(centList)):         # 对每一个质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] # 获取属于第i个簇类的所有数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 对属于第i个簇类的所有数据点进行k=2的聚类
            sseSplit = sum(splitClustAss[:,1]) # 计算对第i个簇类进行聚类后的SSE值,即将二分 kMeans 结果中的平方和的距离进行求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) # 计算不属于第i类的所有数据点的sse值
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:   # 将聚类后的SSE值与最低sse值进行比较,总的（未拆分和已拆分）误差和越小，越相似，效果越优化，划分的结果更好
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) # 将1改变为新增簇的编号,调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit # 将0改变为划分簇的编号
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 使用新生成的两个质心坐标代替原来的质心坐标
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss # 用新的聚类结果替换原来的
    return mat(centList), clusterAssment

datMat2=mat(loadDataSet('testSet2.txt'))
centList,myNewAssments=biKmeans(datMat2,3)
# plotKMeans(datMat2, myNewAssments, centList)

from math import radians, cos, sin, asin, sqrt

def distSLC(vecA, vecB):
    """
    函数说明：计算根据经纬度计算两点之间的距离
    :param vecA: 一个点坐标向量
    :param vecB: 另一个点坐标向量
    :return: 距离
    """
    a = sin(vecA[0,1] * np.pi/180) * sin(vecB[0,1] * np.pi/180)
    b = cos(vecA[0,1]* np.pi/180) * cos(vecB[0,1]* np.pi/180) * \
                      cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0

def clusterClubs(numClust = 5):
    """
    函数说明：对地图坐标进行聚类，并在地图图片上显示聚类结果
    :param numClust: 聚类数目
    :return:
    """
    clubsCoordinate = []
    fr = open('places.txt')
    for line in fr.readlines():
        lineCur = line.strip().split('\t')
        # lineMat = np.mat(lineCur)[0, -2:]   #获取最后两列经纬度
        # fltLinr = map(float, lineMat.tolist()[0])
        # clubsCoordinate.append(list(fltLinr))
        clubsCoordinate.append([float(lineCur[-1]), float(lineCur[-2])])    #获取最后两列经纬度
    clubsCoordinateMat = np.mat(clubsCoordinate)
    cenroids, clusterAssment = biKmeans(clubsCoordinateMat, numClust,distMeas=distSLC)
    plotKMeans(clubsCoordinateMat, clusterAssment, cenroids)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8] #使用矩阵来设置图片占绘制面板的位置，左下角0.1,0.1,右上角0.8,0.8
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<'] #形状列表
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')   #基于一幅图像来创建矩阵
    ax0.imshow(imgP)    #绘制该矩阵
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = clubsCoordinateMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90) #flatten()将m*n的矩阵转化为1*(m×n)的矩阵,.A[0]矩阵转化为数组后获取数组第一维数据
    ax1.scatter(cenroids[:, 0].flatten().A[0], cenroids[:, 1].flatten().A[0], marker='+', s=300)
    print(sum(clusterAssment[:,1]))
    plt.show()

if __name__ == '__main__':
    clusterClubs(4)