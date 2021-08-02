from numpy import *
 
'''
数据取自斑点犬爱好者的留言板，进行此条切分后得到 postingList
classVec 对数据进行标记，1：侮辱性，0：非侮辱性
返回数据集与标签
'''
def loadDataSet():
    #  flea 虱子  dalmation 一种斑点狗    lick 舔   steak 牛排
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec
 
# 建立无重复词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
 
'''
采用词集模型：即对每条文档只记录某个词汇是否存在，而不记录出现的次数
创建一个与词汇表长度一致的0向量，在当前样本中出现的词汇标为1
将一篇文档（一条留言）转换为词向量
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: ", word, "is not in my Vocabulary!")
    return returnVec
 
'''
采用词袋模型：即对每条文档记录各个词汇出现的次数
与词集模型的代码几乎一致，除了计算词汇量的几个地方
'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = returnVec[vocabList.index(word)] + 1
    return returnVec
 
'''
在已知类别的情况下统计各个词出现的频率
trainMatrix: 文档矩阵，trainCategory：标签向量
'''
def trainNB0(trainMatrix, trainCategory):
    # 文档数
    numTrainDocs = len(trainMatrix)
    # 总词汇数
    numWords = len(trainMatrix[0])
    # 文档中侮辱类文档的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    '''
    优化1：为避免一个概率值为0，导致各个概率值的乘积为零，此处进行优化;
    改完之后效果显著
    '''
    # p0Num = p1Num = zeros(numWords)
    # p0Denom = p1Denom = 0.0   # denom：分母项
    p0Num = p1Num = ones(numWords)
    p0Denom = p1Denom = 2.0
 
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 对于每个文档来说，里面的词只分有和没有
            # 对于所有文档来说，关注的是词在每个文档中是否出现的概率
            p1Num = p1Num + trainMatrix[i]
            p1Denom = p1Denom + sum(trainMatrix[i])
        else:
            p0Num = p0Num + trainMatrix[i]
            p0Denom = p0Denom + sum(trainMatrix[i])
    '''
    优化2：很多很小的概率值相乘可能导致溢出，或者浮点数舍入导致的错误
    方法就是取对数 ln, 概率比值会有所变化，但不影响极值点和大小上的比较
    '''
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive
 
# 输入某个文档，计算总概率值，比较两者得出结论
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    try:
        p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    except BaseException:
        print(pClass1)
 
    if p1 > p0:
        return 1
    else:
        return 0
 
def testingNB():
    listOPosts, listClasses = loadDataSet()  # 获取数据集与标签
    myVocabList = createVocabList(listOPosts)  # 建立词汇表
    myVocabList.sort()  # 排序，使其顺序一致，打印出来好查看
    # print(myVocabList)  # ['I', 'ate', 'buying', 'cute', 'dalmation', 'dog', 'flea', 'food', 'garbage', 'has', 'help', 'him', 'how', 'is', 'licks', 'love', 'maybe', 'mr', 'my', 'not', 'park', 'please', 'posting', 'problems', 'quit', 'so', 'steak', 'stop', 'stupid', 'take', 'to', 'worthless']
 
    '''
    词集模型测试
    '''
    trainMat = []  # 将所有文档转化为一个文档矩阵
    for postingDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)  # 训练分类器
    # print(pAb) # 任意文档属于侮辱性文档的概率=0.5
    # print(p0V)  # 在两个类别下的概率
    # print(p1V)
    # print(column_stack((myVocabList, p0V, p1V))) # 更好地查看各个词对应的侮辱性的概率
 
    # 测试
    testEntry = ['love', 'my', 'dalmation', 'not', 'licks']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
 
    '''
    词袋模型测试
    '''
    trainMat = []  # 将所有文档转化为一个文档矩阵
    for postingDoc in listOPosts:
        trainMat.append(bagOfWords2VecMN(myVocabList, postingDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)  # 训练分类器
    testEntry = ['love', 'my', 'dalmation', 'not', 'licks']
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(bagOfWords2VecMN(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
 
# if __name__ == "__main__":
#     testingNB()


'''
使用贝叶斯过滤垃圾邮件：先从文本中得到字符串列表，然后生成词向量
1、收集数据
2、准备数据：文本内容解析成词条向量
3、分析数据：检查词条确保解析的正确性
4、训练算法：已实现
5、测试算法：构建一个新的测试函数来计算文档及的错误率
6、使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上
'''
import random
# from bayes01_base_model import *
 
# 采用正则表达式来切分，把除单词、数字以外的字符全部作为分隔符，去除空字符串，单词统一为小写
# 为避免url中无意义的字符串乱入，把长度小于3的字符串去掉
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split('[^\w\u4e00-\u9fff]+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        filename = 'email/spam/' + str(i) + '.txt'
        wordList = textParse(open(filename).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        filename = 'email/ham/' + str(i) + '.txt'
        wordList = textParse(open(filename).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)                ## 词汇去重
    trainingSet = list(range(50)); testSet=[]           #create test set
    # 将50封邮件中的随机10封邮件加入测试集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    return float(errorCount)/len(testSet)
    #return vocabList,fullText
 
if __name__ == "__main__":
    print("1次交叉验证的平均错误率是，", spamTest())
 
    errAverage = 0
    for i in range(10):
        errAverage = errAverage + spamTest()
    errAverage = errAverage / 10
    print("10次交叉验证的平均错误率是，", errAverage)
