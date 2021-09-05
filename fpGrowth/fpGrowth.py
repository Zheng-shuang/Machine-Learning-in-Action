class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue      # 这个结点所存的字母
        self.count = numOccur      # 结点计数器
        self.nodeLink = None       # 指向下一个同字母的结点的指针
        self.parent = parentNode   # 指向父节点的指针，用于上溯
        self.children = {}         # 子结点的指针集

    def inc(self, numOccur):       # 更新结点计数器
        self.count += numOccur
        
    def disp(self, ind=1):           
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1):          # 根据数据创建FP树，minSup为出现次数的阈值
    headerTable = {}                        # 字母表，需要存储两个信息：1.该字母的出现次数，2.指向该字母出现在FP树上的头指针
    for trans in dataSet:                   # 统计每个字母出现的次数
        for item in trans:           
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]        # get(item, 0)提取item关键词的值，没有则返回0
    for k in list(headerTable.keys()):  # 枚举每一个字母，去除掉那些出现次数低于阈值的字母
        if headerTable[k] < minSup: 
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    # print('freqItemSet: ',freqItemSet)
    if len(freqItemSet) == 0: return None, None 
    for k in headerTable:                   # 为headerTable开辟多一个位置，存放头指针
        headerTable[k] = [headerTable[k], None] # 重新格式化headerTable以使用节点链接
    # print('headerTable: ',headerTable)

    retTree = treeNode('Null Set', 1, None)  # 创建根节点
    for tranSet, count in dataSet.items():  # 将每一条数据插进FP树中，期间需要去除掉数据中出现次数低于阈值的字母，且数据中字母需要按出现次数进行降序排序
        localD = {}        # 用于存放该数据中符合条件的字母
        for item in tranSet:  
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]   #将符合条件的字母按出现次数进行降序排序,v[0]表示只要前面的字母
            updateTree(orderedItems, retTree, headerTable, count)     # 然后将其插入FP树中
    return retTree, headerTable    # 返回FP树和字母表

def updateTree(items, inTree, headerTable, count):       # 将一条数据插进FP树中，类似于将一条字符串插进Trie树中
    if items[0] in inTree.children:   # 若首字母的结点存在，则直接更细该节点的计数器
        inTree.children[items[0]].inc(count)  #增量计数
    else:  
        inTree.children[items[0]] = treeNode(items[0], count, inTree)   # 创建新结点，之后需要将该结点放进字母表的链表中
        if headerTable[items[0]][1] == None:       # 如果该字母首次出现，则直接将字母表的头指针指向该结点
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:                # 否则，需要将其插入到合适的位置，这里的做法是尾插法
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:      # 使用剩余的已排序的items调用updateTree（）
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   # 将新建的字母结点加入到字母表链的链尾，但个人认为头插法更优
    while (nodeToTest.nodeLink != None):    # 不要使用递归遍历链表！
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

simpDat=loadSimpDat()
initSet=createInitSet(simpDat)
myFPtree,myHeaderTab=createTree(initSet,3)
# myFPtree.disp()

def ascendTree(leafNode, prefixPath): # 在FP树，从一个结点开始，上溯至根节点，并记录路径。这样就找到了频繁项的一个前缀路径
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
def findPrefixPath(basePat, treeNode): # 在FP树，找出某个字母所有的前缀路径，即找到对应的条件模式基
    condPats = {}       # 存储前缀路径，为何要用字典的形式？因为还要记录每条前缀路径的出现次数，然后又用来创建FP树
    while treeNode != None:
        prefixPath = []       # 保存当前的前缀路径
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:       # 因为该节点也被加进了路径当中，所以需要路径的长度大于1
            condPats[frozenset(prefixPath[1:])] = treeNode.count     # 将前缀路径并其出现次数存起来
        treeNode = treeNode.nodeLink    # 沿着字母表链，走向下一个结点，继续寻找前缀路径
    return condPats

print(findPrefixPath('t',myHeaderTab['t'][1]))