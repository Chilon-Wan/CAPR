import numpy as np
import os
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 记录数据列表到文本
def recordResult(savefileName,dataList):
    '''
    savefileName: 存储的文件名
    dataList: 数据列表，格式为[[], [], [], ...]
    '''
    txt = ''
    for elem in dataList:
        if hasattr(elem, "__iter__"):
            for j in elem[:-1]:
                txt += '%f,'%(j,)
            txt += '%f\n'%(elem[-1],)
        else:
            txt += '%f,'%(elem,)
    
    path = os.path.dirname(savefileName)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(savefileName, 'a+') as f: #若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原
                                        #内容之后写入。可修改该模式（'w+','w','wb'等）
        f.write(txt) #将字符串写入文件中

# 从字典中得到一个数据集
def getOneDataSetFromDict(dataDict, dataName, dataChildName):
    trainDate, vlideDate = dataDict[dataName][dataChildName]['train'][0], \
                           dataDict[dataName][dataChildName]['vlide'][0]
    trainLabel, vlideLabel = dataDict[dataName][dataChildName]['train'][1], \
                           dataDict[dataName][dataChildName]['vlide'][1]
    return trainDate, trainLabel, vlideDate, vlideLabel

# 获取数据名字典
def getDataNameDict(dataDict):
    resDict = {}
    dataNames = dataDict.keys()
    for dataName in dataNames:
        resDict[dataName] = list(dataDict[dataName].keys())
    # print(resDict)
    return resDict