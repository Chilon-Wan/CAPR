import numpy as np
from auxiliary_fun import *

def CARS_PLS(X, y, iterNum=50, maxLatentVarNum=25, cv=5, useARS=False ,ARSNumTime=1, useSelfFun=False, flag=False):
    XRowNum, XColNum = X.shape # 样本数，数据维度
    randomSampleRate = 0.8 # 随机采样率80%(蒙特卡洛采样法)
    usedDataNum = np.round(XRowNum * randomSampleRate) # 采样的样本数
    u = np.power((XColNum/2), (1/(iterNum-1))) # 衰减指数系数
    k = (1/(iterNum-1)) * np.log(XColNum/2) # 衰减比例系数
    # print(u, k)
    eachChoiceVarRate = [u*np.exp(-1*k*i) for i in range(1, iterNum+1)] # 计算每次迭代使用的变量点的比例
    eachChoiceVarNum = [round(rate*XColNum) for rate in eachChoiceVarRate] # 计算每次迭代使用的变量点数量，指数剔除
    # eachChoiceVarNum = range(5,dataDim+1, 10) # 计算每次迭代使用的变量点数量，匀速剔除
    coef = [1 for _ in range(XColNum)] # 回归系数，未选择的变量为0
    latentVarNumList = [] # 纪录每次迭代最优潜变量数
    RMSEList = [] # 纪录每次迭代最小均方误差
    coefList = [] # 纪录每次迭代最佳回归系数
    choiceVarIndexList = [] # 纪录每次使用的变量指数
    choiceVarNumList = []   # 纪录每次使用的变量数
    for choiceVarNum in eachChoiceVarNum:
        choiceDataIndex = np.random.choice(np.arange(XRowNum), 
                        size=int(usedDataNum), replace=False) # 选取的样本索引
        choiceVarIndex = np.argsort(np.abs(coef))[XColNum-choiceVarNum:] # 选取的变量点索引

        # 自加权重采样
        if useARS:
            choiceVarIndexWeight = np.abs(coef)[choiceVarIndex]
            choiceVarIndexWeight = choiceVarIndexWeight/choiceVarIndexWeight.sum()
            secondChoice = np.unique(np.random.choice(
                choiceVarNum,size=int(choiceVarNum*ARSNumTime),replace=True,p=choiceVarIndexWeight))
            choiceVarIndex = choiceVarIndex[secondChoice]
            choiceVarNum = len(choiceVarIndex)

        choiceVarIndexList.append(choiceVarIndex)
        xCal = X[np.ix_(choiceDataIndex, choiceVarIndex)] # 用于计算的x
        yCal = y[choiceDataIndex] # 用于计算的y
        if useSelfFun:
            res = PLS_Estim2(xCal, yCal, min(maxLatentVarNum,choiceVarNum), cv)
        else:
            res = PLS_Estim1(xCal, yCal, min(maxLatentVarNum,choiceVarNum), cv)
        RMSE, coefAbbr, latentVarNum = res['RMSE'], res['coef'], res['latentVarNum']

        # 记录回归系数
        coef = [0 for _ in range(XColNum)]
        for i in range(choiceVarNum):
            coef[choiceVarIndex[i]] = coefAbbr[i]
        latentVarNumList.append(latentVarNum)
        RMSEList.append(RMSE)
        coefList.append(coef)
        choiceVarNumList.append(choiceVarNum)
    bestIterIndex = np.argmin(RMSEList)
    result = {
        'RMSEList':          RMSEList,
        'choiceVarIndexList':choiceVarIndexList,
        'choiceVarNumList':  choiceVarNumList,
        'latentVarNumList':  latentVarNumList,
        'coefList':          coefList,
        # ------
        'RMSE':          RMSEList[bestIterIndex],
        'choiceVarIndex':choiceVarIndexList[bestIterIndex],
        'choiceVarNum':  choiceVarNumList[bestIterIndex],
        'latentVarNum':  latentVarNumList[bestIterIndex],
        'coef':          coefList[bestIterIndex],
        }
    return result

if __name__ == '__main__':
    allData = readPublicData()
    trainDate, trainLabel, vlideDate, vlideLabel = getOneDataSetFromDict(allData, 'CGL_nir', 'casein')
    print(trainDate.shape, trainLabel.shape, vlideDate.shape, vlideLabel.shape)

    result = CARS_PLS(trainDate, trainLabel, 
                useSelfFun=True, 
                useARS=True,
                flag=True)
    print(result)
