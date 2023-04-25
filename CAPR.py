import numpy as np
from auxiliary_fun import *

def calcUsedWave(waveCoef, declineRate=0.95):
    waveCoef = np.abs(waveCoef)
    sortWaveCoef = np.sort(waveCoef)[::-1]
    sortWaveCoefArg = np.argsort(waveCoef)[::-1]
    sumWaveCoef = np.sum(waveCoef)
    theshold = sumWaveCoef * declineRate
    temp = 0
    for i in range(len(waveCoef)):
        temp += sortWaveCoef[i]
        if temp > theshold:
            usedWaveNum = i
            break
    return usedWaveNum, sortWaveCoefArg[:usedWaveNum]

def CAPR_PLS(X, y, declineRate=0.99, maxLatentVarNum=25, cv=5, useSelfFun=False, flag=False):
    dataNum, dataDim = X.shape # 样本数，数据维度
    randomSampleRate = 0.8 # 随机采样率80%(蒙特卡洛采样法)
    usedDataNum = np.round(dataNum * randomSampleRate) # 采样的样本数
    latentVarNumList = [] # 纪录每次迭代最优主成分数
    RMSEList = [] # 纪录每次迭代最小均方误差
    coefList = [] # 纪录每次迭代最佳回归系数
    choiceVarNumList = []
    choiceVarIndexList = [] # 纪录每次使用的波长指数
    usedWaveNum = dataDim
    usedWaveIndex = np.arange(usedWaveNum) # 选取的波长点索引
    while usedWaveNum >= 2:
        choiceVarNumList.append(usedWaveNum)
        usedDataIndex = np.random.choice(np.arange(dataNum), 
                        size=int(usedDataNum), replace=False) # 选取的样本索引
        choiceVarIndexList.append(usedWaveIndex)
        # print(usedDataIndex,usedWaveIndex)
        xCal = X[np.ix_(usedDataIndex, usedWaveIndex)] # 用于计算的x
        yCal = y[usedDataIndex] # 用于计算的y
        if useSelfFun:
            res = PLS_Estim2(xCal, yCal, min(maxLatentVarNum,usedWaveNum), cv)
        else:
            res = PLS_Estim1(xCal, yCal, min(maxLatentVarNum,usedWaveNum), cv)
        RMSE, coefAbbr, latentVarNum = res['RMSE'], res['coef'], res['latentVarNum']
        # print(len(latentVarNumList),latentVarNum, RMSE)
        waveCoef = [0 for _ in range(dataDim)]
        for i in range(usedWaveNum):
            waveCoef[usedWaveIndex[i]] = coefAbbr[i]
        
        usedWaveNum, usedWaveIndex = calcUsedWave(waveCoef, declineRate=declineRate)
        
        latentVarNumList.append(latentVarNum)
        RMSEList.append(RMSE)
        coefList.append(waveCoef)

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

    result = CAPR_PLS(trainDate, trainLabel, declineRate=0.99, maxLatentVarNum=25,
                     useSelfFun=True, 
                     ) 
    print(result)

