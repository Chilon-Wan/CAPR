import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

def PLS_Estim(X, y, maxLatentVarNum, cv, mpFlag=False):
    '''
    x : n x m
    y : n
    maxLatentVarNum: 最大潜变量数
    cv: 交叉验证数
    mpFlag: 是否使用多线程计算
    '''
    if mpFlag:
        n_jobs = -1
    else:
        n_jobs = None
    parameters = {'n_components':[i+1 for i in range(maxLatentVarNum)],}
    clf = PLSRegression()
    GS = GridSearchCV(clf, param_grid=parameters,
                    scoring='neg_root_mean_squared_error', # 以最大负均方误差（最小均方误差）为优化目标
                    cv=cv, n_jobs=n_jobs) # 运用所有线程计算
    GS = GS.fit(X, y)

    latentVarNum = GS.best_params_['n_components'] # 最优主成分
    RMSE = -GS.best_score_  # 最小均方误差
    coef = np.ravel(GS.best_estimator_.coef_) # 最佳回归系数

    result = {
        'RMSE':          RMSE,
        'latentVarNum':  latentVarNum,
        'coef':          coef,
        }
    
    return result

def calcUsedFeature(featureCoef, varCoefRemainRate):
    featureCoef = np.abs(featureCoef)
    sortFeatureCoef = np.sort(featureCoef)[::-1]
    sortFeatureCoefArg = np.argsort(featureCoef)[::-1]
    sumFeatureCoef = np.sum(featureCoef)
    theshold = sumFeatureCoef * varCoefRemainRate
    temp = 0
    for i in range(len(featureCoef)):
        temp += sortFeatureCoef[i]
        if temp > theshold:
            usedFeatureNum = i
            break
    return usedFeatureNum, sortFeatureCoefArg[:usedFeatureNum]

def CAPR_PLS(X, y, varCoefRemainRate=0.99, maxLatentVarNum=10, cv=5):
    '''
    x : n x m
    y : n
    varCoefRemainRate: 变量系数保留比率
    maxLatentVarNum: 最大潜变量数
    cv: 交叉验证数
    '''
    dataNum, dataDim = X.shape # 样本数，数据维度
    randomSampleRate = 0.8 # 随机采样率80%(蒙特卡洛采样法)
    usedDataNum = np.round(dataNum * randomSampleRate) # 采样的样本数
    latentVarNumList = [] # 纪录每次迭代最优主成分数
    RMSEList = [] # 纪录每次迭代最小均方误差
    coefList = [] # 纪录每次迭代最佳回归系数
    choiceVarNumList = []
    choiceVarIndexList = [] # 纪录每次使用的特征索引
    usedFeatureNum = dataDim
    usedFeatureIndex = np.arange(usedFeatureNum) # 选取的特征索引
    while usedFeatureNum >= 2:
        choiceVarNumList.append(usedFeatureNum)
        usedDataIndex = np.random.choice(np.arange(dataNum), 
                        size=int(usedDataNum), replace=False) # 选取的样本索引
        choiceVarIndexList.append(usedFeatureIndex)
        xCal = X[np.ix_(usedDataIndex, usedFeatureIndex)] # 用于计算的x
        yCal = y[usedDataIndex] # 用于计算的y
        res = PLS_Estim(xCal, yCal, min(maxLatentVarNum,usedFeatureNum), cv)
        RMSE, coefAbbr, latentVarNum = res['RMSE'], res['coef'], res['latentVarNum']
        featureCoef = [0 for _ in range(dataDim)]
        for i in range(usedFeatureNum):
            featureCoef[usedFeatureIndex[i]] = coefAbbr[i]
        
        usedFeatureNum, usedFeatureIndex = calcUsedFeature(featureCoef, varCoefRemainRate=varCoefRemainRate)
        
        latentVarNumList.append(latentVarNum)
        RMSEList.append(RMSE)
        coefList.append(featureCoef)

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

'''
if __name__ == '__main__':
    trainDate = [[]] # n x m
    trainLabel = []  # n
    result = CAPR_PLS(trainDate, trainLabel, varCoefRemainRate=0.99, maxLatentVarNum=6) 
    print(result['RMSE'])
'''
