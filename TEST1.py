# CAPR vs CARS without ARS vs CARS

# This script will take a long time to run. The results have been stored in .\resulting_file\TEST1. 
# If you want to directly view the results, you can run .\resulting_plot\TEST1.py.

from auxiliary_fun import *
from CAPR import CAPR_PLS
from CARS import CARS_PLS
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import psutil
import time

lock = multiprocessing.Lock()
def task(i, trainX, trainY, declineRate, savePathName):
    t = time.time()
    # CAPR
    result1 = CAPR_PLS(trainX, trainY, declineRate=declineRate, maxLatentVarNum=25, 
                        useSelfFun=True, 
                        )
    # CARS without ARS
    result2 = CARS_PLS(trainX, trainY, iterNum=len(result1['latentVarNumList']), maxLatentVarNum=25,
                    useARS=False,
                    useSelfFun=True, 
                    )
    # CARS
    result3 = CARS_PLS(trainX, trainY, iterNum=len(result1['latentVarNumList']), maxLatentVarNum=25,
                    useARS=True,
                    useSelfFun=True, 
                    )

    lock.acquire()
    recordResult(savePathName,
            [[result1['RMSE'], result1['choiceVarNum'], result1['latentVarNum'], len(result1['latentVarNumList']),
              result2['RMSE'], result2['choiceVarNum'], result2['latentVarNum'],
              result3['RMSE'], result3['choiceVarNum'], result3['latentVarNum'],
              ]])
    
    print('第%d次循环花费时间%.4fs'%(i, time.time()-t))
    lock.release()

if __name__ == '__main__':
    cpuUsedRate = 0.8
    repeatNum = 50
    threadTime = 10

    cpuNum = psutil.cpu_count(False)
    processNum  = int(cpuNum*cpuUsedRate)
    pool = multiprocessing.Pool(processNum) 

    print('物理核心数%d, 开启%d个线程开始计算'%(cpuNum, processNum))

    testData = readPublicData()
    dataNameDict = {
        'nir_shootout_2002': ['attr1', 'attr2', 'attr3'],
        'CGL_nir': ['casein', 'glucose', 'lactate', 'moisture'],
        'corn': ['oil', 'protein', 'starch'],
        'SWRI_Diesel_NIR': ['BP50GA', 'CNGA', 'D4052GA', 'FREEZEGA', 'TOTALGA', 'VISCGA'],
    }

    print(getDataNameDict(testData))
    kk = 0 # 记录加入线程池的数量
    declineRateList = [0.9 + i * 0.01 for i in range(10)]
    declineRateList = declineRateList[::-1]
    for dataName in dataNameDict.keys():
        # 新建结果保存路径
        savePath = os.path.join(os.curdir, 'resulting_file', 'TEST1', dataName)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        dataChildNames = dataNameDict[dataName]
        # 遍历数据集的子目录
        for dataChildName in dataChildNames:
            trainX, trainY = np.vstack([testData[dataName][dataChildName]['train'][0], testData[dataName][dataChildName]['vlide'][0]]), \
                             np.append(testData[dataName][dataChildName]['train'][1], testData[dataName][dataChildName]['vlide'][1])
            # print(trainX.shape, trainY.shape)
            for declineRate in declineRateList:
                savePathName = os.path.join(savePath, 
                            'CARS-compare-maxPCN(25)-declineRate(%.2f)-%s-%s.csv'%
                            (declineRate, dataName, dataChildName))
                # 防止意外中断程序，可以重新启动
                if os.path.exists(savePathName):
                    with open(savePathName) as f:
                        totalLineNum = len(f.readlines())
                        remainLineNum = repeatNum - totalLineNum
                    if remainLineNum > 0:
                        print('继续，剩余%d行:%s'%(remainLineNum, savePathName))
                        circleNum = remainLineNum
                    else:
                        print('已完成,共%d行:%s'%(totalLineNum, savePathName))
                        continue
                else:
                    print('开始写入%s:'%(savePathName, ))
                    circleNum = repeatNum

                # 创建进程池
                for i in range(circleNum):
                    # 使⽤异步⽅式执⾏work任务
                    pool.apply_async(task, (i, trainX, trainY, declineRate, savePathName))
                    kk += 1
                    if kk == threadTime*processNum: # 为使用线程数的整数倍
                        kk = 0
                        pool.close()
                        pool.join()
                        pool = multiprocessing.Pool(processNum)
    pool.close()
    pool.join()