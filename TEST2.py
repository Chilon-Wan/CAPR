# Effect of ARS with different number of samplings on the performance of CARS algorithm.

# This script will take a long time to run. The results have been stored in .\resulting_file\TEST2. 
# If you want to directly view the results, you can run .\resulting_plot\TEST2.py.

import os
from auxiliary_fun import *
from CARS import CARS_PLS
import numpy as np
import os
import multiprocessing
import psutil
import time

lock = multiprocessing.Lock()
def task(i, trainX, trainY, sampleNumTime, savePathName):
    t = time.time()
    result = CARS_PLS(trainX, trainY, iterNum=50, maxLatentVarNum=25,
                    useARS=True,
                    ARSNumTime=sampleNumTime,
                    useSelfFun=True, 
                    # flag=True
                    )

    lock.acquire()
    recordResult(savePathName,[[result['RMSE'], result['choiceVarNum'], result['latentVarNum']]])
    # print('第%d次训练: 训练集精度%.4f, 选择的波长数%d, PLS分解数量%d'%
    #         (i, result1['RMSE'], 
    #         result1['choiceVarNum'],
    #         result1['latentVarNum'])))
    print('第%d次循环花费时间%.4fs'%(i, time.time()-t))
    lock.release()

if __name__ == '__main__':
    # 多线程
    cpuUsedRate = 0.8
    repeatNum = 50
    threadTime = 10 # 倍数：加入进程池的计算数/线程数

    cpuNum = psutil.cpu_count(False)
    processNum  = int(cpuNum*cpuUsedRate)
    pool = multiprocessing.Pool(processNum)  # Pool(3) 表示创建容量为3个进程的进程池

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
    sampleNumTimeList = [0.6 + i * 0.2 for i in range(23)]
    sampleNumTimeList = sampleNumTimeList[::-1]
    for dataName in dataNameDict.keys():
        # 新建结果保存路径
        savePath = os.path.join(os.curdir, 'resulting_file', 'TEST2', dataName)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        dataChildNames = dataNameDict[dataName]
        # 遍历数据集的子目录
        for dataChildName in dataChildNames:
            trainX, trainY = testData[dataName][dataChildName]['train'][0], testData[dataName][dataChildName]['train'][1]
            # print(trainX.shape, trainY.shape)
            for sampleNumTime in sampleNumTimeList:
                savePathName = os.path.join(savePath, 
                            'iterNum(50)-maxPCN(25)-sampleNumTime(%.2f)-%s-%s.csv'%
                            (sampleNumTime, dataName, dataChildName))
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
                    pool.apply_async(task, (i, trainX, trainY, sampleNumTime, savePathName))
                    # task(circleNum, trainX, trainY, declineRate, savePathName)
                    kk += 1
                    if kk == threadTime*processNum: # 为使用线程数的整数倍
                        kk = 0
                        pool.close()
                        pool.join()
                        pool = multiprocessing.Pool(processNum)
    pool.close()
    pool.join()