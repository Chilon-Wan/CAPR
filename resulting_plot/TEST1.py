# CAPR vs CARS without ARS vs CARS

import os,sys
res = os.path.abspath(__file__) # __file__是指当前的py文件，这行代码是取当前文件的绝对路径，加入到环境变量中。这样可以在cmd中运行了
base_path = os.path.dirname(os.path.dirname(res))  # os.path.dirname(res)是取父目录，2层是取上2级目录
sys.path.insert(0,base_path) # 把这个目录加入环境变量，这样就不用写死了

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 10
cm = plt.cm.get_cmap('jet')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def setAxStyle(ax):
    ax.ticklabel_format(style='sci', scilimits=(-2,3), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(9)#设置1e6的大小与位置
    ax.set_xticks(declineRateList[::2])
    # ax.set_xticklabels([declineRate for declineRate])

if __name__ == '__main__':
    savePath = os.path.join(os.curdir, 'resulting_file', 'TEST1')
    dataNameDict = {
        'nir_shootout_2002': ['attr1', 'attr2', 'attr3'],
        'CGL_nir': ['casein', 'glucose', 'lactate', 'moisture'],
        'corn': ['oil', 'protein', 'starch'],
        'SWRI_Diesel_NIR': ['BP50GA', 'CNGA', 'D4052GA', 'FREEZEGA', 'TOTALGA', 'VISCGA'],
    }
    petName = { 'nir_shootout_2002': 'Shootout',
                'CGL_nir': 'Grain',
                'corn': 'Corn',
                'SWRI_Diesel_NIR': 'Diesel',
                }

    declineRateList = [0.9 + i * 0.01 for i in range(10)]
    declineRateListLen = len(declineRateList)
    allData0 = []
    titleNames = []
    for dataName in dataNameDict.keys():
        dataChildNames = dataNameDict[dataName]
        for dataChildName in dataChildNames:
            allData1 = []
            for declineRate in declineRateList:
                savePathName = os.path.join(savePath, dataName, 
                                'CARS-compare-maxPCN(25)-declineRate(%.2f)-%s-%s.csv'%
                                (declineRate, dataName, dataChildName))
                # print(savePathName)
                if os.path.exists(savePathName):
                    data = []
                    with open(savePathName, "r") as f:
                        for lineCode, line in enumerate(f.readlines()):
                            if line == '\n': # 消除空行
                                # print('出现空行：%d-%s'%(lineCode, savePathName))
                                continue
                            tempList = list(map(lambda w: float(w), line.strip('\n').split(',')))
                            if len(tempList) < 7: # 消除长度不对的
                                print('长度异常：%d-%s'%(lineCode+1, savePathName))
                                continue
                            data.append(tempList)
                    # print(len(np.mean(data, axis=0)))
                    allData1.append(np.mean(data, axis=0))
            allData0.append(np.array(allData1))
            titleNames.append("%s:%s"%(petName[dataName],dataChildName))

    flag = 0 # Root mean square error = 0 , Number of variables selected = 1 
    figNum = len(allData0)
    fig = plt.figure(figsize=(8.8,6.5))
    for i, titleName, data in zip(range(1,figNum+1), titleNames, allData0):
        ax = fig.add_subplot(4, 4, i)
        ax.set_title(titleName)
        if flag == 0:
            l1, = ax.plot(declineRateList, data[:,4], marker='^', color=colors[0], label='CARS without ARS')
            l2, = ax.plot(declineRateList, data[:,7], marker='o', color=colors[1], label='CARS')
            l3, = ax.plot(declineRateList, data[:,0], marker='s', color='r', label='CAPR')
        else: 
            l1, = ax.plot(declineRateList, data[:,5], marker='^', color=colors[0], label='CARS without ARS')
            l2, = ax.plot(declineRateList, data[:,8], marker='o', color=colors[1], label='CARS')
            l3, = ax.plot(declineRateList, data[:,1], marker='s', color='r', label='CAPR')

        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.text( x1 + (x2 - x1) * 0.95, y1 + (y2 - y1) * 0.85, 'Test %d'%(i,), ha='right')
        ax.ticklabel_format(style='sci', scilimits=(-1,3), axis='y')
        ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
        ax.yaxis.get_offset_text().set_fontsize(9)#设置1e6的大小与位置
        ax.set_xticks(declineRateList[::2])

    # 共享图例
    handles = [l1, l2, l3]
    titleWidth = 0.2*len(handles)
    ax = fig.add_subplot(1, 1, 1)
    ax.legend(handles=handles,  mode='expand', bbox_to_anchor=((1-titleWidth)/2, 1.12, titleWidth, 0), ncol=len(handles), borderpad=0.6)
    ax.axis('off')  # 去掉坐标的刻度

    # plt.subplots_adjust(left=0.08, bottom=0.07, right=0.97, top=0.9, hspace=0.33, wspace=0.2) # 调整子图与整体间距
    plt.subplots_adjust(top=0.895,bottom=0.08,left=0.08,right=0.97,hspace=0.505,wspace=0.2) # 调整子图与整体间距
    lanelFontDict = {'size': 14}
    fig.text( 0.5, 0.07/4, 'Retention rate', ha='center', fontdict=lanelFontDict)
    if flag == 0:
        fig.text( 0.08/4, 0.5, 'Root mean square error', va='center', rotation='vertical', fontdict=lanelFontDict)
    else:
        fig.text( 0.08/4, 0.5, 'Number of variables selected', va='center', rotation='vertical', fontdict=lanelFontDict)
    plt.show()

