# Effect of ARS with different number of samplings on the performance of CARS algorithm.

import os,sys
res = os.path.abspath(__file__) # __file__是指当前的py文件，这行代码是取当前文件的绝对路径，加入到环境变量中。这样可以在cmd中运行了
base_path = os.path.dirname(os.path.dirname(res))  # os.path.dirname(res)是取父目录，2层是取上2级目录
sys.path.insert(0,base_path) # 把这个目录加入环境变量，这样就不用写死了

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 10
cm = plt.cm.get_cmap('jet')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def setAxStyle(ax):
    ax.ticklabel_format(style='sci', scilimits=(-2,3), axis='y')
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(10)#设置1e6的大小与位置
    ax.set_xticks(sampleNumTimeList[::2])
    # ax.set_xticklabels([declineRate for declineRate])

if __name__ == '__main__':
    savePath = os.path.join(os.curdir, 'resulting_file', 'TEST2')
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

    sampleNumTimeList = [0.6 + i * 0.2 for i in range(20)]
    sampleNumTimeListLen = len(sampleNumTimeList)
    allData0 = []
    titleNames = []
    for dataName in dataNameDict.keys():
        dataChildNames = dataNameDict[dataName]
        for dataChildName in dataChildNames:
            allData1 = []
            for sampleNumTime in sampleNumTimeList:
                savePathName = os.path.join(savePath, dataName, 
                                'iterNum(50)-maxPCN(25)-sampleNumTime(%.2f)-%s-%s.csv'%
                                (sampleNumTime, dataName, dataChildName))
                # print(savePathName)
                if os.path.exists(savePathName):
                    data = []
                    with open(savePathName, "r") as f:
                        for lineCode, line in enumerate(f.readlines()):
                            if line == '\n': # 消除空行
                                # print('出现空行：%d-%s'%(lineCode, savePathName))
                                continue
                            tempList = list(map(lambda w: float(w), line.strip('\n').split(',')))
                            if len(tempList) != 3: # 消除长度不对的
                                print('长度异常：%d-%s'%(lineCode+1, savePathName))
                                continue
                            data.append(tempList)
                    # print(len(np.mean(data, axis=0)))
                    allData1.append(np.mean(data, axis=0))
            allData0.append(np.array(allData1))
            titleNames.append("%s:%s"%(petName[dataName],dataChildName))

    figNum = len(allData0)
    flag = 0 # Root mean square error = 0 , Number of variables selected = 1 
    fig = plt.figure(figsize=(9,6.5))
    for i, titleName, data in zip(range(1,figNum+1), titleNames, allData0):
        ax = fig.add_subplot(4, 4, i)
        ax.set_title(titleName)
        # print(data.shape)
        if flag == 0:
            ax.plot(sampleNumTimeList, data[:,0],
                          marker='.', 
                          color=colors[0], label='CARS without ARS')
        else:   
            ax.plot(sampleNumTimeList, data[:,1], 
                          marker='.', 
                          color=colors[0], label='CARS without ARS')
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.text( x1 + (x2 - x1) * 0.95, y1 + (y2 - y1) * 0.85, 'Test %d'%(i,), ha='right')
        # plt.legend(loc='upper right') # 内部图例
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0) # 外部图例
        ax.ticklabel_format(style='sci', scilimits=(-1,3), axis='y')
        ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
        ax.yaxis.get_offset_text().set_fontsize(9)#设置1e6的大小与位置
        ax.set_xticks(sampleNumTimeList[::3])

    # plt.subplots_adjust(left=0.08, bottom=0.07, right=0.97, top=0.95, hspace=0.33, wspace=0.2) # 调整子图与整体间距
    plt.subplots_adjust(top=0.895,bottom=0.08,left=0.08,right=0.97,hspace=0.505,wspace=0.28) # 调整子图与整体间距
    lanelFontDict = {'size': 14}
    fig.text( 0.5, 0.07/4, 'Ratio of number of samplings to number of variables', ha='center', fontdict=lanelFontDict)
    if flag == 0:
        fig.text( 0.08/4, 0.5, 'Root mean square error', va='center', rotation='vertical', fontdict=lanelFontDict)
    else:
        fig.text( 0.08/4, 0.5, 'Number of variables selected', va='center', rotation='vertical', fontdict=lanelFontDict)
    plt.show()



