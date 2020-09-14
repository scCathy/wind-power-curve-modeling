# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:25:37 2020

@author: sc
"""

import matplotlib.pyplot as plt 
#from matplotlib import colors
import pandas as pd
import numpy as np
#import math
#from matplotlib.colors import LinearSegmentedColormap
import os
#import denoise
import svmxunlian
import tezhengliangtiqu
#from scipy import signal
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False


# 遍历文件夹及其子文件夹中的文件
def get_filelist(dirr,name_list):
    if (os.path.isfile(dirr)):
        if os.path.split(os.path.dirname(dirr))[1] == "duanchi":
            name_list['duanchi'].append(dirr)
        elif os.path.split(os.path.dirname(dirr))[1] == "mosun":
            name_list['mosun'].append(dirr)
        elif os.path.split(os.path.dirname(dirr))[1] == "dianshi":
            name_list['dianshi'].append(dirr)
        #elif os.path.split(os.path.dirname(dirr))[1] == "dianmo":
            #name_list['dianmo'].append(dirr)
        elif os.path.split(os.path.dirname(dirr))[1] == "normal":
            name_list['normal'].append(dirr)
    elif os.path.isdir(dirr):
        for s in os.listdir(dirr):
            if s == "duanchi-mosun":
                continue
            newDir=os.path.join(dirr,s)
            get_filelist(newDir,name_list)
    return name_list

path = 'G:/wind-energy/gear-box/faults-data/gear-fault-data-set-1/'
name_list={'normal':[],'dianshi':[],'mosun':[],'duanchi':[]}#'dianmo':[],
name_list = get_filelist(path, name_list)
value={'normal':[],'dianshi':[],'mosun':[],'duanchi':[]}#'dianmo':[],
dlist=list(value.keys())
for i in range(4):
    jian=dlist[i]#索引值即为字符
    for j in range(len(name_list[jian])):
        file_name = (os.path.basename(name_list[jian][j])).split('.')[0]
        dat=pd.read_table(name_list[jian][j], header = None,usecols = range(9))
        
        if j==0:
            value[jian]={'0':dat}
        else:
            #value[jian]={str(j):dat}
            value[jian].update({str(j):dat})
'''
#傅里叶变换
def fuliye(xinhao):
    n=len(xinhao)
    k=np.arange(n)/n
    f=2000*2.56 #Hz，采样频率
    frq=f*k
    frq=frq[range(int(n/2))]
    Y=np.fft.fft(xinhao)/n
    Y=Y[range(int(n/2))]
    return frq,Y
'''
#df_news = pd.read_table(r'G:\wind-energy\gear-box\faults-data\gear-fault-data-set-1\mosun\mosun880.txt',header = None,usecols = range(9))
name=['normal','dianshi','mosun','duanchi']#,'dianmo'
length=[len(value[name[0]]),len(value[name[1]]),len(value[name[2]]),len(value[name[3]])]#,len(value[name[4]]
sampledata=[]#样本库
for i in range(len(value)):
    for j in range(length[i]):
        for z in range(8,9):#只用了通道8的信号
            xinhao=value[name[i]][str(j)][z].tolist() #循环取通道信号
            te1=tezhengliangtiqu.tezhengti(xinhao) #某一通道信号的特征量提取
            te1=np.append(te1,int(i))#i为标签，0-4与name对应
            if not len(sampledata):
                sampledata=[te1.tolist()]
            else:
                sampledata=np.concatenate((sampledata,[te1.tolist()]),axis=0)

#相关度计算
from pandas.core.frame import DataFrame
samp=DataFrame(sampledata[:,0:-1])
print(samp.corr())
samp.corr().to_excel('test1.xlsx', index=True, header=None)

markers=('s','x','o','^','v')
colors=('red','blue','lightgreen','gray','cyan')
i=2
for idx,c1 in enumerate(np.unique(y)):
    tezheng1=sampledata[sampledata[:,-1]==c1,0]
    plt.plot(range(len(tezheng1)),tezheng1,alpha=0.8,c=colors[idx],marker=markers[idx],label=c1)
plt.legend(['normal','dianshi','mosun','duanchi'],loc='best')
plt.show()

plt.scatter(range,sampledata[:,0])

svmxunlian.svmxunlian(sampledata,length)
            

    
f=2000*2.56 #Hz,采样频率
T=1/f
zhuansu_real=880
changdu=len(zhuansu)
t=np.arange(0,changdu,1)*T

'''
#绘制原始振动信号的图
fig, ax = plt.subplots(8, 1, figsize=(12,12))
for i in range(1, 9):
    da=df_news[[i]]
    #F_f,F_da=fuliye(da)
    #ax[i-1].plot(frq,abs(Y),'r') # plotting the spectrum
    ax[i-1].plot(t,da,'r') # plotting the spectrum
    ax[i-1].set_xlabel('Freq (Hz)')
    ax[i-1].set_ylabel('幅值')
    ax[i-1].set_title('CH%d' %i)
    plt.tight_layout()
plt.savefig('信号.jpg',dpi=600) 
plt.show()
#da,t1=denoise.denoise(ax_lout_mv,t)#降噪
#绘制FFT变换后的振动信号
fig, ax = plt.subplots(8, 1, figsize=(12,12))
for i in range(1, 9):
    da=df_news[[i]]
    F_f,F_da=fuliye(da)
    ax[i-1].plot(F_f,abs(F_da),'r') # plotting the spectrum
    ax[i-1].set_xlabel('Freq (Hz)')
    ax[i-1].set_ylabel('幅值')
    ax[i-1].set_title('CH%d' %i)
    plt.tight_layout()
plt.savefig('信号FFT.jpg',dpi=600) 
plt.show()
#绘制降噪后FFT后的振动信号
fig, ax = plt.subplots(8, 1, figsize=(15,15))
for i in range(1, 9):
    da=df_news[i].tolist()
    da,t1=denoise.denoise(da,t)#降噪
    F_f,F_da=fuliye(da)
    ax[i-1].plot(F_f,abs(F_da),'r') # plotting the spectrum
    ax[i-1].set_xlabel('Freq (Hz)')
    ax[i-1].set_ylabel('幅值')
    ax[i-1].set_title('CH%d' %i)
    plt.tight_layout()
plt.savefig('降噪信号FFT.jpg',dpi=600) 
plt.show()
'''
