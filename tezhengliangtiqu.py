# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:37:17 2020

@author: sc
"""

def tezhengti (dataa):
    import numpy as np
    import math
    import pywt
    def fuliye(xinhao):
        n=len(xinhao)
        k=np.arange(n)/n
        f=2000*2.56 #Hz，采样频率
        frq=f*k
        frq=frq[range(int(n/2))]
        Y=np.fft.fft(xinhao)/n
        Y=Y[range(int(n/2))]
        return frq,Y

    N=len(dataa)
    #时域特征参量
    Vmax=np.max(dataa)
    Vmin=np.min(dataa)
    Vpp=Vmax-Vmin
    Vc=np.sum(dataa)/N
    Vcabs=np.sum(np.abs(dataa))/N
    Vrms=np.sqrt(np.sum(np.square(dataa))/N)
    aa=np.mean(dataa)
    dataa3 = [(c-aa)**2 for c in dataa]
    sigma=np.sqrt(np.sum(dataa3)/N)
    dataa4 = [c**3 for c in dataa]
    alfa=np.sum(dataa4)/N
    dataa5 = [(c-aa)**4 for c in dataa]
    beta=np.sum(dataa5)/N
    Kv=beta/(sigma**4)
    Sf=Vrms/Vcabs
    Cf=Vmax/Vrms
    CLf=Vmax/((np.sum(np.sqrt(np.abs(dataa)))/N)**2)
    If=Vmax/Vcabs
    #功率谱熵
    F_f,F_da=fuliye(dataa)
    S1=np.square(np.abs(F_da))
    S=[c/(2*N*math.pi) for c in S1]
    Hf=0
    for i in range(len(S)):
        p=S[i]/np.sum(S)
        Hf=Hf+p*np.log(p)
    Hf=-1*Hf
    #小波包熵
    w = pywt.Wavelet('db8')
    #max_level=pywt.dwt_max_level(N, w)
    n=3
    wp = pywt.WaveletPacket(dataa, w,maxlevel=n)#wp为小波包树
    #计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = dataa
    for row in range(1,n+1):
        #lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data
    re = []  #第n层所有节点的分解系数
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    #第n层能量特征
    energy = []
    for i in re:
        energy.append(pow(np.linalg.norm(i,ord=None),2))
    Hwp=0
    for i in range(2**n):
        p=energy[i]/np.sum(energy)
        Hwp=Hwp+p*np.log(p)
    Hwp=-1*Hwp
    tezheng=[Hf,Hwp,Vpp,sigma,Vc,Vmax]#Kv,Vcabs,beta,Sf,Cf,CLf,Vrms,alfa,If
    return tezheng
    '''
    能量图绘制
    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(10, 7), dpi=80)
    # 柱子总数
    NN = 8
    values = energy
    # 包含每个柱子下标的序列
    index = np.arange(NN)
    # 柱子的宽度
    width = 0.45
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(index, values, width, label="num", color="#87CEFA")
    plt.xlabel('clusters')
    plt.ylabel('number of reviews')
    plt.title('Cluster Distribution')
    plt.xticks(index, ('1', '2', '3', '4', '5', '6', '7', '8'))
    plt.show()
    '''
