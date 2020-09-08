# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:39:01 2020
去噪 小波变换
@author: sc
"""
def denoise (dataa,tt):
    import pywt
    import numpy as np
    import matplotlib.pyplot as plt 
    min_index = 0
    max_index = len(dataa)
    data=np.array(dataa[min_index:max_index]) #小波变换数据
    dartanew=np.sort(np.abs(data))
    dartanew = [c**2 for c in dartanew]
    riskk=np.zeros(max_index)
    for kk in range(max_index):
        riskk[kk]=(max_index-2*kk+np.sum(dartanew[0:kk+1])+(max_index-kk)*dartanew[max_index-kk-1])/max_index
    threshold=np.sqrt(dartanew[np.argmin(riskk)])
    w = pywt.Wavelet('db8') #选用db4小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    #threshold = 0.01  # Threshold for filtering
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解

    plt.figure()
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    t1=tt[min_index:max_index]
    return datarec,t1
'''
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(tt[min_index:max_index], data[min_index:max_index])
    plt.xlim(2,2.5)
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("Raw signal")
    plt.subplot(3, 1, 2)
    plt.plot(tt[min_index:max_index], data[min_index:max_index])
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("Raw signal")
    plt.subplot(3, 1, 3)
    plt.plot(tt[min_index:max_index], datarec[min_index:max_index])
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("De-noised signal using wavelet techniques")
    plt.tight_layout()
    plt.savefig('de-noise.jpg',dpi=600) 
    plt.show()
'''
