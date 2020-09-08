# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:47:26 2019

@author: sc
"""

import xlrd 
#import xlwt
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei']
#LocalOutlierFactor

#clf = LocalOutlierFactor(n_neighbors=4, contamination=0.12)

wb=xlrd.open_workbook('15-1.xlsx')
#print(wb.sheet_names())
sheet6 = wb.sheet_by_name('Sheet11')
#workbook = xlwt.Workbook() 
#sheet11 = workbook.add_sheet('Worksheet')




#y_pred=np.zeros((1, 1))
    #i=16


X11 = sheet6.col_values(1)#获取列内容
X1 = [x for x in X11 if x != '']
    #nsampl=len(X1)

#X1= np.array(X1).reshape(1,-1)
#print(X1)

X22 = sheet6.col_values(2)
X2 = [x for x in X22 if x != '']
#X2= np.array(X2).reshape(1,-1)
#print(X2)
    #plt.scatter(X1, X2, c = "red", marker='o')  
XX=np.dstack((X1,X2)).reshape(-1,2)
XX=np.vstack(sorted(XX, key=lambda XX: XX[0]))
lenXX=np.size(XX,0)
N=15
geshu=23/N
x0=[]
x1=[]
x2=[]
#print(X)
for ii in range(3,4):
    nn=0;X1=zeros((1,2));X=zeros((1,2))
    startt=ii*geshu;endd=(ii+1)*geshu
    for j in range(0,lenXX):
        if (XX[j][0]>=np.float(startt)) & (XX[j][0]<np.float(endd)):
            if nn==0:
                X[nn]=[XX[j][0],XX[j][1]]
                nn=nn+1
            else:
                X1[0]=[XX[j][0],XX[j][1]]
                X=np.concatenate((X,X1),axis=0)
                nn=nn+1
        elif XX[j][0]<np.float(startt):
            continue
        elif XX[j][0]>=np.float(endd):
            break
    #X=X.reshape(-1,2)
    estimator = KMeans(n_clusters=3)#构造聚类器
    estimator.fit(X)#聚类
    label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
    x00 = X[label_pred == 0]
    x11 = X[label_pred == 1]
    x22 = X[label_pred == 2]
    if ii==ii:
        x0=x00
        x1=x11
        x2=x22
    else:
        x0=np.r_[x0, x00]
        x1=np.r_[x1, x11]
        x2=np.r_[x2, x22]
    
plt.scatter(x0[:, 0], x0[:, 1], c = "blue", marker='+', label='正常数据') 
plt.scatter(x2[:, 0], x2[:, 1], c = [0.87,0.63,0.87], marker='*', label='异常数据1') 
plt.scatter(x1[:, 0], x1[:, 1], c = [0.82,0.41,0.12], marker='x', label='异常数据2') #huang 
font1 = {'weight' : 'normal',
'size'   : 13,
}
font2 = {'weight' : 'normal',
'size'   : 15,
}
plt.tick_params(labelsize=15)
plt.xlabel('风速/(m/s)',font2)  
plt.ylabel('功率/kW',font2)  
plt.xlim(0,25)  
plt.ylim(-100,1600)
#plt.legend(loc=6,prop=font1)  
plt.savefig('2.jpg',dpi=600)  
plt.show()   



np.savetxt('out1.txt', x2, fmt="%f", delimiter=',')
    #clf=clf.fit(X)
    #yy = clf.fit_predict(X) 
    #sheet11.write_row(i+1, yy)
    #for j in range(0,len(yy)):
     #   sheet11.write(j,i,float(yy[j])) #向第1行第1列写入获取到的值
 
    #y_pred=np.concatenate(y_pred,yy)
    
#workbook.save('xqtest.xls')
#f = open('label_pred','w')
#for x in label_pred:
 #   print(x);
 #   print(x,file=f);
#f.close()