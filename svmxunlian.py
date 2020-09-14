# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:25:02 2020

@author: sc
"""

def svmxunlian(data,length):
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions
    from sklearn.decomposition import PCA
    #数据预处理
    data=sampledata
    sampledata[5:,-1]=1
    x,y=np.split(data,indices_or_sections=(-1,),axis=1) #x为数据，y为标签
    
    #plt.scatter(range(train_data.shape[1]))
    y1=[int(c[0]) for c in y]
    y1=np.array(y1)
    train_data,test_data,train_label,test_label =train_test_split(x,y1, random_state=0,stratify=y1,train_size=0.7,test_size=0.3) #sklearn.model_selection.
    
    #管道
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1,C=5,gamma=0.1,kernel='rbf',decision_function_shape='ovr'))
    pipe_svc.fit(train_data,train_label)
    tes_label=pipe_svc.predict(test_data)
    print("训练集：",pipe_svc.score(train_data,train_label))
    print("测试集：",pipe_svc.score(test_data,test_label))
    #网格定最优参数
    from sklearn.model_selection import GridSearchCV
    param_range=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]
    gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=4,n_jobs=-1)
    gs=gs.fit(train_data,train_label)
    print(gs.best_score_)
    print(gs.best_params_)
    #验证曲线定参数gamma、C
    from sklearn.model_selection import validation_curve
    param_range=np.arange(0.1,50.1,5)
    train_scores,test_scores=validation_curve(estimator=pipe_svc,X=train_data,y=train_label,param_name='svc__C',param_range=param_range,cv=4, n_jobs=-1)
    train_mean=np.mean(train_scores,axis=1)
    train_std=np.std(train_scores,axis=1)
    test_mean=np.mean(test_scores,axis=1)
    test_std=np.std(test_scores,axis=1)
    plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
    plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
    plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
    plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    #plt.xlim([0.01,10])
    plt.show()
    
    #print(train_data.shape)
    stdsc=StandardScaler()    
    train_data=stdsc.fit_transform(train_data)
    test_data=stdsc.transform(test_data)
    #随机森林
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=500,random_state=1)
    forest.fit(train_data,train_label)
    importances=forest.feature_importances_
    indices=np.argsort(importances)[::-1]
    #for i in range(train_data.shape[1]):
    zhibiao=['Hf','Hwp','Vpp','sigma','Vc','Vmax']
    plt.bar(range(train_data.shape[1]),importances[indices],align='center')
    plt.xticks(indices,zhibiao,rotation=90)
    
    #3.训练svm分类器
    classifier=svm.SVC(C=0.0001,kernel='rbf',gamma=0.1,decision_function_shape='ovr') # ovr:一对多策略
    classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
    
    #4.计算svc分类器的准确率
    print("训练集：",classifier.score(train_data,train_label))
    print("测试集：",classifier.score(test_data,test_label))
    
    #也可直接调用accuracy_score方法计算准确率
    #from sklearn.metrics import accuracy_score
    tra_label=pipe_svc.predict(train_data) #训练集的预测标签
    tes_label=pipe_svc.predict(test_data) #测试集的预测标签
    print("训练集真实：", test_label.flatten('A').tolist())
    print("训练集预测：", tes_label)
    #print("测试集：", accuracy_score(test_label,tes_label) )
    
    #查看决策函数
    print('train_decision_function:\n',classifier.decision_function(train_data)) # (90,3)
    #value=1.5;width=0.75
    #plot_decision_regions(test_data, tes_label.ravel(), classifier,
    #          feature_index=[13,14],                        #these one will be plotted  
    #          filler_feature_values={0: value,1: value,2: value,3: value,4: value,5: value,6: value,7: value,8: value,9: value,10: value,11: value,12: value},  #these will be ignored
    #          filler_feature_ranges={0: width,1: width,2: width,3: width,4: width,5: width,6: width,7: width,8: width,9: width,10: width,11: width,12: width})
    pca = PCA(n_components = 2)
    test_data2 = pca.fit_transform(test_data)
    fangcha=pca.explained_variance_ratio_ #解释方差比
    cum_fangcha=np.cumsum(fangcha)
    plt.bar(range(1,len(fangcha)+1),fangcha,alpha=0.5,align='center',label='individual explained variance')
    plt.step(range(1,len(fangcha)+1),cum_fangcha,where='mid',label='cumulative explained variance')
    #classifier.fit(train_data2, train_label.ravel())
    #train_label = np.array([[int(c[0])] for c in train_label])
    #plot_decision_regions(train_data2, train_label.ravel(),classifier, legend=2)