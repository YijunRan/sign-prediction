# -*- coding: utf-8 -*-
"""
Created on Fri May 22 01:27:12 2020

@author: Yuan
"""
# %matplotlib inline
import networkx as nx
from xgboost import XGBClassifier
import numpy as np
import seaborn as sns
from PIL import Image
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold  #里面含有t-sne
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import math
import os
import warnings
warnings.filterwarnings("ignore")




'''

def role_value_edge_ebayes(df,predictor):   
    print ('边角色')
    para=1
    list2 = []
    for xx in df[predictor+'_role']:
        xx = eval(xx)
        role = [math.log((para*xx[i][0]+(1-para)*xx[i][1]),10) for i in xx]
        list2.append(sum(role)) 
        
        
    motif_role = [i[0]+i[1] for i in zip(list(df[predictor]),list2)]    
        
    df[predictor+'+edge_role'] =  motif_role

    return df



def role_value_node_ebayes(df,predictor): 
    print ('点角色')
    para=0
    list2 = []
    for xx in df[predictor+'_role']:
        xx = eval(xx)
        role = [math.log((para*xx[i][0]+(1-para)*xx[i][1]),10) for i in xx]
        list2.append(sum(role)) 
        
        
    motif_role = [i[0]+i[1] for i in zip(list(df[predictor]),list2)]    
        
    df[predictor+'+node_role'] =  motif_role

    return df

'''


def role_value_bayes(df,predictor):   
    print ('朴素贝叶斯')
    list2 = []
    for xx in df[predictor+'_role']:
        xx = eval(xx)
        role = [math.log((xx[i][0]+xx[i][2]+1)/(xx[i][1]+xx[i][3]+1),10) for i in xx]     
        #对每个共同邻居节点，计算其角色（（共用边的闭合模体+共用点的闭合模体+1）/(共用边的不闭合模体+共用点的不闭合模体+1)），并取对数
        list2.append(sum(role))   
        
        
    motif_role = [i[0] + i[1] for i in zip(list(df[predictor]),list2)]    
        
    df[predictor+'_bayes'] =  motif_role

    return df




def role_value_ebayes_two_part(df,predictor):  
    print ('模体数，点角色，边角色三维特征')
    list2 = []
    list3 = []
    for xx in df[predictor+'_role']:
        xx = eval(xx)
        # print (xx)
        role_edge = [math.log((xx[i][0]+1)/(xx[i][1]+1),10) for i in xx]
        role_node = [math.log((xx[i][2]+1)/(xx[i][1]+3),10) for i in xx] 

        
        list2.append(sum(role_edge)) 
        list3.append(sum(role_node))
        
    df[predictor+'_role_edge'] =  list2
    df[predictor+'_role_node'] =  list3

    return df





method = 'motif'

predictor_list = ['S'+str(i) for i in range(1,10)]
    

all_predictor_auc_ave = []
all_predictor_pre_ave = []
all_predictor_acc_ave = []

#all_predictor_auc_std = []
#all_predictor_pre_std = []
#all_predictor_acc_std = []


for predictor in predictor_list:
    

    
    
    
    print(predictor)
    
    
    
    ave_auc_motif = []
    ave_precision_motif = []
    ave_accuracy_motif = []
    
    
    for num in range(1,5):
        print (num)
        data_test = pd.read_csv('feature/bitcoinalpha/train0.8/bayes_20201007/bitcoinalpha_test0.2_'+str(num)+'.csv')
        data_train = pd.read_csv('feature/bitcoinalpha/train0.8/bayes_20201007/bitcoinalpha_train0.8_'+str(num)+'.csv')
        
        # data_test = role_value_bayes(data_test,predictor)
        # data_train = role_value_bayes(data_train,predictor)
        
        
        if method == 'motif':
            predictor_pre = [predictor]
            print (predictor)
        if method == 'bayes':
            predictor_pre  = [predictor+'_bayes']   #贝叶斯
            data_test = globals().get('role_value_bayes')(data_test,predictor)
            data_train = globals().get('role_value_bayes')(data_train,predictor)
        elif method == 'extend_bayes':
            predictor_pre = [predictor,predictor+'_role_edge',predictor+'_role_node']  #模体数，点角色，边角色三维特征
            data_test = globals().get('role_value_ebayes_two_part')(data_test,predictor)
            data_train = globals().get('role_value_ebayes_two_part')(data_train,predictor)
    
    
    
        x_test,y_test = data_test[predictor_pre],data_test['label']
        x_train,y_train = data_train[predictor_pre],data_train['label']
         
        
        
        clf = XGBClassifier(
            learning_rate =0.2, #默认0.3
            n_estimators=1000, #树的个数
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic', #逻辑回归损失函数
            nthread=4,  #cpu线程数
            eval_metric = 'auc',
            scale_pos_weight=1,
            seed =27)  #随机种子
        clf.fit(x_train, y_train)
        y_pre = clf.predict(x_test)
        y_pro = clf.predict_proba(x_test)[:,1]
    

    
        ave_auc_motif.append(roc_auc_score(y_test,y_pro))
        ave_precision_motif.append(precision_score(y_test,y_pre))
        ave_accuracy_motif.append(accuracy_score(y_test,y_pre))
        
        print (roc_auc_score(y_test,y_pro))
        print (accuracy_score(y_test,y_pre))
    
    
    all_predictor_auc_ave.append(np.mean(ave_auc_motif))    
    all_predictor_pre_ave.append(np.mean(ave_precision_motif))
    all_predictor_acc_ave.append(np.mean(ave_accuracy_motif))    
    
print (all_predictor_auc_ave)
print (all_predictor_pre_ave)
print (all_predictor_acc_ave)



'''

 
############################################################################
#############################################################################


def role_value_edge_ebayes_families(df):   
    print ('边角色')
    
    for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']:
    
    
        para=1
        list2 = []
        for xx in df[predictor+'_role']:
            xx = eval(xx)
            role = [math.log((para*xx[i][0]+(1-para)*xx[i][1]),10) for i in xx]
            list2.append(sum(role)) 
            
            
        motif_role = [i[0]+i[1] for i in zip(list(df[predictor]),list2)]    
            
        df[predictor+'+edge_role'] =  motif_role

    return df



def role_value_node_ebayes_families(df): 
    print ('点角色')
    
    for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']:
        para=0
        list2 = []
        for xx in df[predictor+'_role']:
            xx = eval(xx)
            role = [math.log((para*xx[i][0]+(1-para)*xx[i][1]),10) for i in xx]
            list2.append(sum(role)) 
            
            
        motif_role = [i[0]+i[1] for i in zip(list(df[predictor]),list2)]    
            
        df[predictor+'+node_role'] =  motif_role

    return df



def role_value_bayes_families(df):   
    print ('朴素贝叶斯')
    for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']:
        
        list2 = []
        for xx in df[predictor+'_role']:
            xx = eval(xx)
            role = [math.log((xx[i][0]+xx[i][1]),10) for i in xx]
            list2.append(sum(role)) 
            
        motif_role = [i[0]+i[1] for i in zip(list(df[predictor]),list2)]    
            
        df[predictor+'+edge_role+node_role'] =  motif_role

    return df



def role_value_ebayes_two_part_families(df):  
    print ('模体数，点角色，边角色三维特征')
    
    for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']:
        list2 = []
        list3 = []
        for xx in df[predictor+'_role']:
            xx = eval(xx)
            edge_role = 0
            node_role = 0
            
            for i in xx:
                edge_role += xx[i][0]
                node_role += xx[i][1]
            
            list2.append(edge_role) 
            list3.append(node_role)
            
        df[predictor+'_role_edge'] =  list2
        df[predictor+'_role_node'] =  list3
        

    return df





 
    
    
    
 
    

#predictor_list = [predictor+'+edge_role' for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']]
#predictor_list = ['S1','S2','S3','S4','S5','S6','S7','S8','S9']
#predictor_list = [predictor+'+node_role' for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']]
#predictor_list = [predictor+'+edge_role+node_role' for predictor in ['S1','S2','S3','S4','S5','S6','S7','S8','S9']]
predictor_list = [predictor+'_role_edge' for predictor in ['S_'+str(i) for i in range(1,17)]]+[predictor+'_role_node' for predictor in ['S_'+str(i) for i in range(1,17)]]+['S_'+str(i) for i in range(1,17)]


ave_auc_motif = []
ave_precision_motif = []
ave_accuracy_motif = []


for num in range(3):
    print (num)
    data_test = pd.read_csv('feature/bitcoinalpha/test/bitcoinalpha_test_'+str(num)+'.csv')
    data_train = pd.read_csv('feature/bitcoinalpha/train/bitcoinalpha_train_'+str(num)+'.csv')
    

 
    data_test = role_value_ebayes_two_part_families(data_test)
    data_train = role_value_ebayes_two_part_families(data_train)


    x_test,y_test = data_test[predictor_list],data_test['label']
    x_train,y_train = data_train[predictor_list],data_train['label']
    clf = XGBClassifier(
            learning_rate =0.2, #默认0.3
            n_estimators=1000, #树的个数
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic', #逻辑回归损失函数
            nthread=4,  #cpu线程数
            eval_metric = 'auc',
            scale_pos_weight=1,
            seed =27)  #随机种子
    clf.fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    y_pro = clf.predict_proba(x_test)[:,1]
    

    
    ave_auc_motif.append(roc_auc_score(y_test,y_pro))
    ave_precision_motif.append(precision_score(y_test,predict_result))
    ave_accuracy_motif.append(accuracy_score(y_test,y_pre))
    
    
    
    
    
print (np.mean(ave_auc_motif))
print (np.mean(ave_precision_motif))
print (np.mean(ave_accuracy_motif))
'''