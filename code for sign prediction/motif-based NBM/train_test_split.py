# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:17:52 2020

@author: Yuan
"""

import random
import copy
import networkx as nx
import pandas as pd
import itertools
import math
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from minepy import MINE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier



def loadfile():
    with open(filename) as f:
        edges = []
        for each_line in f:
            n1,n2,w = each_line.strip('\t\r\n').split(' ')      #N46edge以\t分隔，其它五个网络以空格分隔
            edges.append((n1,n2,int(w)))
        return edges
                      
filename = 'data_undirected/wiki.txt'
G = nx.Graph()
G.add_weighted_edges_from(loadfile())



for kk in range(5):
#    print (kk)
    positive_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 1]  #所有正边
    negative_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 2]   #所有负边
    
    negative_edge = random.sample(negative_edge,1000)    #抽取部分负边
    positive_edge = random.sample(positive_edge,1000)    #抽取部分正边
    

    train_G = copy.deepcopy(G)
    print ('网络中的总边数',len(train_G.edges()))

    negative_test = random.sample(negative_edge,int(len(negative_edge)*0.1))    #选取20%负样本作为测试边
    
    #######################################################################
    #############################wiki网络的测试集在之前的基础上在抽取10%, slashdot网络3%
#    negative_test = random.sample(negative_test,int(len(negative_test)*0.5))    #
    ##############################################################
    #######################################################################
    print ('测试集的负边数',len(negative_test))
    
    positive_test = random.sample(positive_edge,len(negative_test))    #选取等量的正样本作为测试边
    print ('测试集的正边数',len(positive_test))
    
    
    positive_train = list(set(positive_edge).difference(set(positive_test)))     #剩下的正边作为训练集
    negative_train = list(set(negative_edge).difference(set(negative_test)))     #剩下的负边作为训练集
    
    print ('训练集的负边数',len(negative_train))
    print ('训练集的正边数',len(positive_train))
    
    

    train_G.remove_edges_from(negative_test)
    train_G.remove_edges_from(positive_test)
    
    


    positive_train_edges = [i for i in train_G.edges() if G[i[0]][i[1]]['weight'] == 1]  #训练网络中的所有正边
    negative_train_edges = [i for i in train_G.edges() if G[i[0]][i[1]]['weight'] == 2]   #训练网络中的所有负边，都用作训练集
    
    print ('训练网络中的正边',len(positive_train_edges))
    print ('训练网络中的负边',len(negative_train_edges))
    
    
    k = len(negative_train_edges)/len(positive_train_edges)

    
    #######################################################################
    #############################wiki网络的训练集在之前的基础上在抽取10%,slashdot网络3%
#    negative_train = random.sample(negative_train,int(len(negative_train)*0.5))    
    ################################################################################
    #######################################################################
    

#    positive_train = random.sample(positive_edge_in_train,len(negative_train))   #在训练网络中选取等量的正边作为训练集（为了样本平衡）
    


    train_list = positive_train + negative_train
    
    train_label = [1]*len(positive_train)+[0]*len(negative_train)
    
    test_list = positive_test + negative_test
    test_label = [1]*len(positive_test)+[0]*len(negative_test)
    
    
#    print (len(train_list),len(test_list))
    
    
    
    
    df_train = pd.DataFrame()
    df_train['edge'] = train_list
    df_train['label'] = train_label
    df_train.to_csv('train_set/wiki/train0.9_'+str(kk)+'.csv')
    
    df_test = pd.DataFrame()
    df_test['edge'] = test_list
    df_test['label'] = test_label
    df_test.to_csv('test_set/wiki/test0.1_'+str(kk)+'.csv')