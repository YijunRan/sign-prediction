# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:16:52 2020

@author: Yuan
"""


import copy
import networkx as nx
import pandas as pd
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random




def S1(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    common = list(nx.common_neighbors(G,n1,n2))
    common = [i for i in common if G[n1][i]['weight']==1 and G[n2][i]['weight']==1]
    num = len(common)
    return num,common

def S2(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]
    # print (n1)
    # print (n2)
     
    
    neighbor_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 1] 
    print (len(neighbor_edge))
    
    
    neighbor_edge = [edge for edge in neighbor_edge if ( G.has_edge(n1,edge[0]) and G[n1][edge[0]]['weight']==1 and G.has_edge(n2,edge[1]) and G[n2][edge[1]]['weight']==1) \
    or  ( G.has_edge(n1,edge[1]) and G[n1][edge[1]]['weight']==1 and G.has_edge(n2,edge[0]) and G[n2][edge[0]]['weight']==1)]
    

    num = len(neighbor_edge)
    print (num)
            
    return num,neighbor_edge


def S3(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    neighbor_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 2] 
    print (len(neighbor_edge))
    

    neighbor_edge = [edge for edge in neighbor_edge if (G.has_edge(n1,edge[0]) and G[n1][edge[0]]['weight']==1 and G.has_edge(n2,edge[1]) and G[n2][edge[1]]['weight']==1) or \
    (G.has_edge(n1,edge[1]) and G[n1][edge[1]]['weight']==1 and G.has_edge(n2,edge[0]) and  G[n2][edge[0]]['weight']==1)]
    

    num = len(neighbor_edge)
    print (num)
            
    return num,neighbor_edge


def S4(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    common = list(nx.common_neighbors(G,n1,n2))
    common = [i for i in common if G[n1][i]['weight']==1 and G[n2][i]['weight']==2]
    num = len(common)
    return num,common


def S5(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    neighbor_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 1] 
    print (len(neighbor_edge))
    

    neighbor_edge = [edge for edge in neighbor_edge if ( G.has_edge(n1,edge[0]) and G[n1][edge[0]]['weight']==1 and G.has_edge(n2,edge[1]) and G[n2][edge[1]]['weight']==2) or \
    (G.has_edge(n1,edge[1]) and G[n1][edge[1]]['weight']==2 and G.has_edge(n2,edge[0]) and G[n2][edge[0]]['weight']==1)]
    

    num = len(neighbor_edge)
    print (num)
            
    return num,neighbor_edge

def S6(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    neighbor_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 2] 
    print (len(neighbor_edge))
    

    neighbor_edge = [edge for edge in neighbor_edge if ( G.has_edge(n1,edge[0]) and G[n1][edge[0]]['weight']==1 and G.has_edge(n2,edge[1]) and G[n2][edge[1]]['weight']==2) or \
    (G.has_edge(n1,edge[1]) and G[n1][edge[1]]['weight']==2 and G.has_edge(n2,edge[0]) and G[n2][edge[0]]['weight']==1)]
    

    num = len(neighbor_edge)
    print (num)
            
    return num,neighbor_edge


def S7(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    common = list(nx.common_neighbors(G,n1,n2))
    common = [i for i in common if G[n1][i]['weight']==2 and G[n2][i]['weight']==2]
    num = len(common)
    return num,common


def S8(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    neighbor_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 1] 
    print (len(neighbor_edge))
    

    neighbor_edge = [edge for edge in neighbor_edge if (G.has_edge(n1,edge[0]) and G[n1][edge[0]]['weight']==2 and G.has_edge(n2,edge[1]) and G[n2][edge[1]]['weight']==2) or \
    (G.has_edge(n1,edge[1]) and G[n1][edge[1]]['weight']==2 and G.has_edge(n2,edge[0]) and G[n2][edge[0]]['weight']==2)]
    

    num = len(neighbor_edge)
    print (num)
            
    return num,neighbor_edge


def S9(G,nodeij):
    
    n1 = nodeij[0]
    n2 = nodeij[1]

    neighbor_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 2] 
    print (len(neighbor_edge))
    
    neighbor_edge = [edge for edge in neighbor_edge if (G.has_edge(n1,edge[0]) and G[n1][edge[0]]['weight']==2 and G.has_edge(n2,edge[1]) and G[n2][edge[1]]['weight']==2) or \
    (G.has_edge(n1,edge[1]) and G[n1][edge[1]]['weight']==2 and G.has_edge(n2,edge[0]) and G[n2][edge[0]]['weight']==2)]
    

    num = len(neighbor_edge)
    print (num)
            
    return num,neighbor_edge




def role_S1(G,node,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    
    try:
        neighbor_list = list(nx.neighbors(G,node))
        neighbor_list = [i for i in neighbor_list if G[node][i]['weight']==1]
        

        turple_list = itertools.combinations(neighbor_list,2)
        
        for i in turple_list:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1            
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0    
    return count1_edge,count1_node,count2_edge,count2_node



    

def role_S4(G,node,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    
    try:
        neighbor_list = list(nx.neighbors(G,node))
        
        negative_neighbor = [i for i in neighbor_list if G[node][i]['weight']==2]
        positive_neighbor = [i for i in neighbor_list if G[node][i]['weight']==1]

        
        nbr1_nbr2 = itertools.product(negative_neighbor,positive_neighbor)

        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:

                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
                
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1           
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0    
    return count1_edge,count1_node,count2_edge,count2_node




def role_S7(G,node,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    
    try:

        neighbor_list = list(nx.neighbors(G,node))
        neighbor_list = [i for i in neighbor_list if G[node][i]['weight']==2]
        
        
        turple_list = itertools.combinations(neighbor_list,2)
        
        for i in turple_list:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1            
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0    
    return count1_edge,count1_node,count2_edge,count2_node



def role_S2(G,edge,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    try:
        node1 = edge[0]
        node2 = edge[1]
        
        neighbor_list1 = list(nx.neighbors(G,node1))
        neighbor_list2 = list(nx.neighbors(G,node2))
        
        neighbor_list1 = [i for i in neighbor_list1 if G[node1][i]['weight']==1]
        neighbor_list2 = [i for i in neighbor_list2 if G[node2][i]['weight']==1]
        
        nbr1_nbr2 = itertools.product(neighbor_list1,neighbor_list2)
        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1                
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0   
    return count1_edge,count1_node,count2_edge,count2_node





def role_S3(G,edge,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    try:
        node1 = edge[0]
        node2 = edge[1]
        
        neighbor_list1 = list(nx.neighbors(G,node1))
        neighbor_list2 = list(nx.neighbors(G,node2))
        
        neighbor_list1 = [i for i in neighbor_list1 if G[node1][i]['weight']==1]
        neighbor_list2 = [i for i in neighbor_list2 if G[node2][i]['weight']==1]
        
        nbr1_nbr2 = itertools.product(neighbor_list1,neighbor_list2)
        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1            
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0   
    return count1_edge,count1_node,count2_edge,count2_node




def role_S5(G,edge,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    try:
        node1 = edge[0]
        node2 = edge[1]
        
        neighbor_list1 = list(nx.neighbors(G,node1))
        neighbor_list2 = list(nx.neighbors(G,node2))
        
        neighbor_list1 = [i for i in neighbor_list1 if G[node1][i]['weight']==2]
        neighbor_list2 = [i for i in neighbor_list2 if G[node2][i]['weight']==1]
        
        nbr1_nbr2 = itertools.product(neighbor_list1,neighbor_list2)
        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1           
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0   
    return count1_edge,count1_node,count2_edge,count2_node 


def role_S6(G,edge,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    try:
        node1 = edge[0]
        node2 = edge[1]
        
        neighbor_list1 = list(nx.neighbors(G,node1))
        neighbor_list2 = list(nx.neighbors(G,node2))
        
        neighbor_list1 = [i for i in neighbor_list1 if G[node1][i]['weight']==2]
        neighbor_list2 = [i for i in neighbor_list2 if G[node2][i]['weight']==1]
        
        nbr1_nbr2 = itertools.product(neighbor_list1,neighbor_list2)
        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1               
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0   
    return count1_edge,count1_node,count2_edge,count2_node 



def role_S8(G,edge,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    try:
        node1 = edge[0]
        node2 = edge[1]
        
        neighbor_list1 = list(nx.neighbors(G,node1))
        neighbor_list2 = list(nx.neighbors(G,node2))
        
        neighbor_list1 = [i for i in neighbor_list1 if G[node1][i]['weight']==2]
        neighbor_list2 = [i for i in neighbor_list2 if G[node2][i]['weight']==2]
        
        nbr1_nbr2 = itertools.product(neighbor_list1,neighbor_list2)
        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1             
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0   
    return count1_edge,count1_node,count2_edge,count2_node 


def role_S9(G,edge,linkij):
    
    count1_edge = 0    #node和正边构成预测器的数量(共用边)
    count1_node = 0    #node和正边构成预测器的数量（共用节点）
    
    count2_edge = 0    #node和负边构成预测器的数量(共用边)
    count2_node = 0    #node和负边构成预测器的数量（共用节点）
    try:
        node1 = edge[0]
        node2 = edge[1]
        
        neighbor_list1 = list(nx.neighbors(G,node1))
        neighbor_list2 = list(nx.neighbors(G,node2))
        
        neighbor_list1 = [i for i in neighbor_list1 if G[node1][i]['weight']==2]
        neighbor_list2 = [i for i in neighbor_list2 if G[node2][i]['weight']==2]
        
        nbr1_nbr2 = itertools.product(neighbor_list1,neighbor_list2)
        
        for i in nbr1_nbr2:
            if i in G.edges() and G[i[0]][i[1]]['weight']==1:
                if linkij[0] in i or linkij[1] in i:
                    count1_edge +=1
                else:
                    count1_node +=1
            elif i in G.edges() and G[i[0]][i[1]]['weight']==2:
                if linkij[0] in i or linkij[1] in i:
                    count2_edge +=1
                else:
                    count2_node +=1             
    except:
        count1_edge = 0    
        count1_node = 0    
    
        count2_edge = 0   
        count2_node = 0   
    return count1_edge,count1_node,count2_edge,count2_node 




def loadfile():
    with open(filename) as f:
        edges = []
        for each_line in f:
            n1,n2,w = each_line.strip('\t\r\n').split(' ')      #N46edge以\t分隔，其它五个网络以空格分隔
            edges.append((n1,n2,int(w)))
        return edges
                      
filename = 'data_undirected/bitcoinalpha.txt'
G = nx.Graph()
G.add_weighted_edges_from(loadfile())


#predictor_list = ['S2']
predictor_list = ['S1','S2','S3','S4','S5','S6','S7','S8','S9']

#k = 0.1436853908001969


for kk in range(1,5):
    
    
    
    positive_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 1]  #所有正边
    negative_edge = [i for i in G.edges() if G[i[0]][i[1]]['weight'] == 2]   #所有负边
    
    negative_edge = random.sample(negative_edge,1000)    #抽取部分负边
    positive_edge = random.sample(positive_edge,1000)    #抽取部分正边
    

    train_G = copy.deepcopy(G)
    print ('网络中的总边数',len(train_G.edges()))

    negative_test = random.sample(negative_edge,int(len(negative_edge)*0.1))    #选取10%负样本作为测试边
    
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
    



    
    
    predictor_edge_value_test = []         #基于贝叶斯的特征
     
    for predictor in predictor_list:
        
        edge_motif = []
        bayes_role = []
        
        nn = 0
        for linkij in test_list: 
            nn += 1
            
            motif_num,edges  = globals().get(predictor)(train_G,linkij)
            print (edges,'test',predictor,nn)
            edge_motif.append(motif_num*math.log(k,10))
            

            role_value1 = {}
            for i in edges:
                count1_edge,count1_node,count2_edge,count2_node = globals().get('role_'+predictor)(train_G,i,linkij)

                # edge_role = (count1_edge+1.0)/(count2_edge+1.0)
                # node_role = (count1_node+1.0)/(count2_node+1.0)
                role_value1[i] = (count1_edge,count2_edge,count1_node,count2_node)
                
            bayes_role.append(role_value1)
#            
        predictor_edge_value_test.append((edge_motif,bayes_role))
    
            
            
    df_test = pd.DataFrame()
    df_test['edge'] = test_list

    for i in range(len(predictor_list)):
        for num in range(2):
            if num == 0:
                df_test[predictor_list[i]] = predictor_edge_value_test[i][num]
            if num == 1:
                df_test[predictor_list[i]+'_'+'role'] = predictor_edge_value_test[i][num]
                
                
        
    df_test['label'] = test_label
    df_test.to_csv('feature/bitcoinalpha/train0.9/bayes_20201007/bitcoinalpha_test0.1_'+str(kk)+'.csv',index = False)  
    



#    
    predictor_edge_value_train = []         #基于贝叶斯的特征
     
    for predictor in predictor_list:
        print (predictor)
        
        edge_motif = []
        bayes_role = []
        
        
        nn = 0
        for linkij in train_list: 
            nn += 1
            
            
#            print (linkij,'bayes_train')
            motif_num,edges  = globals().get(predictor)(train_G,linkij)
            print (edges,'train',predictor,nn)
            edge_motif.append(motif_num*math.log(k,10))
            

            role_value1 = {}
            for i in edges:
                count1_edge,count1_node,count2_edge,count2_node = globals().get('role_'+predictor)(train_G,i,linkij)
                # edge_role = (count1_edge+1.0)/(count2_edge+1.0)
                # node_role = (count1_node+1.0)/(count2_node+1.0)
                role_value1[i] =  (count1_edge,count2_edge,count1_node,count2_node)
                
            bayes_role.append(role_value1)
            
        predictor_edge_value_train.append((edge_motif,bayes_role))
#    
#            
#            
    df_train = pd.DataFrame()
    df_train['edge'] = train_list

    for i in range(len(predictor_list)):
        for num in range(2):
            if num == 0:
                df_train[predictor_list[i]] = predictor_edge_value_train[i][num]
            if num == 1:
                df_train[predictor_list[i]+'_'+'role'] = predictor_edge_value_train[i][num]
                
                
        
    df_train['label'] = train_label
    df_train.to_csv('feature/bitcoinalpha/train0.9/bayes_20201007/bitcoinalpha_train0.9_'+str(kk)+'.csv',index = False) 

#                
                
                     
                
                
                
                
                
                


        
        
        
        
        
        
        
        
        

