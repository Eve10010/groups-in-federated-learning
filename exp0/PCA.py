        # -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:18:29 2020

@author: asus
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain


def takefirst(elem):
    return elem[0]
def PCA(x,k):
    #x is feature data,k is the components of you want.
    shape = x.shape #get the shape of x array
    mean=np.array([np.mean(X[:,i]) for i in range(shape[1])])
    #normalization
    norm_x=x-mean
    #get Covariance matrix
    C=np.dot(norm_x.T,norm_x)
    #Find eigenvalues and eigenvectors
    eigenvalue,featurevector=np.linalg.eig(C)
    eig_pairs = [(np.abs(eigenvalue[i]), featurevector[:,i]) for i in range(shape[1])]

    eig_pairs.sort(key=takefirst,reverse=True)

    P = np.array([ele[1] for ele in eig_pairs[:k]])
    data = np.dot(P,norm_x.T)

    return data

def group(X,categories):
    # X为特征矩阵，categories为需要分成类别数
    dem_reduction = PCA(X,2)
    print(dem_reduction)
    max_ = []
    min_ = []
    max_= max_+[ np.max(dem_reduction[0])] + [np.max(dem_reduction[1])] 
    min_= min_+[np.min(dem_reduction[0]) ]+ [np.min(dem_reduction[1])]
    for i in range(2):
        for j in range(len(dem_reduction[0])):
            print(dem_reduction[i][j],' ',min_[i],' ',max_[i],'\n')
            dem_reduction[i][j] = (dem_reduction[i][j]-min_[i])/(max_[i]-min_[i])
            
        
    mean_x = np.mean(dem_reduction[0])
    mean_y = np.mean(dem_reduction[1])

    plt.scatter(dem_reduction[0],dem_reduction[1])
    plt.scatter(mean_x,mean_y,marker='x')
    for i in range(len(dem_reduction[0])):
        plt.text(dem_reduction[0][i]*1.1, dem_reduction[1][i]*1.1,\
                i+1,fontsize=10, color = "r", style = "italic",\
                weight = "light",verticalalignment='center',\
                horizontalalignment='right',rotation=0) #给散点加标签

    #dids 为距离向量
    dis = []
    dis.append(((dem_reduction[0]-mean_x)**2+(dem_reduction[1]-mean_y)**2)**0.5)
    dis = list(chain.from_iterable(dis))
    
    
    radius = max(dis)/categories
    for temp in range(categories):
        Circle = plt.Circle((mean_x,mean_y),radius*(temp+1),alpha=0.25 )
        plt.gcf().gca().add_artist(Circle)
    plt.show()
    
    all_categories = []
    #开始分类
    for i in range(categories):
        index = 0  
        one_category = []
        for j in dis: 
            index = index + 1
            if radius * i <= j <= radius * (i+1) :
                one_category.append(index)
        all_categories.append(one_category)
    
    return all_categories
        
            
            
    
if __name__=='__main__':
    X = np.array([[8000,0,0,0,1200,0,0,0,0,0], 
                  [0,1000,0,9000,0,0,0,0,0,0],
                  [0,0,10000,0,10000,0,0,0,0,0], 
                  [0,0,0,10000,0,0,0,0,0,0], 
                  [0,0,0,0,10000,0,0,2200,0,0],
                  [0,0,0,0,0,10000,0,0,0,0],
                  [0,5200,0,0,0,0,10000,0,0,0],
                  [0,0,0,0,0,0,0,10000,0,0], 
                  [0,0,10000,0,0,0,0,0,10000,0],
                  [0,0,0,0,0,0,0,0,0,10000],
                  [0,10000,0,0,0,0,0,0,0,0]])
    re = group(X,5)
    print(re)
#    re = PCA(X,2)
#    x = np.linspace(1,10)
#    mean_x = np.mean(re[0])
#    mean_y = np.mean(re[1])
#    
#dis = []
#dis.append(((re[0]-mean_x)**2+(re[1]-mean_y)**2)**0.5)
#palette = np.array(sns.color_palette("hls", 10))
#plt.scatter(re[0],re[1],c=palette)
#plt.scatter(mean_x,mean_y,marker='x')
#
##circle
    