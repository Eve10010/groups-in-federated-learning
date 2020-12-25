# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:35:28 2020

@author: asus
"""

import pickle
import numpy as np

def get_data_weight():  
    weight = np.zeros((10,10))
    file = []
    data = []
    for i in range(10):
        x = "data"+str(i)+".plk"
        file = open(x,"rb")
        data = pickle.load(file)
        weight[i][i] = len(data)
        file.close()
    return weight