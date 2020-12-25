# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:40:22 2020

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:03:46 2020

@author: asus
"""

import torch
from torchvision import datasets, transforms
import pickle
# 下载训练集
train_dataset = datasets.MNIST(root='F:\git\graduation_project',
                train=True,
                transform=transforms.ToTensor(),
                download=True)
# 下载测试集
test_dataset = datasets.MNIST(root='F:\git\graduation_project',
               train=False,
               transform=transforms.ToTensor(),
               download=True)
# 装载训练集
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                      batch_size=64,
#                      shuffle=True)
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=1,
                     shuffle=True)

data = []
file_name = []
for i in range(10):
    data.append('data'+str(i))
    file_name.append('data'+str(i)+'.plk')
    data[i] = []

while True:
    try:
        image, label = next(iter(test_loader))
        for i in range(10):
            if label == i:
                data[i].append(image)

    except StopIteration:
        break;

for i in range(9):  
    with open(file_name[i],'wb') as i:
        pickle.dump(data[i], i)
        pickle.dump(data[i+1], i)
        
with open(file_name[9],'wb') as f9:
        pickle.dump(data[9], f9)
        pickle.dump(data[0], f9)
