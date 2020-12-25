haimport torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
import pickle
# 下载训练集
train_dataset = datasets.MNIST(root='F:\毕设\exp0',
                train=True,
                transform=transforms.ToTensor(),
                download=True)
# 下载测试集
test_dataset = datasets.MNIST(root='F:\毕设\exp0',
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
data0 = []
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []
while True:
    try:
        image, label = next(iter(test_loader))
        if label == 0:
            data0.append(image)     
        if label == 1:
            data1.append(image)           
        if label == 2:
            data2.append(image) 
        if label == 3:
            data3.append(image) 
        if label == 4:
            data4.append(image) 
        if label == 5:
            data5.append(image) 
        if label == 6:
            data6.append(image) 
        if label == 7:
            data7.append(image) 
        if label == 8:
            data8.append(image) 
        if label == 9:
            data9.append(image) 
    except StopIteration:
        break;
        
with open('data0.plk','wb') as f0:
    pickle.dump(data0, f0)
with open('data1.plk','wb') as f1:
    pickle.dump(data1, f1)
with open('data2.plk','wb') as f2:
    pickle.dump(data2, f2)
with open('data3.plk','wb') as f3:
    pickle.dump(data3, f3)
with open('data4.plk','wb') as f4:
    pickle.dump(data4, f4)
with open('data5.plk','wb') as f5:
    pickle.dump(data5, f5)
with open('data6.plk','wb') as f6:
    pickle.dump(data6, f6)
with open('data7.plk','wb') as f7:
    pickle.dump(data7, f7)
with open('data8.plk','wb') as f8:
    pickle.dump(data8, f8)
with open('data9.plk','wb') as f9:
    pickle.dump(data9, f9)
