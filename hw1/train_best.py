# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:13:44 2019

@author: Mike
"""

import numpy as np
import csv
import sys

def feature_scaling(arr):
    for i in range(arr.shape[0]):
        arr[i] = (arr[i] - np.sum(arr[0])/arr.shape[1])/np.std(arr[0])
    return arr

def validation(train_x, train_y):
    divide = 10
    valid = np.random.randint(train_x.shape[0], size = train_x.shape[0]//divide)
    valid = sorted(valid)
    valid.reverse()
    valid_x = train_x[valid[0]:valid[0]+1, :]
    valid_y = train_y[valid[0]:valid[0]+1]
    train_x = np.delete(train_x, valid[0], 0)
    train_y = np.delete(train_y, valid[0], 0)
    for i in range(1, train_x.shape[0]//divide):
        valid_x = np.concatenate((valid_x, train_x[valid[i]:valid[i]+1, :]), axis=0)
        valid_y = np.concatenate((valid_y, train_y[valid[i]:valid[i]+1]), axis=0)
        train_x = np.delete(train_x, valid[i], 0)
        train_y = np.delete(train_y, valid[i], 0)
    return train_x, train_y, valid_x, valid_y 
        
def validation_cost(w, valid_x, valid_y):
    a = []
    for i in range(len(valid_x)):
        a.append(np.dot(w, valid_x[i]))
    return np.sqrt(np.sum((a - valid_y)**2)/len(valid_x))

        
file = open('train.csv', 'r', encoding='big5')
train_data = []


for i in file:
    split = i.split(",")
    train_data.append(split)
file.close()


train_data = np.array(train_data)
train_data = np.delete(train_data, 0, 0)
train_data = np.delete(train_data, [0,1,2], 1)


train = []
for i in range(18):
    train.append([])
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        if(i%18!=10):
            train[i%18].append(float(train_data[i][j]))
        elif((train_data[i][j] != "NR\n")&(train_data[i][j] != "NR")):
            train[i%18].append(10*float(train_data[i][j]))
        else:
            train[i%18].append(0)

train = np.array(train)

train = np.concatenate((train, np.reshape((np.cos(train[14]*np.pi/180)), (1, 5760)), np.reshape((np.sin(train[14]*np.pi/180)), (1, 5760))))
train = np.concatenate((train, np.reshape((np.cos(train[15]*np.pi/180)), (1, 5760)), np.reshape((np.sin(train[15]*np.pi/180)), (1, 5760))))

train = np.delete(train, [14,15], 0)


temp = train
train = feature_scaling(np.array(train))


train_x = []
train_y = []

for month in range(12):
    for hour in range(480-9):
        train_x.append([])
        train_y.append(temp[9][480*month+hour+9])
        for feature in range(20):
            for hs in range(9):
                train_x[471*month+hour].append(train[feature][480*month+hour+hs])
train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = np.concatenate((train_x, np.ones((train_x.shape[0],1))), axis=1)


train_x, train_y, valid_x, valid_y = validation(train_x, train_y)

w = np.zeros(train_x.shape[1])
w_lr = 0.5
epochs = 80000
lamda = 1
w[89] = 1

x_t = np.transpose(train_x)
sum_gra = np.zeros(train_x.shape[1])

val_cost = []
cost = []

for i in range(epochs):
    temp_y = np.dot(train_x, w)
    loss = temp_y - train_y
    gra = 2* np.dot(x_t,loss) + 2 * lamda * w
    sum_gra += gra**2
    ada = np.sqrt(sum_gra)
    w = w - w_lr * gra/ada
    cost.append(np.sqrt((np.sum(loss**2) + lamda * np.sum(np.square(w)))/ len(train_x)))
    val_cost.append(validation_cost(w, valid_x, valid_y))
    
    if(i % 1000==0):
        print (i, " times cost=", cost[i])
        print (i, " times val_cost=", val_cost[i])
    # early stopping
    if(i>1000):
        if(val_cost[i]>val_cost[i-1000]):
            break

np.save('weight_best.npy',w)

