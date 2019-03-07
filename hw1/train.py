import numpy as np
import csv
import sys

def feature_scaling(arr):
    for i in range(arr.shape[0]):
        arr[i] = (arr[i] - np.sum(arr[0])/arr.shape[1])/np.std(arr[0])
    return arr

# train data
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
        if((train_data[i][j] != "NR\n")&(train_data[i][j] != "NR")):
            train[i%18].append(float(train_data[i][j])) 
        else:
            train[i%18].append(0) 

temp = train
train = feature_scaling(np.array(train))


train_x = []
train_y = []

for month in range(12):
    for hour in range(480-9):
        train_x.append([])
        train_y.append(temp[9][480*month+hour+9])
        for feature in range(18):
            for hs in range(9):
                train_x[471*month+hour].append(train[feature][480*month+hour+hs])
train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = np.concatenate((train_x, np.ones((train_x.shape[0],1))), axis=1)


w = np.ones(train_x.shape[1])
w_lr = 1.7
epochs = 10000

x_t = np.transpose(train_x)
sum_gra = np.zeros(train_x.shape[1])


for i in range(epochs):
    temp_y = np.dot(train_x, w)
    loss = temp_y - train_y
    gra = 2* np.dot(x_t,loss)
    sum_gra += gra**2
    ada = np.sqrt(sum_gra)
    w = w - w_lr * gra/ada

np.save('weight.npy',w)
