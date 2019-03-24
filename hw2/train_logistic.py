import numpy as np
import csv
import sys

def minmax_normalization(arr):
    arr = np.transpose(arr)
    for i in (0,1,3,4,5):
        arr[i] = (arr[i]-min(arr[i]))/ (max(arr[i])-min(arr[i]))
    arr = np.transpose(arr)
    return arr


file = open(sys.argv[1], 'r', encoding='big5')
file2 = open(sys.argv[2], 'r', encoding='big5')
train_xdata = []
train_ydata = []

for i in file:
    split = i.split(",")
    train_xdata.append(split)

for i in file2:
    split = i.split(",")
    train_ydata.append(i)

file.close()
file2.close()

train_xdata = np.array(train_xdata[1:])
train_ydata = np.array(train_ydata[1:])


train_x = []
train_y = []

for i in range(train_xdata.shape[0]):
    train_x.append([])

for i in range(train_xdata.shape[0]):
    train_y.append(int(train_ydata[i]))
    for j in range(train_xdata.shape[1]):
        train_x[i].append(float(train_xdata[i][j]))

train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = minmax_normalization(train_x)
w = np.zeros(train_x.shape[1])
w_lr = 0.5
b_lr = 0.5
epochs = 35000
lamda = 0
bias = 0

x_t = np.transpose(train_x)
sum_gra_w = np.zeros(train_x.shape[1])
sum_gra_b = 0


cost = []

for i in range(epochs):
    temp_y = 1/(1 + np.exp(-np.dot(train_x, w) - bias))
    loss = temp_y - train_y
    w_gra = 2* np.dot(x_t, loss)
    b_gra = np.mean(2* loss)
    sum_gra_w += w_gra**2
    ada_w = np.sqrt(sum_gra_w)
    w = w - w_lr * w_gra/ada_w
    
    sum_gra_b += b_gra**2
    ada_b = np.sqrt(sum_gra_b)
    bias = bias - b_lr * b_gra/ada_b
    cost = -(np.dot(train_y, np.log(temp_y))+np.dot((1-train_y), (np.log(1-temp_y))))
    
    if(i % 1000==0):
        print (i, " times cost=", cost)



np.save('weight.npy',w)
np.save('bias.npy',bias)