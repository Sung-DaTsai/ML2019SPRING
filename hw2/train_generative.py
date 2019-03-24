import numpy as np
import csv
import sys

def minmax_normalization(arr):
    arr = np.transpose(arr)
    for i in (0,1,3,4,5):
        arr[i] = (arr[i]-min(arr[i]))/ (max(arr[i])-min(arr[i]))
    arr = np.transpose(arr)
    return arr


def sigmoid(z):
    ans = 1/(1+np.exp(-z))
    return np.clip(ans, 0.00000000000001, 0.99999999999999)


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


mean_x0 = np.zeros((train_x.shape[1]))
mean_x1 = np.zeros((train_x.shape[1]))
count_x0 = 0
count_x1 = 0

for i in range(train_x.shape[0]):
    if(train_y[i] == 0):
        count_x0 += 1
        mean_x0 += train_x[i]
    else:
        count_x1 += 1
        mean_x1 += train_x[i]

mean_x0 /= count_x0
mean_x1 /= count_x1

sigma_x0 = np.zeros((train_x.shape[1], train_x.shape[1]))
sigma_x1 = np.zeros((train_x.shape[1], train_x.shape[1]))
for i in range(train_x.shape[0]):
    if(train_y[i] == 0):
        sigma_x0 += np.dot(np.reshape(train_x[i]-mean_x0, (train_x.shape[1], 1)), np.reshape(np.transpose(train_x[i]-mean_x0), (1, train_x.shape[1])))
    else:
        sigma_x1 += np.dot(np.reshape(train_x[i]-mean_x1, (train_x.shape[1], 1)), np.reshape(np.transpose(train_x[i]-mean_x1), (1, train_x.shape[1])))

sigma_x0 /= count_x0
sigma_x1 /= count_x1
sigma = sigma_x0 * count_x0/train_x.shape[0] + sigma_x1 * count_x1/train_x.shape[0]

sigma_inv = np.linalg.inv(sigma)
w = np.dot(np.transpose(mean_x1 - mean_x0), sigma_inv)
b = -0.5 *(np.dot(np.dot(np.transpose(mean_x1), sigma_inv), mean_x1)) + 0.5*(np.dot(np.dot(np.transpose(mean_x0), sigma_inv), mean_x0))
b += np.log(count_x1/count_x0)

np.save("generative_weight.npy", w)
np.save("generative_bias.npy", b)
