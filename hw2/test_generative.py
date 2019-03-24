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


w = np.load('generative_weight.npy')
bias = np.load('generative_bias.npy')

file3 = open(sys.argv[3], 'r', encoding='big5')
test_xdata = []

for i in file3:
    split = i.split(",")
    test_xdata.append(split)
file3.close()

test_xdata = np.array(test_xdata[1:])

test_x = []

for i in range(test_xdata.shape[0]):
    test_x.append([])

for i in range(test_xdata.shape[0]):
    for j in range(test_xdata.shape[1]):
        test_x[i].append(float(test_xdata[i][j]))

test_x = minmax_normalization(np.array(test_x))


pre = []
for i in range(len(test_x)):
    pre.append([str(i+1)])
    a = sigmoid(np.dot(w, test_x[i])+bias)
    if(a>=0.5):
        a = 1
    else:
        a = 0
    pre[i].append(a)
    
    
predict = open(sys.argv[4], "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()
