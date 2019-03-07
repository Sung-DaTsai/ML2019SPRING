import numpy as np
import csv
import sys

def feature_scaling(arr):
    for i in range(arr.shape[0]):
        arr[i] = (arr[i] - np.sum(arr[0])/arr.shape[1])/np.std(arr[0])
    return arr


w = np.load('weight.npy')
file2 = open(sys.argv[1], 'r', encoding='big5')
test_data = []

for i in file2:
    split = i.split(",")
    test_data.append(split)
file2.close()

test_data = np.array(test_data)
test_data = np.delete(test_data, [0,1], 1)

test_temp = test_data[0:18]
for i in range(1, 240):
    test_temp = np.concatenate((test_temp, test_data[18*i:18*(i+1)]), axis=1)

test_data2 = np.zeros((test_temp.shape[0], test_temp.shape[1]))

for i in range(test_temp.shape[0]):
    for j in range(test_temp.shape[1]):
        if((test_temp[i][j] != "NR\n")&(test_temp[i][j] != "NR")):
            test_data2[i][j] = float(test_temp[i][j])


test_data2 = feature_scaling(np.array(test_data2))

test_x = np.reshape(test_data2[:, 0:9], (162, 1))
for i in range(1, 240):
    test_x_temp = np.reshape(test_data2[:, 9*i:9*(i+1)], (162, 1))
    test_x = np.concatenate((test_x, test_x_temp), axis = 1)    

test_x = np.transpose(test_x)

test_x = np.concatenate((test_x, np.ones((test_x.shape[0],1))), axis=1)


pre = []
for i in range(len(test_x)):
    pre.append(["id_"+str(i)])
    a = np.dot(w, test_x[i])
    pre[i].append(a)
    
    
predict = open(sys.argv[2], "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()