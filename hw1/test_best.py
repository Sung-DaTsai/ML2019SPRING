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

        

w = np.load('weight_best.npy')
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
            if(i%18==10):
                test_data2[i][j] *= 10


test_data2 = np.concatenate((test_data2, np.reshape((np.cos(test_data2[14]*np.pi/180)), (1, 2160)), np.reshape((np.sin(test_data2[14]*np.pi/180)), (1, 2160))))
test_data2 = np.concatenate((test_data2, np.reshape((np.cos(test_data2[15]*np.pi/180)), (1, 2160)), np.reshape((np.sin(test_data2[15]*np.pi/180)), (1, 2160))))

test_data2 = np.delete(test_data2, [14,15], 0)


test_data2 = feature_scaling(np.array(test_data2))

test_x = np.reshape(test_data2[:, 0:9], (180, 1))
for i in range(1, 240):
    test_x_temp = np.reshape(test_data2[:, 9*i:9*(i+1)], (180, 1))
    test_x = np.concatenate((test_x, test_x_temp), axis = 1)     

test_x = np.transpose(test_x)

test_x = np.concatenate((test_x, np.ones((test_x.shape[0],1))), axis=1)


pre = []
for i in range(len(test_x)):
    pre.append(["id_"+str(i)])
    a = np.dot(w, test_x[i])
    pre[i].append(abs(a))
    
    
predict = open(sys.argv[2], "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()
