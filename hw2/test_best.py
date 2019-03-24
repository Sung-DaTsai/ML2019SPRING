from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv 
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
import sys
import pickle

file = open(sys.argv[3], 'r', encoding='big5')
file2 = open(sys.argv[4], 'r', encoding='big5')
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


file = open(sys.argv[1], 'r', encoding='big5')
train_data = []

for i in file:
    split = i.split(",")
    train_data.append(split)
file.close()
train_data = np.array(train_data)[1:,4:5]


file = open(sys.argv[2], 'r', encoding='big5')
test_data = []
for i in file:
    split = i.split(",")
    test_data.append(split)
file.close()
test_data = np.array(test_data)[1:,4:5]

train_temp = np.zeros((train_data.shape[0],1))
for i in range(train_data.shape[0]):
    train_temp[i] = float(train_data[i][0])
    
test_temp = np.zeros((test_data.shape[0],1))
for i in range(test_data.shape[0]):
    test_temp[i] = float(test_data[i][0])


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


file3 = open(sys.argv[5], 'r', encoding='big5')
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

test_x = np.array(test_x)
train_x = np.concatenate((train_x, train_temp), axis=1)
test_x = np.concatenate((test_x, test_temp), axis=1)
sc = MinMaxScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

'''
gbc = GradientBoostingClassifier()
gbc.fit(train_x, train_y)
joblib.dump(gbc, 'model_gbc.pkl')
'''


gbc = joblib.load('model_gbc.pkl') 


a = gbc.predict(test_x) 
pre = []
for i in range(len(test_x)):
    pre.append([str(i+1)])
    pre[i].append(int(a[i]))



predict = open(sys.argv[6], "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()

