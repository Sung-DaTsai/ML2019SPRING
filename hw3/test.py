import numpy as np
#import matplotlib.pyplot as plt
import csv
import sys
from keras.models import load_model


model = load_model('my_model.h5')
model_1 = load_model('my_model_old_1.h5')
model_2 = load_model('my_model_old_2.h5')



file = open(sys.argv[1], 'r', encoding='big5')
test_data = []

for i in file:
    split = i.split(",")
    test_data.append(split)
file.close()

test_data = np.array(test_data)
test_data = np.delete(test_data, 0, 0)

test_x = []
for i in range(test_data.shape[0]):
    a = test_data[i][1].split( )
    for j in range(2304):
        a[j] = int(a[j])
    test_x.append(a)

test_x = np.array(test_x)

test_x = np.reshape(test_x, (test_x.shape[0], 48, 48, 1))

ans = model.predict(test_x)
ans_1 = model_1.predict(test_x)
ans_2 = model_2.predict(test_x)

ans = ans + ans_1 + ans_2

pre = []
for i in range(len(test_x)):
    pre.append([str(i)])
    pre[i].append(np.argmax(ans[i]))



predict = open(sys.argv[2], "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()
