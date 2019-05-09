import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional, Flatten
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
import csv

'''
jieba.load_userdict('dict.txt.big')
word2vec_model = word2vec.Word2Vec.load("word2vec_dnn.model")

#embedding layer
embedding_matrix = np.zeros((len(word2vec_model.wv.vocab.items()) + 1, word2vec_model.vector_size))
dictionary = {}

vocab = []
for word, _ in word2vec_model.wv.vocab.items():
    vocab.append((word, word2vec_model.wv[word]))

# build dictionary
for i, info in enumerate(vocab):
    word, vec = info
    dictionary[word] = i + 1
    embedding_matrix[i + 1] = vec


file = open('train_x.csv', 'r', encoding='utf8')
file2 = open('train_y.csv', 'r', encoding='big5').read().split('\n')[1:-1]
train_xdata = []
train_ydata = []

for i in file:
    split = i.split(",")[1]
    split = list(jieba.cut(split))
    train_xdata.append(split[:-1])  # delete "\n"

for i in file2:
    train_ydata.append(int(i.split(",")[1]))


file.close()
train_xdata = train_xdata[1:]
'''
'''

train_x = np.zeros((120000,10595))
for i in range(120000):
    for j in range(len(train_xdata[i])):
        try:
            train_x[i][dictionary[train_xdata[i][j]]] += 1
        except:
            train_x[i][0] += 1

#train_y = train_ydata[0:80000]

model = Sequential()
model.add(Dense(500, input_dim=10595))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])


history = model.fit(x=train_x, y=train_ydata, batch_size=200, epochs=5, validation_split=0.1)


plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend(['train','validation'])
plt.savefig('Loss.png')
plt.show()


model.save('my_model_dnn.h5')
#np.save("mean_std.npy", np.array(mean,std))
del model

model = load_model('my_model_dnn.h5')
'''



file = open('test_x.csv', 'r', encoding='utf8')
test_xdata = []

for i in file:
    split = i.split(",")[1]
    split = list(jieba.cut(split))
    test_xdata.append(split[:-1])


file.close()
test_xdata = test_xdata[1:]

test_x = np.zeros((20000,10595))
for i in range(20000):
    for j in range(len(test_xdata[i])):
        try:
            test_x[i][dictionary[test_xdata[i][j]]] += 1
        except:
            test_x[i][0] += 1


ans = model.predict(test_x)

pre = []
for i in range(len(test_x)):
    pre.append([str(i)])
    if(ans[i]>=0.5):
        pre[i].append(int(1))
    else:
        pre[i].append(int(0))



file4 = "predict.csv"
predict = open(file4, "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()


