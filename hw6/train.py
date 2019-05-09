import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, Dropout, LSTM, GRU, Bidirectional, Flatten
import sys
from keras.models import load_model
import csv

jieba.load_userdict(sys.argv[4])
word2vec_model = word2vec.Word2Vec.load("word2vec.model")

# embedding layer
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


file = open(sys.argv[1], 'r', encoding='utf8')
file2 = open(sys.argv[2], 'r', encoding='big5').read().split('\n')[1:-1]
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

train_x = []
for sentence in train_xdata:
    embedding = []
    for word in sentence:
        try:
            embedding.append(dictionary[word])
        except:
            embedding.append(0)
    train_x.append(embedding)

train_x = np.array(train_x)

padding_length = 200
train_x = pad_sequences(train_x, maxlen=padding_length, padding='post')


train_y = np.zeros((120000,1,2))
for i in range(120000):
    if(train_ydata[i]==0):
        train_y[i][0][0] = 1
    else:
        train_y[i][0][1] = 1
        

model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                             weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(TimeDistributed(Dense(256, activation='relu')))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy', 'accuracy'])

history = model.fit(x=train_x, y=train_y, batch_size=100, epochs=5, validation_split=0.1)

'''
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend(['train','validation'])
plt.savefig('Loss.png')
plt.show()
'''

model.save('my_model.h5')
del model