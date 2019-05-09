import numpy as np
from keras.preprocessing.sequence import pad_sequences
import sys
from keras.models import load_model
import csv
import jieba
from gensim.models import word2vec


model_2 = load_model('my_model_old2.h5')
model_3 = load_model('my_model_old3.h5')
model_4 = load_model('my_model_old5.h5')


word2vec_model = word2vec.Word2Vec.load("word2vec.model")
print("models are loaded")
jieba.load_userdict(sys.argv[2])


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



file = open(sys.argv[1], 'r', encoding='utf8')
test_xdata = []

for i in file:
    split = i.split(",")[1]
    split = list(jieba.cut(split))
    test_xdata.append(split[:-1])


file.close()
test_xdata = test_xdata[1:]

test_x = []

for sentence in test_xdata:
    embedding = []
    for word in sentence:
        try:
            embedding.append(dictionary[word])
        except:
            embedding.append(0)
    test_x.append(embedding)

test_x = np.array(test_x)

padding_length = 200
test_x = pad_sequences(test_x, maxlen=padding_length, padding='post')


ans_2 = model_2.predict_classes(test_x)
ans_3 = model_3.predict_classes(test_x)
ans_4 = model_4.predict_classes(test_x)

ans = ans_2+ans_3+ans_4

pre = []
for i in range(len(test_x)):
    pre.append([str(i)])
    if(np.mean(ans[i])>=1.5):
        pre[i].append(int(1))
    else:
        pre[i].append(int(0))


predict = open(sys.argv[3], "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()
