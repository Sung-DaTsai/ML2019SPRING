import jieba
import numpy as np
from gensim.models import word2vec


jieba.load_userdict('dict.txt.big')

file = open('train_x.csv', 'r', encoding='utf8')
file2 = open('train_y.csv', 'r', encoding='big5').read().split('\n')[1:-1]
train_xdata = []
train_ydata = []

for i in file:
    split = i.split(",")[1]
    split = list(jieba.cut(split))
    train_xdata.append(split[:-1])

for i in file2:
    train_ydata.append(int(i.split(",")[1]))


file.close()
train_xdata = train_xdata[1:]


model = word2vec.Word2Vec(train_xdata, size=250, window=5, min_count=5, workers=4, iter=10, sg=1)

model.save("word2vec.model")