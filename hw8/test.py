import numpy as np
import csv
import sys
from keras.models import Sequential 
from keras.layers import Flatten, Dropout
from keras.layers import Dense, DepthwiseConv2D 
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import keras



model = Sequential()  
model.add(Conv2D(8, kernel_size = (3, 3), strides=(1, 1), padding='same', input_shape = (48, 48, 1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# depthwise-1
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-1
model.add(Conv2D(16, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# depthwise-2
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-2
model.add(Conv2D(16, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# depthwise-3
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-3
model.add(Conv2D(32, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# depthwise-4
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-4
model.add(Conv2D(32, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))


# depthwise-5
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-5
model.add(Conv2D(64, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# depthwise-6
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-6
model.add(Conv2D(64, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))


# depthwise-7
model.add(DepthwiseConv2D(kernel_size = (3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-7
model.add(Conv2D(128, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# depthwise-8
model.add(DepthwiseConv2D(kernel_size = (2, 2), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

# pointwise-8
model.add(Conv2D(128, kernel_size = (1, 1), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding='valid'))
model.add(Dropout(0.2))


model.add(Flatten())


model.add(Dense(units = 64))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(units = 7, activation = 'softmax'))
model.summary()

'''
#model.load_weights('mobilenet_weights.h5')

temp_array = np.array(model.get_weights())

# weight quantization: dtype from float32->float16
for i in range(len(temp_array)):
    temp_array[i] = temp_array[i].astype('float16')
np.savez_compressed('testing.npz', a=temp_array)
'''

loaded = np.load('testing.npz', allow_pickle = True)
new_weight = np.array(loaded['a'])
model.set_weights(new_weight)

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
