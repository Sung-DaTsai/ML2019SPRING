import numpy as np
#import matplotlib.pyplot as plt
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
from keras.preprocessing.image import ImageDataGenerator
import keras


file = open(sys.argv[1], 'r', encoding='big5')
train_data = []

for i in file:
    split = i.split(",")
    train_data.append(split)
file.close()

train_data = np.array(train_data)
train_data = np.delete(train_data, 0, 0)

train_x = []
for i in range(train_data.shape[0]):
    a = train_data[i][1].split( )
    for j in range(2304):
        a[j] = int(a[j])
    train_x.append(a)

train_x = np.array(train_x)

train_x = np.reshape(train_x, (train_x.shape[0], 48, 48, 1))

train_y = np.zeros((train_x.shape[0], 7))
for i in range(train_y.shape[0]):
    train_y[i][int(train_data[i][0])] = 1

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.1,  # set range for random channel shifts
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

datagen.fit(train_x)




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



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy','accuracy'])
history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=100), steps_per_epoch=int(train_x.shape[0] / 100) , epochs=150, validation_data=(train_x[:6000,:,:,:], train_y[:6000,:]))

'''
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Training process_CNN")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend(['acc','val_acc'])
plt.savefig('Loss.png')
plt.show()
'''
temp_array = np.array(model.get_weights())
for i in range(len(temp_array)):
    temp_array[i] = temp_array[i].astype('float16')
np.savez_compressed('testing.npz', a=temp_array)

loaded = np.load('testing.npz')
new_weight = np.array(loaded['a'])
model.set_weights(new_weight)
