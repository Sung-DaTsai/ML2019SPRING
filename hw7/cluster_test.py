from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose
import numpy as np
from sklearn.cluster import KMeans
#from keras.preprocessing.image import ImageDataGenerator
from skimage import io
#import matplotlib.pyplot as plt
from keras.models import load_model
import csv
from sklearn.decomposition import PCA 
import sys

n_clusters = 2


def autoencoderConv2D_1(input_shape=(32, 32, 3), filters=[16, 32, 64, 10]):
    input_img = Input(shape=input_shape)
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)

    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)

    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    #x = BatchNormalization()(x)
    x = Conv2D(128, 3, strides=2, padding=pad3, activation='relu', name='conv4')(x)

    x = Flatten()(x)
    encoded = Dense(units=filters[3], name='embedding')(x)
    x = Dense(units=128*int(input_shape[0]/16)*int(input_shape[0]/16), activation='relu')(encoded)

    x = Reshape((int(input_shape[0]/16), int(input_shape[0]/16), 128))(x)
    x = Conv2DTranspose(filters[2], 3, strides=2, padding=pad3, activation='relu', name='deconv4')(x)

    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)

    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

    #x = BatchNormalization()(x)

    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1')(x)
    #x = BatchNormalization()(x)
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

autoencoder, encoder = autoencoderConv2D_1()
#autoencoder.summary()


train_x = np.zeros((40000,32,32,3))
for number in range(1, 40001):
    number_format = "%06d" % number
    img_path = sys.argv[1]+str(number_format)+'.jpg'
    img = io.imread(img_path)
    img = img.reshape((1,32,32,3))
    train_x[number-1] = img
    #train_x = np.concatenate((train_x, img), axis=0)
train_x = train_x/255




'''
np.save("train_x.npy",train_x)

train_x = np.load('train_x.npy')
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.1,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.1,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_x)




autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse','acc'])
#history = autoencoder.fit(train_x, train_x, batch_size=250, epochs=100, validation_split=0.1)
history = autoencoder.fit_generator(datagen.flow(train_x, train_x, batch_size=100), steps_per_epoch=int(train_x.shape[0] / 100) , epochs=150, validation_data=(train_x[:6000,:,:,:], train_x[:6000,:,:,:]))


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("AutoEncoder model reconstruction loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(['train','validation'])
plt.savefig('AutoEncoder model mse.png')
plt.show()

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("AutoEncoder model reconstruction accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train','validation'])
plt.savefig('AutoEncoder model accuracy.png')
plt.show()


autoencoder.save_weights('conv_ae_weights.h5')

'''

'''
new_picture = autoencoder.predict(train_x[32:64])

new_picture = np.clip(new_picture*255, 0, 255).astype(np.uint8)


plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)   
for i in range(32):
    plt.subplot(8,4,i+1) 
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)   
    plt.imshow(new_picture[i])    
 
plt.savefig('new_picture.jpg')
'''

autoencoder.load_weights('conv_ae_weights.h5')

np.random.seed(3)

pca = PCA(n_components=0.95, whiten=True)

X_train_pca = pca.fit_transform(encoder.predict(train_x))

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(X_train_pca)


file = open(sys.argv[2], 'r', encoding='utf8')
test_x = []

for i in file:
    split = i.split(",")
    test_x.append(split[1:])


file.close()
test_x = test_x[1:]

pre = []
for i in range(len(test_x)):
    pre.append([str(i)])
    if(y_pred[int(test_x[i][0])-1]==y_pred[int(test_x[i][1])-1]):
        pre[i].append(int(1))
    else:
        pre[i].append(int(0))



file4 = sys.argv[3]
predict = open(file4, "w+")
s = csv.writer(predict, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(pre)):
    s.writerow(pre[i]) 
predict.close()