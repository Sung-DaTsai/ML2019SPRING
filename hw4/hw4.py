import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Flatten, Dropout
from keras.layers import Dense 
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros
from keras.preprocessing.image import save_img
from keras import layers
from skimage import transform
from lime import lime_image
from skimage.segmentation import slic
from skimage import color


'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
'''


np.random.seed(16)

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

'''
model = Sequential()  
model.add(Conv2D(32, kernel_size = (3, 3), strides=(1, 1), padding='same', input_shape = (48, 48, 1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))


model.add(Conv2D(32, kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Conv2D(32, kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2), padding='same'))
model.add(Dropout(0.3))


model.add(Conv2D(64, kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))


model.add(Conv2D(64, kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Conv2D(64, kernel_size = (3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size = (2, 2), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Conv2D(128, kernel_size = (2, 2), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Conv2D(128, kernel_size = (2, 2), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(MaxPooling2D(pool_size=(2, 2), strides = (2, 2), padding='valid'))
model.add(Dropout(0.3))


model.add(Flatten())


model.add(Dense(units = 512))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(LeakyReLU(alpha=0.3))


model.add(Dense(units = 7, activation = 'softmax'))
model.summary()
'''
model = load_model('my_model.h5')

# -------------------------Q1----------------------

class SaliencyMask(object):
    def __init__(self, model, output_index=0):
        pass

    def get_mask(self, input_image):
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class GradientSaliency(SaliencyMask):

    def __init__(self, model, output_index = 0):

        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):

        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients


class VisualBackprop(SaliencyMask):
    def __init__(self, model, output_index = 0):
        inps = [model.input]           
        outs = [layer.output for layer in model.layers]    
        self.forward_pass = K.function(inps, outs)         
        
        self.model = model

    def get_mask(self, input_image):
        x_value = np.expand_dims(input_image, axis=0)
        
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        for i in range(len(self.model.layers) - 1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis = 3, keepdims = True)
                layer = layer - np.min(layer)
                layer = layer / (np.max(layer) - np.min(layer) + 1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        x = Input(shape = (None, None, 1))
        y = Conv2DTranspose(filters = 1, 
                            kernel_size = (3, 3), 
                            strides = (2, 2), 
                            padding = 'same', 
                            kernel_initializer = Ones(), 
                            bias_initializer = Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input]                                   
        outs = [deconv_model.layers[-1].output]          
        deconv_func = K.function(inps, outs)             
        
        return deconv_func([feature_map, 0])[0]

n_classes = 7


X_train = train_x
Y_train = train_y

real_val = []
for i in range(len(train_x)):
    real_val.append(np.argmax(train_y[i]))


Y_train_label = np.array(real_val)

feature_list = [0, 299, 5, 7, 6, 15, 4]


for i in range(n_classes):
    np.random.seed(16)
    k = feature_list[i]
    img = np.array(X_train[k])
    
    vanilla = GradientSaliency(model, Y_train_label[k])
    mask = vanilla.get_mask(img)
    
    plt.imshow(mask.reshape((48, 48)), cmap = 'jet')
    plt.title("Saliency map for feature "+str(i)+", picture number "+str(k))
    plt.colorbar()
    plt.savefig(sys.argv[2]+'fig1_'+str(i)+'.jpg')
    #plt.show()
    plt.close('all')

#--------------------Q2---------------------------
np.random.seed(16)


def normalize(x):

    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_name,
                    step=1,
                    epochs=30,
                    upscaling_steps=9,
                    upscaling_factor=1.2,
                    output_dim=(412, 412),
                    filter_range=(0, None)):


    def _generate_filter_image(input_img, layer_output, filter_index):
        
        nolist = [0,2,3,5,7,8,10,11,12,13,14,15,16,17,19,23,26,27,28,29,30,
                  31,32,33,34,35,36,37,38,39,40,45,46,48,49,50,51,52,54,55,56,58,60,61]
        if(filter_index in nolist):
            return None
        
        loss = K.mean(layer_output[:, :, :, filter_index])

        grads = K.gradients(loss, input_img)[0]

        grads = normalize(grads)

        iterate = K.function([input_img], [loss, grads])


        intermediate_dim = tuple(
            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
        
        np.random.seed(16)
        input_img_data = np.random.random((1, intermediate_dim[0], intermediate_dim[1], 1))
        input_img_data = (input_img_data - 0.5) * 20 + 128


        for up in reversed(range(upscaling_steps)):
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                if loss_value <= K.epsilon():
                    return None


            intermediate_dim = tuple(int(x / (upscaling_factor ** up)) for x in output_dim)

            img = deprocess_image(input_img_data[0])
            img = img.reshape((img.shape[0], img.shape[0]))
            img = transform.resize(img, intermediate_dim)

            input_img_data = [process_image(img, input_img_data[0])]

            
            input_img_data = np.array(input_img_data)
            input_img_data = input_img_data.reshape((1, input_img_data.shape[1],input_img_data.shape[1],1))          

            

        img = deprocess_image(input_img_data[0])
        #print('Costs of filter {:3}: {:5.0f} '.format(filter_index, loss_value))
        return img, loss_value

    def _draw_filters(filters, n=None):

        if n is None:
            n = int(np.floor(np.sqrt(len(filters))))
            
        filters.sort(key=lambda x: x[1], reverse=True)
        filters = filters[:n * n]

        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 3), dtype='uint8')

        for i in range(n):
            for j in range(n):
                img, _ = filters[i * n + j]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[
                    width_margin: width_margin + output_dim[0],
                    height_margin: height_margin + output_dim[1], :] = img


        save_img(sys.argv[2]+'fig2_1.jpg', stitched_filters)


    assert len(model.inputs) == 1
    input_img = model.inputs[0]


    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

    output_layer = layer_dict[layer_name]


    assert isinstance(output_layer, layers.Conv2D)


    filter_lower = filter_range[0]
    filter_upper = (filter_range[1]
                    if filter_range[1] is not None
                    else len(output_layer.get_weights()[1]))
    filter_upper = 64
    assert(filter_lower >= 0
           and filter_upper <= len(output_layer.get_weights()[1])
           and filter_upper > filter_lower)
    #print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))


    processed_filters = []
    for f in range(filter_lower, filter_upper):
        img_loss = _generate_filter_image(input_img, output_layer.output, f)

        if img_loss is not None:
            processed_filters.append(img_loss)

    #print('{} filter processed.'.format(len(processed_filters)))
    _draw_filters(processed_filters)



LAYER_NAME = model.layers[22].name
visualize_layer(model, LAYER_NAME)




conv_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_7').output)
conv_output = conv_layer_model.predict(train_x[0].reshape((1,48,48,1)))
conv_output = conv_output.reshape((11,11,128))


plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)   
for i in range(64):
    plt.subplot(8,8,i+1) 
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)   
    plt.imshow(conv_output[:,:,i], cmap=plt.cm.gray)    
 
plt.savefig(sys.argv[2]+'fig2_2.jpg')

#-------------------------Q3---------------------------

plt.close('all')


feature_list = [0, 473, 5, 7, 6, 15, 11]

def predict(input):
    a = color.rgb2gray(input).reshape((1, 48, 48, 1))

    return model.predict(a)
def segmentation(input):
    a = color.rgb2gray(input).reshape((48, 48))

    return slic(a)



explainer = lime_image.LimeImageExplainer()
#plt.figure(figsize=(30, 30))

def explain(instance, predict_fn, **kwargs):
    return explainer.explain_instance(instance, predict_fn, **kwargs)

for i in range(7):
    train_x_rgb = color.gray2rgb(train_x[feature_list[i]]).reshape(48, 48, 3)
    np.random.seed(16)
    explaination = explainer.explain_instance(image = train_x_rgb, classifier_fn=predict, top_labels=7, batch_size=1, segmentation_fn=segmentation)

    image, mask = explaination.get_image_and_mask(
                                label = Y_train_label[feature_list[i]],
                                positive_only=False,
                                hide_rest=False,
                                num_features=10,
                                min_weight=0.0)
    

    image = image.astype(dtype='float32')
    plt.imshow(image, cmap=plt.cm.jet)

    plt.colorbar()

    plt.savefig(sys.argv[2]+'fig3_'+str(i)+'.jpg')
    plt.close('all')

