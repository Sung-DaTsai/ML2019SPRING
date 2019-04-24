import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications import vgg19
from PIL import Image
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.applications.densenet import preprocess_input, decode_predictions
#from keras.applications.densenet import DenseNet169
#from keras_applications import resnet
#from keras_applications.resnet import ResNet101
#from keras_applications.vgg16 import preprocess_input, decode_predictions
import sys

model = vgg19.VGG19(weights='imagenet')
#model = ResNet50(include_top=True, weights='imagenet')
#model = ResNet101(weights='imagenet', backend=keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
#model = DenseNet169(weights='imagenet')
np.random.seed(16)

for number in range(200):

    def plot_img_caffe(x): #VGG16/VGG19/ResNet50
        """
        x is a BGR image with shape (? ,224, 224, 3) 
        """
        #plt.rcParams['savefig.dpi'] = 224 #图片像素
        t = np.zeros_like(x[0])
        
        t[:,:,0] = x[0][:,:,2]
        t[:,:,1] = x[0][:,:,1]
        t[:,:,2] = x[0][:,:,0]
        
        t = np.clip((t+[123.68, 116.779, 103.939]), 0, 255).astype(np.uint8)#/255
        plt.imshow(t)        
        #plt.show()

        im = Image.fromarray(t, 'RGB')
        im.save(sys.argv[2]+'/'+str(number_format)+'.png')
        #im.save('D:/MLHW5/answer/'+str(number_format)+'.png')

    
    number_format = "%03d" % number
    img_path = sys.argv[1]+'/'+str(number_format)+'.png'
    #img_path = 'D:/MLHW5/'+str(number_format)+'.png'
    img = image.load_img(img_path, target_size=(224, 224))
    '''
    plt.imshow(img)
    plt.grid('off')
    plt.axis('off')
    plt.close()
    '''
    # Create a batch and preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the initial predictions
    preds = model.predict(x)
    
    
    initial_class = np.argmax(preds)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    initial = decode_predictions(preds, top=3)[0][0][1]

    '''
    data = [decode_predictions(preds, top=3)[0][0][2], decode_predictions(preds, top=3)[0][1][2], decode_predictions(preds, top=3)[0][2][2]] 
    labels = [decode_predictions(preds, top=3)[0][0][1], decode_predictions(preds, top=3)[0][1][1], decode_predictions(preds, top=3)[0][2][1]] 
    plt.title(r'Original image '+str(number_format)+'.png')
    plt.yticks(np.linspace(0,1,6)) 
    plt.ylim([0,1])

    plt.bar(range(len(data)), data, tick_label=labels)
    plt.savefig('Original image '+str(number_format)+'.png')

    plt.show()
    plt.close()
    '''

    # Get current session (assuming tf backend)
    sess = K.get_session()
    # Initialize adversarial example with input image
    x_adv = x
    # Added noise
    x_noise = np.zeros_like(x)

    good = 1
    # Set variables
    epochs = 1
    epsilon = 4.7 #1.79
    prev_probs = []
        
    for i in range(epochs): 
        # One hot encode the initial class
        target = K.one_hot(initial_class, 1000)
        
        # Get the loss and gradient of the loss wrt the inputs
        loss = K.categorical_crossentropy(target, model.output)
        grads = K.gradients(loss, model.input)
        
        # Get the sign of the gradient
        delta = K.sign(grads[0])
        x_noise = x_noise + delta
        
        # Perturb the image
        x_adv = x_adv + epsilon*delta
        
        # Get the new image and predictions
        x_adv = sess.run(x_adv, feed_dict={model.input:x})
        preds = model.predict(x_adv)
        
        # Store the probability of the target class
        prev_probs.append(preds[0][initial_class])
        
        #print(i, preds[0][initial_class], decode_predictions(preds, top=3)[0])
        
        if(decode_predictions(preds, top=3)[0][0][1] == initial):
            good = 0

    #plot_img(x_adv-x)
    if(good==0):
        plot_img_caffe(x)
    if(good==1):
        plot_img_caffe(x_adv)

    '''
    data = [decode_predictions(preds, top=3)[0][0][2], decode_predictions(preds, top=3)[0][1][2], decode_predictions(preds, top=3)[0][2][2]] 
    labels = [decode_predictions(preds, top=3)[0][0][1], decode_predictions(preds, top=3)[0][1][1], decode_predictions(preds, top=3)[0][2][1]] 
    plt.title(r'Adversarial image '+str(number_format)+'.png')
    plt.yticks(np.linspace(0,1,6)) 
    plt.ylim([0,1])

    plt.bar(range(len(data)), data, tick_label=labels)
    plt.savefig('Adversarial image '+str(number_format)+'.png')

    plt.show()
    '''
    plt.close()
    
    #K.clear_session()
