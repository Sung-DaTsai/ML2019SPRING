import numpy as np
from keras import backend as K
from keras.applications.vgg19 import decode_predictions
from keras.preprocessing import image
from keras.applications import vgg19
from PIL import Image
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.densenet import preprocess_input#, decode_predictions
#from keras.applications.densenet import DenseNet169
#from keras_applications import resnet
#from keras_applications.resnet import ResNet101
import sys
'''
model = VGG16()

model.summary()
'''
model = vgg19.VGG19(weights='imagenet')
#model = ResNet50(include_top=True, weights='imagenet')
#model = ResNet101(weights='imagenet', backend=keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
#model = DenseNet169(weights='imagenet')
np.random.seed(16)

for number in range(200):

    def plot_img_torch(x): #DenseNet121/169
        """
        x is a RGB image with shape (? ,224, 224, 3) 
        """
        #plt.rcParams['savefig.dpi'] = 224 #图片像素
        t = np.zeros_like(x[0])
        t[:,:,0] = x[0][:,:,0]
        t[:,:,1] = x[0][:,:,1]
        t[:,:,2] = x[0][:,:,2]
        t = np.clip(((t * [0.229, 0.224, 0.225]+[0.485, 0.456, 0.406])*255), 0, 255).astype(np.uint8)#/255

        im = Image.fromarray(t, 'RGB')
        #im.save('D:/MLHW5/answer/'+str(number_format)+'.png')
        im.save(sys.argv[2]+'/'+str(number_format)+'.png')
    
   
    
    number_format = "%03d" % number
    img_path = sys.argv[1]+'/'+str(number_format)+'.png'
    #img_path = 'D:/MLHW5/'+str(number_format)+'.png'
    img = image.load_img(img_path, target_size=(224, 224))


    # Create a batch and preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #x = x / 255

    # Get the initial predictions
    preds = model.predict(x)
    
    
    initial_class = np.argmax(preds)
    #print('Predicted:', decode_predictions(preds, top=3)[0])
    initial = decode_predictions(preds, top=3)[0][0][1]





    # Get current session (assuming tf backend)
    sess = K.get_session()
    # Initialize adversarial example with input image
    x_adv = x
    # Added noise
    x_noise = np.zeros_like(x)

    good = 1
    # Set variables
    epochs = 1
    epsilon = 0.35 #1.79
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
        
        '''
        if(decode_predictions(preds, top=3)[0][0][1] == initial):
            good = 0
        '''


    #plot_img(x_adv-x)
    if(good==0):
        plot_img_torch(x)
    if(good==1):
        plot_img_torch(x_adv)



    #K.clear_session()
