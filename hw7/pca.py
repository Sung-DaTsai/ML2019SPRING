import os
import sys
import numpy as np 
from skimage.io import imread, imsave

IMAGE_PATH = sys.argv[1]

# Images for compression & reconstruction
test_image = [] 
for i in range(415):
    test_image.append(str(i)+'.jpg')

# Number of principal components used
k = 5
def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M


filelist = os.listdir(IMAGE_PATH) 

# Record the shape of images
img_shape = (600, 600, 3)

img_data = []
for filename in test_image:
    if(filename != '.DS_Store'):
        tmp = imread(sys.argv[1]+filename)  
        img_data.append(tmp.flatten())  


training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 


# Load image & Normalize
picked_img = imread(sys.argv[1]+sys.argv[2])  
X = picked_img.flatten().astype('float32') 
X -= mean


# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data.transpose(), full_matrices = False)  
    
# Compression
weight = np.array([np.dot(X, u[:,i]) for i in range(k)])  
smat = np.diag(s)
# Reconstruction
reconstruct = process(np.dot(weight, np.transpose(u)[0:5,:]) + mean)
imsave(sys.argv[3], reconstruct.reshape(img_shape)) 

'''
average = process(mean)
imsave('average.jpg', average.reshape(img_shape))  


for x in range(5):
    eigenface = process(u[:,x])
    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  


for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)
'''
