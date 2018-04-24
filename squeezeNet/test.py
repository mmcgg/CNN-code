from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K

from keras.models import load_model
import numpy as np

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def create_couple_rgb(file_path):

    folder = np.random.choice(glob.glob(file_path + "*"))

    depth_file1 = np.random.choice(glob.glob(folder + "/*.bmp"))
    img = Image.open(depth_file1)
    #img.thumbnail((640,480))
    img = np.asarray(img)
    #img = img[140:340,220:420,:3]
    img = img[:,:,:3]
    img = (img - np.mean(img)) / np.max(img)

    while True:
        depth_file2 = np.random.choice(glob.glob(folder + "/*.bmp"))
        if depth_file1 != depth_file2:
            break

    img2 = Image.open(depth_file2)
    #img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    #img2 = img2[160:360,240:440,:3]
    img2 = img2[:,:,:3]
    img2 = (img2 - np.mean(img2)) / np.max(img2)
    
    return np.array([img, img2])


def create_wrong_rgb(file_path):

    folder1 = np.random.choice(glob.glob(file_path + "*"))

    depth_file1 = np.random.choice(glob.glob(folder1 + "/*.bmp"))
    img = Image.open(depth_file1)
    #img.thumbnail((640,480))
    img = np.asarray(img)
    #img = img[140:340,220:420,:3]
    img = img[:,:,:3]
    img = (img - np.mean(img)) / np.max(img)

    while True:
        folder2 = np.random.choice(glob.glob(file_path + "*"))
        if folder1 != folder2:
            break

    depth_file2 = np.random.choice(glob.glob(folder2 + "/*.bmp"))
    img2 = Image.open(depth_file2)
    #img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    #img2 = img2[160:360,240:440,:3]
    img2 = img2[:,:,:3]
    img2 = (img2 - np.mean(img2)) / np.max(img2)
    
    return np.array([img, img2])

def contrastive_loss(y_true, y_pred):
	margin = 1.
	return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

model_final = load_model('C:/Users/Spencer/Documents/GitHub/CNN-code/squeezeNet/model_faceId_new.h5', {'contrastive_loss':contrastive_loss})

filepath = "C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/nFaceId/faceid_val/"
ly = []
ln = []
for i in range(500):
    print(i)
    cop = create_couple_rgb(filepath)
    pr = model_final.predict([cop[0].reshape((1,100,100,3)), cop[1].reshape((1,100,100,3))])
    ly.append(pr)
    cop = create_wrong_rgb(filepath)
    pr = model_final.predict([cop[0].reshape((1,100,100,3)), cop[1].reshape((1,100,100,3))])
    ln.append(pr)
    
ly = [x[0][0] for x in ly]
ln = [x[0][0] for x in ln]

lx = np.linspace(0,1,41)
ly1 = [sum([x<=i for x in ly]) for i in lx]
ly2 = [sum([x<=i for x in ln]) for i in lx]

plt.plot(lx,ly1,"y-",label="migration time")  
plt.plot(lx,ly2,"m-",label="request delay")
plt.show()