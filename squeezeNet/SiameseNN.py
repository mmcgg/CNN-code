from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard

import numpy as np

import squeezeNet
import __init__

from dataset import loadPairSample

def euclidean_distance(inputs):
    assert len(inputs) == 2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
	margin = 1.
	return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def train():
	imodel = squeezeNet.SqueezeNet(input_shape=(200,200,3),classes=128)
	input_img1 = Input(shape=(200, 200, 3))
	input_img2 = Input(shape=(200, 200, 3))
	ouput_img1 = imodel(input_img1)
	ouput_img2 = imodel(input_img2)

	lambda_merge = Lambda(euclidean_distance)([ouput_img1, ouput_img2])
	model = Model(inputs=[input_img1, input_img2], outputs = lambda_merge)
	model.summary()

	adam = Adam(lr=0.001)
	sgd = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=adam, loss=contrastive_loss)

	gen = loadPairSample.generator(16)
	val_gen = loadPairSample.val_generator(4)

	outputs = model.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20, callbacks=[TensorBoard(log_dir='tmp/log2')])
	model.save("model_faceId.h5")

	# tensorboard --logdir=tmp/log --host 0.0.0.0 -- port 8888

if __name__ == '__main__':
	train()
	#test()

