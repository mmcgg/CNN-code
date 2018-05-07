from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.callbacks import TensorBoard

import numpy as np

import squeezeNet
import __init__

from dataset import loadData

def train():
	imodel = squeezeNet.SqueezeNet(input_shape=(32,32,3),classes=10)
	input_img = Input(shape=(32, 32, 3))
	ouput_img = imodel(input_img)
	model = Model(inputs=[input_img], outputs = ouput_img)
	model.summary()

	adam = Adam(lr=0.001)
	sgd = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

	cifar10_dir = 'C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = loadData.load_CIFAR10(cifar10_dir)

	y_train = to_categorical(y_train, num_classes=None)
	y_test = to_categorical(y_test, num_classes=None)

	model.fit(X_train, y_train, epochs=300, batch_size=128, validation_data=(X_test,y_test), callbacks=[TensorBoard(log_dir='tmp/log12')])
	model.save("model_300_0002.h5")

	# tensorboard --logdir=tmp/log --host 0.0.0.0 -- port 8888

def test():
	model = load_model('model.h5')
	cifar10_dir = 'C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/cifar-10-batches-py'
	X_train, y_train, X_test, y_test = loadData.load_CIFAR10(cifar10_dir)

	y_test = to_categorical(y_test, num_classes=None)
	y_pred = model.predict(X_test)

	right = 0
	for i in range(len(y_test)):
		right += np.sum((y_pred[i] == np.max(y_pred[i]))*(y_test[i]))

	print("accuracy: %f" %(right/len(y_test)))

if __name__ == '__main__':
	train()
	#test()

