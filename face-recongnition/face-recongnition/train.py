from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, ELU, Lambda, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras import losses
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
import data


def main():
    datasets = data.load_data('olivettifaces.gif')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_set_x = train_set_x.reshape((320,57,47,1))
    valid_set_x = valid_set_x.reshape((40,57,47,1))
    test_set_x = test_set_x.reshape((40,57,47,1))

    train_set_y = np_utils.to_categorical(train_set_y, 40)
    valid_set_y = np_utils.to_categorical(valid_set_y, 40)
    test_set_y = np_utils.to_categorical(test_set_y, 40)

    model = modelNet(input_shape=(57, 47, 1), classes=40)
    model.summary()

    sgd = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(train_set_x, train_set_y, batch_size=40, epochs=100, verbose=1, validation_data=(valid_set_x, valid_set_y), callbacks=[early_stopping])

    model.save("model3.h5")


def modelNet(input_shape=(57, 47, 1), classes=40):

    img_input = Input(shape=input_shape)

    x = Convolution2D(5, (5, 5), strides=(1, 1), padding='valid', name='conv1')(img_input)
    #x = Activation('relu', name='relu_conv1')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    x = Convolution2D(10, (5, 5), strides=(1, 1), padding='valid', name='conv2')(x)
    x = LeakyReLU()(x)
    #x = Activation('relu', name='relu_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
    
    x = Flatten(name='flatten1')(x)

    x = Dense(128, name='dense1')(x)
    x = LeakyReLU()(x)
    #x = Activation('relu', name='relu_conv3')(x)

    x = Dense(classes, name='dense2')(x)
    output = Activation('softmax', name='loss')(x)

    model = Model(inputs=[img_input], outputs=output, name='modelnet')

    return model


if __name__ == '__main__':
    main()
