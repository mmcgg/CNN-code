from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, ELU, Lambda, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import backend as K


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64):

    s_id = 'fire' + str(fire_id) + '/'
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')

    return x

def SqueezeNet(input_shape=(224,224,3),classes=1000):
    """Instantiates the SqueezeNet architecture.
    """
    img_input = Input(shape=input_shape)

    x = Convolution2D(64, (3, 3), strides=(1, 1), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=32)
    x = fire_module(x, fire_id=3, squeeze=16, expand=32)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=64)
    x = fire_module(x, fire_id=5, squeeze=32, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=96)
    x = fire_module(x, fire_id=7, squeeze=48, expand=96)
    x = fire_module(x, fire_id=8, squeeze=64, expand=128)
    x = fire_module(x, fire_id=9, squeeze=64, expand=128)
    x = Dropout(0.2, name='drop9')(x)

    x = Convolution2D(512, (1, 1), padding='same', name='conv10')(x)
    x = Activation('relu', name='relu_conv11')(x)

    x = Convolution2D(classes, (1, 1), padding='same', name='conv12')(x)
    x = Activation('relu', name='relu_conv13')(x)

    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)
    # x = Activation('linear', name='loss')(x)
    # x = Lambda(lambda  xx: K.l2_normalize(xx,axis=1))(x)

    model = Model(inputs=[img_input], outputs=x, name='squeezenet')

    return model

if __name__ == '__main__':
	model = SqueezeNet(input_shape=(32,32,3), classes=10)
	model.summary()

