from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, ELU, Input, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import backend as K

conv_dw = "conv_dw"
conv_c = "conv_c"
relu = "relu_"

def depthwise_conv(x, num_pwc_filter, dw_id, width_multiplier=1, downsample=False):

    s_id = 'dw' + str(dw_id) + '/'

    _stride = 2 if downsample else 1
    num_pwc_filter = round(num_pwc_filter * width_multiplier)
    
    x = Convolution2D(1, (3, 3), strides=(_stride, _stride), padding='same', name=s_id + conv_dw)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + conv_dw)(x)

    x = Convolution2D(num_pwc_filter, (1, 1), padding='same', name=s_id + conv_c)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + conv_c)(x)
    
    return x

def mobileNet(input_shape=(224,224,3), classes=1000): 
    ''''' 
    net structs 
    -------------------------------------------------- 
    conv / s2         | 3x3x3x32         | 224x224x3   
    conv dw / s1      | 3x3x1x32 dw      | 112x112x32 
    conv / s1         | 1x1x32x64        | 112x112x32   
    conv dw / s2      | 3x3x1x64 dw      | 112x112x64 
    conv / s1         | 1x1x64x128       | 56x56x64   
    conv dw / s1      | 3x3x1x128 dw     | 56x56x128 
    conv / s1         | 1x1x128x128      | 56x56x128  
    conv dw / s2      | 3x3x1x128 dw     | 56x56x128 
    conv / s1         | 1x1x128x256      | 28x28x128   
    conv dw / s1      | 3x3x1x256 dw     | 28x28x256 
    conv / s1         | 1x1x256x256      | 28x28x256   
    conv dw / s2      | 3x3x1x256 dw     | 28x28x256 
    conv / s1         | 1x1x256x512      | 14x14x256 
    -------------------------------------------------- 
    5x   
       conv dw / s1   | 3x3x1x512 dw     | 14x14x512 
       conv / s1      | 1x1x512x512      | 14x14x512   
    -------------------------------------------------- 
    conv dw / s2      | 3x3x1x512 dw     | 14x14x512 
    conv / s1         | 1x1x512x1024     | 7x7x512   
    conv dw / s2      | 3x3x1x1024 dw    | 7x7x1024 
    conv / s1         | 1x1x1024x1024    | 7x7x1024 
    Avg Pool / s1     | Pool 7x7         | 7x7x1024   
    FC / s1           | 1204x1000        | 1x1x1024 
    Softmax / s1      | Classifier       | 1x1x1000 
    -------------------------------------------------- 
    ''' 
    img_input = Input(shape=input_shape)

    x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn1')(x)
    x = depthwise_conv(x, num_pwc_filter=64, dw_id=2)
    x = depthwise_conv(x, num_pwc_filter=128, dw_id=3, downsample=True)
    x = depthwise_conv(x, num_pwc_filter=128, dw_id=4)
    x = depthwise_conv(x, num_pwc_filter=256, dw_id=5, downsample=True)
    x = depthwise_conv(x, num_pwc_filter=256, dw_id=6)
    x = depthwise_conv(x, num_pwc_filter=512, dw_id=7, downsample=True)

    x = depthwise_conv(x, num_pwc_filter=512, dw_id=8)
    x = depthwise_conv(x, num_pwc_filter=512, dw_id=9)
    x = depthwise_conv(x, num_pwc_filter=512, dw_id=10)
    x = depthwise_conv(x, num_pwc_filter=512, dw_id=11)
    x = depthwise_conv(x, num_pwc_filter=512, dw_id=12)

    x = depthwise_conv(x, num_pwc_filter=1024, dw_id=13)
    x = depthwise_conv(x, num_pwc_filter=1024, dw_id=14, downsample=True)
    x = AveragePooling2D((7,7), name='avg_poll15')(x)

    x = Dense(classes, activation=None, name='dense17')(x)
    x = Activation('softmax', name='loss')(x)

    model = Model(img_input, x, name='mobilenet')

    return model

if __name__ == '__main__':
    model = mobileNet(classes=1000)
    model.summary()





