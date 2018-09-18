# coding: utf-8

from keras.models import Model
from keras.layers import concatenate, Input, Reshape, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers.advanced_activations import PReLU

import tensorflow as tf

class FaceNet(object):

    """FaceNet"""

    def __init__(self):
        self.net_radio = None

    def cal_mask(self, label_true, _type='face'):
        """ 针对不同的任务过滤对应的数据"""

        def true_func():
            return 1

        def false_func():
            return 0

        label_true_float32 = K.cast(label_true, dtype=tf.int32)
        if _type == 'face':
            label_filtered = K.map_fn(lambda x: tf.cond(tf.logical_or(tf.equal(x[0], 0), tf.equal(x[0], 1)), true_func, false_func), label_true_float32)
        elif _type == 'bbox':
            label_filtered = K.map_fn(lambda x: tf.cond(tf.logical_or(tf.equal(x[0], -1), tf.equal(x[0], 1)), true_func, false_func), label_true_float32)
        elif _type == 'landmark':
            label_filtered = K.map_fn(lambda x: tf.cond(tf.equal(x[0], -2), true_func, false_func), label_true_float32)
        else:
            raise ValueError('Unknown type of: {} while calculate mask'.format(_type))

        mask = K.cast(label_filtered, dtype=tf.int32)
        return mask

    def loss_face(self, label_true, label_pred):

        mask = self.cal_mask(label_true, 'face')

        label_true1 = tf.boolean_mask(label_true, mask, axis=0)
        label_pred1 = tf.boolean_mask(label_pred, mask, axis=0)

        loss = label_true1*(-K.log(label_pred1+1e-10)) + (1-label_true1)*(-K.log(1-label_pred1+1e-10))

        return tf.reduce_mean(loss)

    def loss_box(self, label_true, bbox_true, bbox_pred):

        mask = self.cal_mask(label_true, 'bbox')

        bbox_true1 = tf.boolean_mask(bbox_true, mask, axis=0)
        bbox_pred1 = tf.boolean_mask(bbox_pred, mask, axis=0)

        square_error = K.square(bbox_pred1 - bbox_true1)
        square_error = tf.reduce_sum(square_error, axis=1)

        return tf.reduce_mean(square_error)

    def loss_landmark(self, label_true, landmark_true, landmark_pred):

        mask = self.cal_mask(label_true, 'landmark')

        landmark_true1 = tf.boolean_mask(landmark_true, mask)
        landmark_pred1 = tf.boolean_mask(landmark_pred, mask)

        square_error = K.square(landmark_pred1 - landmark_true1)
        square_error = tf.reduce_sum(square_error, axis=1)

        return tf.reduce_mean(square_error)

    def loss_func(self, y_true, y_pred):

        """ 损失函数 """
        """ 
            y_true 有15列 (1+4+10)
            y_pred 有16列（2+4+10）
        """

        face_true = y_true[:, :1]
        box_true = y_true[:, 1:5]
        landmark_true = y_true[:, 5:]

        face_pred = y_pred[:, :1]
        box_pred = y_pred[:, 2:6]
        landmark_pred = y_pred[:, 6:]

        face_loss = self.loss_face(face_true, face_pred)
        box_loss = self.loss_box(face_true, box_true, box_pred)
        landmark_loss = self.loss_landmark(face_true, landmark_true, landmark_pred)

        return face_loss*self.net_radio[0] + box_loss*self.net_radio[1] + landmark_loss*self.net_radio[2]

    def accuracy(self, y_true, y_pred, threshold=0.6):

        mask = self.cal_mask(y_true, 'face')

        y_true1 = tf.boolean_mask(y_true, mask, axis=0)
        y_pred1 = tf.boolean_mask(y_pred, mask, axis=0)

        def true_func():
            return 1

        def false_func():
            return 0

        y_pred1 = K.map_fn(lambda x: tf.cond(x[0] >= threshold, true_func, false_func), y_pred1)
        acc = K.map_fn(lambda x: tf.cond(tf.equal(x[0], 0), true_func, false_func), y_true1 - y_pred1)

        return tf.reduce_mean(acc)


class Pnet(FaceNet):

    """docstring for Pnet"""

    def __init__(self):
        self.net_radio = [1, 0.5, 0.5]

    def model(self, training=False):

        img_input = Input(shape=(12, 12, 3)) if training else Input(shape=(None, None, 3))

        x = Convolution2D(10, (3, 3), strides=(1, 1), padding='valid', name='conv1')(img_input)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv1')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

        x = Convolution2D(16, (3, 3), strides=(1, 1), padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv2')(x)
        
        x = Convolution2D(32, (3, 3), strides=(1, 1), padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv3')(x)
        
        face = Convolution2D(2, (1, 1), strides=(1, 1), padding='valid', activation='softmax', name='conv4_face')(x)
        box = Convolution2D(4, (1, 1), strides=(1, 1), padding='valid', name='conv4_2')(x)
        landmark = Convolution2D(10, (1, 1), strides=(1, 1), padding='valid', name='conv4_3')(x)

        if training:
            face = Reshape((2,), name='face')(face)
            box = Reshape((4,), name='box')(box)
            landmark = Reshape((10,), name='landmark')(landmark)
            outputs = concatenate([face, box, landmark])
            model = Model(inputs=[img_input], outputs=[outputs], name='P_Net')
        else:
            model = Model(inputs=[img_input], outputs=[face, box, landmark], name='P_Net')
        return model

class Rnet(FaceNet):

    def __init__(self):
        self.net_radio = [1, 0.5, 0.5]

    def model(self, training=False):

        img_input = Input(shape=(24, 24, 3))

        x = Convolution2D(28, (3, 3), padding='same', strides=(1, 1), name='conv1')(img_input)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pooling1')(x)

        x = Convolution2D(48, 3, padding='valid', strides=(1, 1), name='conv2')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv2')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pooling2')(x)

        x = Convolution2D(64, 2, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv3')(x)

        x = Dense(128, name='dense')(x)
        x = PReLU(name='prelu4')(x)

        x = Flatten()(x)

        face = Dense(2, activation='softmax', name='face')(x)
        bbox = Dense(4, name='box')(x)
        landmark = Dense(10, name='landmark')(x)

        if training:
            outputs = concatenate([face, bbox, landmark])
            model = Model(inputs=[img_input], outputs=[outputs], name='R_Net')
        else:
            model = Model(inputs=[img_input], outputs=[face, bbox, landmark], name='R_Net')

        return model



class Onet(FaceNet):

    def __init__(self):
        self.net_radio = [1, 0.5, 1]

    def model(self, training=False):

        img_input = Input(shape=(48, 48, 3))

        x = Convolution2D(32, 3, padding='same', strides=(1, 1), name='conv1')(img_input)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pooling1')(x)

        x = Convolution2D(64, 3, padding='valid', strides=(1, 1), name='conv2')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv2')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pooling2')(x)

        x = Convolution2D(64, 3, padding='valid', strides=(1, 1), name='conv3')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv3')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling3')(x)

        x = Convolution2D(128, 2, padding='valid', strides=(1, 1), name='conv4')(x)
        x = PReLU(shared_axes=(1, 2), name='prelu_conv4')(x)

        x = Dense(256, name='dense')(x)
        x = PReLU(name='prelu5')(x)

        x = Flatten()(x)

        face = Dense(2, activation='softmax', name='face')(x)
        box = Dense(4, name='box')(x)
        landmark = Dense(10, name='landmark')(x)

        if training:
            outputs = concatenate([face, box, landmark])
            model = Model(inputs=[img_input], outputs=[outputs], name='O_Net')
        else:
            model = Model(inputs=[img_input], outputs=[face, box, landmark], name='O_Net')

        return model

if __name__ == '__main__':
    
    pnet = Pnet()
    pnet.model(training=True).summary()

    rnet = Rnet()
    rnet.model(training=True).summary()

    onet = Onet()
    onet.model(training=True).summary()






        
        