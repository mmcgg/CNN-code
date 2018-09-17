# coding: utf-8

from keras.models import Model
from keras.layers import concatenate, Input, Reshape, Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.layers.advanced_activations import PReLU

class FaceNet(object):

    """FaceNet"""

    def __init__(self):
        self.net_radio = None

    def cal_mask(self, label_true, _type='face'):

        def true_func():
            return 0

        def false_func():
            return 1

        label_true_int32 = K.cast(label_true, dtype=K.int32)
        if _type == 'face':
            label_filtered = K.map_fn(lambda x: K.cond(K.equal(x[0], x[1]), true_func, false_func), label_true_int32)
        elif _type == 'bbox':
            label_filtered = K.map_fn(lambda x: K.cond(K.equal(x[0], 1), true_func, false_func), label_true_int32)
        elif _type == 'landmark':
            label_filtered = K.map_fn(lambda x: K.cond(K.logical_and(K.equal(x[0], 1), K.equal(x[1], 1)),
                                                         false_func, true_func), label_true_int32)
        else:
            raise ValueError('Unknown type of: {} while calculate mask'.format(_type))

        mask = K.cast(label_filtered, dtype=K.int32)
        return mask

    def loss_face(self, label_true, label_pred):

        label_int = self.cal_mask(label_true, 'face')

        num_cls_prob = K.size(label_pred)
        print('num_cls_prob: ', num_cls_prob)
        cls_prob_reshape = K.reshape(label_pred, [num_cls_prob, -1])
        print('label_pred shape: ', K.shape(label_pred))
        num_row = K.shape(label_pred)[0]
        num_row = K.to_int32(num_row)
        row = K.range(num_row) * 2
        indices_ = row + label_int
        label_prob = K.squeeze(K.gather(cls_prob_reshape, indices_))
        loss = -K.log(label_prob + 1e-10)

        valid_inds = cal_mask(label_true, 'face')
        num_valid = K.reduce_sum(valid_inds)

        keep_num = K.cast(K.cast(num_valid, dtype=K.float32) * num_keep_radio, dtype=K.int32)
        # set 0 to invalid sample
        loss = loss * K.cast(valid_inds, dtype=K.float32)
        loss, _ = K.nn.top_k(loss, k=keep_num)
        return K.reduce_mean(loss)

    def loss_box(self, label_true, bbox_true, bbox_pred):

        mask = self.cal_mask(label_true, 'bbox')
        num = K.reduce_sum(mask)
        keep_num = K.cast(num, dtype=K.int32)

        bbox_true1 = K.boolean_mask(bbox_true, mask, axis=0)
        bbox_pred1 = K.boolean_mask(bbox_pred, mask, axis=0)

        square_error = K.square(bbox_pred1 - bbox_true1)
        square_error = K.reduce_sum(square_error, axis=1)

        _, k_index = K.nn.top_k(square_error, k=keep_num)
        square_error = K.gather(square_error, k_index)

        return K.reduce_mean(square_error)

    def loss_landmark(self, landmark_pred, landmark_target, label):

        mask = self.cal_mask(label_true, 'landmark')
        num = K.reduce_sum(mask)
        keep_num = K.cast(num, dtype=K.int32)

        landmark_true1 = K.boolean_mask(landmark_true, mask)
        landmark_pred1 = K.boolean_mask(landmark_pred, mask)

        square_error = K.square(landmark_pred1 - landmark_true1)
        square_error = K.reduce_sum(square_error, axis=1)

        _, k_index = K.nn.top_k(square_error, k=keep_num)
        square_error = K.gather(square_error, k_index)

        return K.reduce_mean(square_error)

    def loss_func(y_true, y_pred):

        labels_true = y_true[:, :2]
        bbox_true = y_true[:, 2:6]
        landmark_true = y_true[:, 6:]

        labels_pred = y_pred[:, :2]
        bbox_pred = y_pred[:, 2:6]
        landmark_pred = y_pred[:, 6:]

        face_loss = self.loss_face(labels_true, labels_pred)
        box_loss = self.loss_box(labels_true, bbox_true, bbox_pred)
        landmark_loss = self.loss_landmark(labels_true, landmark_true, landmark_pred)

        return label_loss*self.net_radio[0] + bbox_loss*self.net_radio [1]+ landmark_loss*self.net_radio[2]

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






        
        