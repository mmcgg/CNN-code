# -*- coding: utf-8 -*-

from keras.optimizers import Adam
import os

from model_net import Pnet
from load_data import DataGenerator

filepath = os.path.dirname(os.path.abspath(__file__))

GAN_DATA_ROOT_DIR = filepath + '/data/traindata/'
model_file = filepath + '/model/pnet.h5'

def train_with_data_generator(dataset_root_dir=GAN_DATA_ROOT_DIR, model_file=model_file, weights_file=None):

    net_name = 'p_net'
    batch_size = 64*7
    epochs = 100
    learning_rate = 0.001
    
    pos_dataset_path = os.path.join(dataset_root_dir, 'pos_shuffle.h5')
    neg_dataset_path = os.path.join(dataset_root_dir, 'neg_shuffle.h5')
    part_dataset_path = os.path.join(dataset_root_dir, 'part_shuffle.h5')
    landmarks_dataset_path = os.path.join(dataset_root_dir, 'landmarks_shuffle.h5')

    data_generator = DataGenerator(pos_dataset_path, neg_dataset_path, part_dataset_path, landmarks_dataset_path, batch_size, im_size=12)
    data_gen = data_generator.generate()
    steps_per_epoch = data_generator.steps_per_epoch()

    _p_net = Pnet()
    _p_net_model = _p_net.model(training=True)
    _p_net_model.summary()
    if weights_file is not None:
        _p_net_model.load_weights(weights_file)

    #_p_net_model.compile(Adam(lr=learning_rate), loss=_p_net.loss_func, metrics=[_p_net.accuracy])
    _p_net_model.compile(Adam(lr=learning_rate), loss=_p_net.loss_func, metrics=['accuracy'])

    _p_net_model.fit_generator(data_gen,
                                steps_per_epoch=steps_per_epoch,
                                initial_epoch=0,
                                epochs=epochs)

    _p_net_model.save_weights(model_file)

if __name__ == '__main__':

    train_with_data_generator()