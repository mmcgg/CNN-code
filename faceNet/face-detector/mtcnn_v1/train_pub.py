# -*- coding: utf-8 -*-
import os

from load_data import DataGenerator


def main():
    filepath = os.path.dirname(os.path.abspath(__file__))

    dataset_root_dir = filepath + '/data/traindata/'

    pos_dataset_path = os.path.join(dataset_root_dir, 'pos_shuffle.h5')
    neg_dataset_path = os.path.join(dataset_root_dir, 'neg_shuffle.h5')
    part_dataset_path = os.path.join(dataset_root_dir, 'part_shuffle.h5')
    landmarks_dataset_path = os.path.join(dataset_root_dir, 'landmarks_shuffle.h5')

    data_generator = DataGenerator(pos_dataset_path, neg_dataset_path, part_dataset_path, landmarks_dataset_path, 64*7, im_size=12)
    data_gen = data_generator.generate()

    for item in data_gen:
        print(item[1][:,:1])
        exit(0)

if __name__ == '__main__':
    main()