# -*-coding:utf8-*-#

import os
import sys
import time

import numpy
from PIL import Image


def load_data(dataset_path):
    img = Image.open(dataset_path)
    img_ndarray = numpy.asarray(img, dtype='float64')/256
    faces = numpy.empty((400,2679))
    for row in range(20):
        for column in range(20):
            faces[row*20+column] = numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

    label = numpy.empty(400)
    for i in range(40):
        label[i*10:i*10+10] = i
        label=label.astype(numpy.int)

    #分成训练集、验证集、测试集，大小如下
    train_data = numpy.empty((320,2679))
    train_label = numpy.empty(320)
    valid_data = numpy.empty((40,2679))
    valid_label = numpy.empty(40)
    test_data = numpy.empty((40,2679))
    test_label = numpy.empty(40)

    for i in range(40):
        train_data[i*8:i*8+8] = faces[i*10:i*10+8]
        train_label[i*8:i*8+8] = label[i*10:i*10+8]
        valid_data[i] = faces[i*10+8]
        valid_label[i] = label[i*10+8]
        test_data[i] = faces[i*10+9]
        test_label[i] = label[i*10+9]

    rval = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label)]
    return rval

if __name__ == '__main__':
	print(load_data('olivettifaces.gif'))