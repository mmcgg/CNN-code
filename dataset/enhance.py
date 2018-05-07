# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
import numpy as np
import glob
from PIL import Image

def getAllFile(path):
	filepathlist = []
	filelist = glob.glob(path + "*")
	for nowpath in filelist:
		allFilePath = glob.glob(nowpath + "/*.jpg")
		if not os.path.exists(nowpath.replace('nlfw', 'nnlfw')):
			os.mkdir(nowpath.replace('nlfw', 'nnlfw'))
		for item in allFilePath:
			filepathlist.append(item)
	return filepathlist

datagen = ImageDataGenerator(
	rotation_range = 0.2,
	width_shift_range = 0.05,
	height_shift_range = 0.05,
	rescale = 1/255,
	shear_range = 0.2,
	zoom_range = 0.1,
	channel_shift_range = 10,
	fill_mode='nearest')

for paths in getAllFile('nlfw/train/'):
	print(paths)
	img = load_img(paths)
	x = img_to_array(img)
	x = x.reshape((1,) + x.shape)
	filepath = '/'.join(paths.replace('nlfw', 'nnlfw').split('\\')[:-1])
	i = 0
	for batch in datagen.flow(x,
		batch_size=1,
		save_to_dir=filepath,#生成后的图像保存路径
		save_prefix='lena',
		save_format='jpg'):
	    i += 1
	    if i > 10: #这个20指出要扩增多少个数据
	        break  # otherwise the generator would loop indefinitely