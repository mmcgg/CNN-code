# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

def create_couple_rgb(file_path):

    folder = np.random.choice(glob.glob(file_path + "*"))

    depth_file1 = np.random.choice(glob.glob(folder + "/*.bmp"))
    img = Image.open(depth_file1)
    img.thumbnail((640,480))
    img = np.asarray(img)
    img = img[140:340,220:420,:3]
    img = (img - np.mean(img)) / np.max(img)

    while True:
        depth_file2 = np.random.choice(glob.glob(folder + "/*.bmp"))
        if depth_file1 != depth_file2:
            break

    img2 = Image.open(depth_file2)
    img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    img2 = img2[160:360,240:440,:3]
    img2 = (img2 - np.mean(img2)) / np.max(img2)
    
    return np.array([img, img2])


def create_wrong_rgb(file_path):

    folder1 = np.random.choice(glob.glob(file_path + "*"))

    depth_file1 = np.random.choice(glob.glob(folder1 + "/*.bmp"))
    img = Image.open(depth_file1)
    img.thumbnail((640,480))
    img = np.asarray(img)
    img = img[140:340,220:420,:3]
    img = (img - np.mean(img)) / np.max(img)

    while True:
        folder2 = np.random.choice(glob.glob(file_path + "*"))
        if folder1 != folder2:
            break

    depth_file2 = np.random.choice(glob.glob(folder2 + "/*.bmp"))
    img2 = Image.open(depth_file2)
    img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    img2 = img2[160:360,240:440,:3]
    img2 = (img2 - np.mean(img2)) / np.max(img2)
    
    return np.array([img, img2])

def generator(batch_size):
	while 1:
		X, y = [], []
		switch = True
		for _ in range(batch_size):
			if switch:
				X.append(create_couple_rgb("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/faceId/faceid_train/").reshape((2,200,200,3)))
				y.append(np.array([0.]))
			else:
				X.append(create_wrong_rgb("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/faceId/faceid_train/").reshape((2,200,200,3)))
				y.append(np.array([1.]))
			switch = not switch
		X = np.asarray(X)
		y = np.asarray(y)
		yield [X[:,0], X[:,1]], y

def val_generator(batch_size):
	while 1:
		X, y = [], []
		switch=True
		for _ in range(batch_size):
			if switch:
				X.append(create_couple_rgb("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/faceId/faceid_val/").reshape((2,200,200,3)))
				y.append(np.array([0.]))
			else:
				X.append(create_wrong_rgb("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/faceId/faceid_val/").reshape((2,200,200,3)))
				y.append(np.array([1.]))
			switch = not switch
		X = np.asarray(X)
		y = np.asarray(y)
		yield [X[:,0], X[:,1]], y

if __name__ == '__main__':
   	create_couple_rgb("faceid_val/")
