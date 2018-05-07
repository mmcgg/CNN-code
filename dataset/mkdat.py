# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def create_couple_rgb(file_path):
    while True:
        folder = np.random.choice(glob.glob(file_path + "*"))
        filelist = glob.glob(folder + "/*.jpg")
        if filelist:
            break
    depth_file1 = np.random.choice(filelist)
    img = Image.open(depth_file1)
    #img.thumbnail((640,480))
    img = np.asarray(img)
    #img = img[140:340,220:420,:3]
    img = img[:,:,:3]
    #img = (img - np.mean(img)) / np.std(img)
    img = (img - np.mean(img)) / np.max(img)

    while True:
        depth_file2 = np.random.choice(filelist)
        if depth_file1 != depth_file2:
            break

    img2 = Image.open(depth_file2)
    #img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    #img2 = img2[160:360,240:440,:3]
    img2 = img2[:,:,:3]
    #img2 = (img2 - np.mean(img2)) / np.std(img2)
    img2 = (img2 - np.mean(img2)) / np.max(img2)
    
    return np.array([img, img2])


def create_wrong_rgb(file_path):
    while True:
        folder1 = np.random.choice(glob.glob(file_path + "*"))
        filelist = glob.glob(folder1 + "/*.jpg")
        if filelist:
            break
    depth_file1 = np.random.choice(filelist)
    img = Image.open(depth_file1)
    #img.thumbnail((640,480))
    img = np.asarray(img)
    #img = img[140:340,220:420,:3]
    img = img[:,:,:3]
    #img = (img - np.mean(img)) / np.std(img)
    img = (img - np.mean(img)) / np.max(img)

    while True:
        folder2 = np.random.choice(glob.glob(file_path + "*"))
        if folder1 != folder2 :
            filelist = glob.glob(folder1 + "/*.jpg")
            if filelist:
                break

    depth_file2 = np.random.choice(filelist)
    img2 = Image.open(depth_file2)
    #img2.thumbnail((640,480))
    img2 = np.asarray(img2)
    #img2 = img2[160:360,240:440,:3]
    img2 = img2[:,:,:3]
    #img2 = (img2 - np.mean(img2)) / np.std(img2)
    img2 = (img2 - np.mean(img2)) / np.max(img2)
    
    return np.array([img, img2])

def generator(batch_size = 100):
    X, y = [], []
    switch = True
    for i in range(batch_size):
        print(i)
        if switch:
            X.append(create_couple_rgb("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/nnlfw/train/").reshape((2,100,100,3)))
            y.append(np.array([0.]))
        else:
            X.append(create_wrong_rgb("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/nnlfw/train/").reshape((2,100,100,3)))
            y.append(np.array([1.]))
        switch = not switch
    X = np.asarray(X)
    y = np.asarray(y)

    X = [X[:,0], X[:,1]]

    fw = open('trainX.dat','wb')
    pickle.dump(X, fw, -1)
    fw.close()

    fw = open('trainY.dat','wb')
    pickle.dump(y, fw, -1)
    fw.close()

if __name__ == '__main__':
    generator(10000)
