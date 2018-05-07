# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import cv2
from PIL import Image
import pandas as pd

face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def getAllPath(path):
	typelist = []
	filepathlist = []
	filelist = glob.glob(path + "*")
	for i, nowpath in enumerate(filelist):
		allFilePath = glob.glob(nowpath + "/*.bmp")
		for item in allFilePath:
			typelist.append(i)
			filepathlist.append(item)

	res = pd.DataFrame()
	res['type'] = typelist
	res['path'] = filepathlist
	return res

def getAllFile(path):
	filepathlist = []
	filelist = glob.glob(path + "*")
	for nowpath in filelist:
		allFilePath = glob.glob(nowpath + "/*.jpg")
		if not os.path.exists(nowpath.replace('lfw', 'nlfw')):
			os.mkdir(nowpath.replace('lfw', 'nlfw'))
		for item in allFilePath:
			filepathlist.append(item)
	return filepathlist

def dealImage(path):
	sample_image = cv2.imread(path)
	faces = face_patterns.detectMultiScale(sample_image,
		scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))
	return sample_image, faces

if __name__ == '__main__':
	pathList = getAllFile('lfw/test/')
	for path in pathList:
		img, face = dealImage(path)
		print(path)
		if face != ():
			(x, y, w, h) = face[0]
			print(x, y, w, h)
			img = img[(y):(y+h),(x):(x+w)]
			img = cv2.resize(img, (100,100), interpolation=cv2.INTER_AREA)
			newPath = path.replace('lfw', 'nlfw')
			cv2.imwrite(newPath, img)