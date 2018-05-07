# -*- coding: utf-8 -*-
import sys
import dlib
import cv2
import os
import glob
import pandas as pd
import numpy as np

import __init__

from dataset import dealFace

current_path = os.getcwd()  # 获取当前路径
# 模型路径
predictor_path = current_path + "\\shape_predictor_68_face_landmarks.dat"
face_rec_model_path = current_path + "\\dlib_face_recognition_resnet_model_v1.dat"

# 读入模型
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

def main():

	pathData = dealFace.getAllPath("C:/Users/Spencer/Documents/GitHub/CNN-code/dataset/faceId/faceid_train/")

	numlist = []
	typelist = []
	veclist = []

	for i in range(len(pathData)):

		img_path = pathData['path'][i]
		print("Processing file: {}".format(img_path))
		# opencv 读取图片，并显示
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		# opencv的bgr格式图片转换成rgb格式
		b, g, r = cv2.split(img)
		img2 = cv2.merge([r, g, b])

		dets = detector(img, 1)   # 人脸标定
		print("Number of faces detected: {}".format(len(dets)))

		if len(dets) != 1:
			continue

		numlist.append(i)
		typelist.append(pathData['type'][i])

		shape = shape_predictor(img2, dets[0])   # 提取68个特征点
		face_descriptor = face_rec_model.compute_face_descriptor(img2, shape)   # 计算人脸的128维的向量
		print(face_descriptor)
		exit(0)

		veclist.append([x for x in face_descriptor])

	templist = []
	for i in range(128):
		temp = []
		for item in veclist:
			temp.append(item[i])
		templist.append(temp)

	res = pd.DataFrame()
	res['num'] = numlist
	res['type'] = typelist
	for i in range(128):
		res['key' + str(i)] = templist[i]

	res.to_csv('vec.csv', index=False)

if __name__ == '__main__':
	main()