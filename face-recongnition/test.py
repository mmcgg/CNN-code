# coding: utf-8

import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

if __name__ == '__main__':

    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), accurate_landmark = False)

    img = cv2.imread('test2.jpg')
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()

    # run detector
    results = detector.first_stage(img)

    results2 = detector.second_stage(img, results)

    results3, point = detector.third_stage(img, results2)

    point1 = detector.extend_stage(img, results3, point)

    if results is not None:

        for box in results:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255))

        cv2.imshow("detection result", img)

    if results2 is not None:

        for box in results2:
            cv2.rectangle(img2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255))

        cv2.imshow("detection result2", img2)

    if results3 is not None and point is not None:

        for box in results3:
            cv2.rectangle(img3, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255))

        for p in point:
            for i in range(5):
                cv2.circle(img3, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        cv2.imshow("detection result3", img3)

    if point1 is not None:
        for p in point1:
            for i in range(5):
                cv2.circle(img4, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
        cv2.imshow("detection result4", img4)

    
    cv2.waitKey(0)


