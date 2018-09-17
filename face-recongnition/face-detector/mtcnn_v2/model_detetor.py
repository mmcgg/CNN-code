# coding: utf-8

import numpy as np


class Detector(object):

    """Detector"""

    def __init__(self):
        pass
        


class DetectorMtcnn(Detector):

    """DetectorMtcnn"""

    def __init__(self):
        pass

    def adjust_input(self, in_data):
        """
            adjust the input from (h, w, c) to ( 1, c, h, w) for network input

            Parameters:
            ----------
                in_data: numpy array of shape (h, w, c)
                input data
            Returns:
            -------
                out_data: numpy array of shape (1, c, h, w)
                reshaped array
        """

        if in_data.dtype is not np.dtype('float32'):
            out_data = in_data.astype(np.float32)
        else:
            out_data = in_data

        out_data = out_data.transpose((2,0,1))
        out_data = np.expand_dims(out_data, 0)
        out_data = (out_data - 127.5)*0.0078125

        return out_data

    def nms(self, boxes, overlap_threshold, mode='Union'):
        """
            non max suppression 非极大抑制
            用于去除有重叠的box
            Parameters:
            ----------
                box: numpy array n x 5
                    input bbox array
                overlap_threshold: float number
                    threshold of overlap
                mode: float number
                    how to compute overlap ratio, 'Union' or 'Min'
            Returns:
            -------
                index array of the selected bbox
        """

        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
        #计算每个box的面积、根据socre进行排序
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(score)

        # 循环检测，每次都从最后位置取
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            #获取与其他box重叠的区域的坐标
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # 获取与其他box重叠的区域的面积
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            #计算重叠度
            inter = w * h
            if mode == 'Min':
                overlap = inter / np.minimum(area[i], area[idxs[:last]])
            else:
                overlap = inter / (area[i] + area[idxs[:last]] - inter)

            # 将重叠度高于阈值的box去除掉
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        return pick
