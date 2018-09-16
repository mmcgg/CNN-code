# coding: utf-8

import os
import mxnet as mx
import numpy as np
import math
import cv2
from itertools import repeat
from model import DetectorMtcnn

class MtcnnDetector(DetectorMtcnn):
    """
        多联级任务人脸检测及对齐
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
    """
    def __init__(self,
                 model_folder='.',
                 minsize = 20,
                 threshold = [0.6, 0.7, 0.8],
                 factor = 0.709,
                 ctx=mx.cpu()):
        """
            Initialize the detector

            Parameters:
            ----------
                model_folder : 模型存放地址
                minsize : 脸的最小检测尺寸
                threshold : detect threshold for 3 stages
                factor: 图像金字塔的scale

        """

        # 加载四个模型
        models = ['det1', 'det2', 'det3','det4']
        models = [ os.path.join(model_folder, f) for f in models]
        
        self.PNet = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
        self.RNet = mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.ONet = mx.model.FeedForward.load(models[2], 1, ctx=ctx)
        self.LNet = mx.model.FeedForward.load(models[3], 1, ctx=ctx)

        self.minsize = float(minsize)
        self.factor = float(factor)
        self.threshold = threshold


    def convert_to_square(self, bbox):
        """
            将bbox转换成正方形
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h,w)
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            标准化 bboxes

        Parameters:
        ----------
            bbox: numpy array, shape n x 5 原始图像的位置
            reg:  numpy array, shape n x 4 新图像的相对位置
        Returns:
        -------
            调整过后的位置
        """
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox[:, 0:4] = bbox[:, 0:4] + aug
        return bbox

 
    def pad(self, bboxes, w, h):
        """
            处理box，剪裁有超出原图像的部分
        Parameters:
        ----------
            bboxes: numpy array, n x 5
            w: 原图像的宽度
            h: 原图像的长度
        Returns :
        ----------
            dy, dx : 目标图起点
            edy, edx : 目标图终点
            y, x : 原图像起点
            ex, ey : 原图像终点
            tmph, tmpw: box的高度、宽度
        """
        #box的width和height
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        #原图中 box的起始点和终点
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        #目标图中 box的起始点和终点（默认为(0, 0, width, height)）
        dx, dy = np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1

        #如果终点横坐标在图之外时
        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        #如果终点纵坐标在图之外时
        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        #如果起点横坐标在图之外时
        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        #如果起点纵坐标在图之外时
        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_first_stage(self, img, net, scale, threshold):
        """
            运行 PNet for first stage
        
        Parameters:
        ----------
            img: numpy array, bgr order
                input image
            scale: float number
                input image 缩放尺度 
            net: PNet
                worker
        Returns:
        -------
            total_boxes : bboxes
        """

        #图片进行缩放
        height, width, _ = img.shape
        hs = int(math.ceil(height * scale))
        ws = int(math.ceil(width * scale))
        im_data = cv2.resize(img, (ws,hs))
        
        #调整图片输入顺序
        input_buf = self.adjust_input(im_data)
        #预测
        output = net.predict(input_buf)
        # output[0] box
        # output[1] face,output[1][0,1,:,:] 是脸的可能性

        #将现在得到的位置转换为原来尺寸上的位置，前4列为原来尺寸的位置，5为分数，后4列为对应位置上的数值
        boxes = self.generate_bbox(output[1][0,1,:,:], output[0], scale, threshold)

        if boxes.size == 0:
            return None

        # 对位置进行去重叠
        pick = self.nms(boxes[:,0:5], 0.5, mode='Union')
        boxes = boxes[pick]

        return boxes

    def generate_bbox(self, map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: numpy array , n x m x 1
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12

        #判断是否是脸
        t_index = np.where(map>threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                                  np.round((stride*t_index[0]+1)/scale),
                                  np.round((stride*t_index[1]+1+cellsize)/scale),
                                  np.round((stride*t_index[0]+1+cellsize)/scale),
                                  score,
                                  reg])
        return boundingbox.T



    def first_stage(self, img):

        #############################################
        # 第一步 PNet
        #############################################

        # check
        if img is None:
            return None

        if len(img.shape) != 3:
            return None

        # 获取有效的缩放比例
        MIN_DET_SIZE = 12
        height, width, _ = img.shape
        minl = min(height, width)

        scales = []
        m = MIN_DET_SIZE/self.minsize
        minl *= m
        factor_count = 0
        while minl > MIN_DET_SIZE:
            scales.append(m*self.factor**factor_count)
            minl *= self.factor
            factor_count += 1
        
        # 开始检测
        total_boxes = [self.detect_first_stage(img, self.PNet, ival, self.threshold[0]) for ival in scales]
        
        # 尺寸过小时，可能没有检测结果 
        total_boxes = [ i for i in total_boxes if i is not None]

        if len(total_boxes) == 0:
            return None

        total_boxes = np.vstack(total_boxes)

        if total_boxes.size == 0:
            return None

        # 对位置进行去重叠
        pick = self.nms(total_boxes[:, 0:5], 0.7, 'Union')
        total_boxes = total_boxes[pick]

        #获取box的width、height
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1

        # refine the bboxes
        total_boxes = np.vstack([total_boxes[:, 0]+total_boxes[:, 5] * bbw,
                                 total_boxes[:, 1]+total_boxes[:, 6] * bbh,
                                 total_boxes[:, 2]+total_boxes[:, 7] * bbw,
                                 total_boxes[:, 3]+total_boxes[:, 8] * bbh,
                                 total_boxes[:, 4]
                                 ])

        total_boxes = total_boxes.T
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        return total_boxes

    def second_stage(self, img, first_boxes):

        #############################################
        # second stage RNet
        #############################################

        total_boxes = first_boxes
        height, width, _ = img.shape

        num_box = total_boxes.shape[0]

        #对超出图片边界的box进行调整
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, width, height)

        #RNet input shape is (3, 24, 24)
        input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.float32)

        #赋值(先填充，在resize)
        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = self.adjust_input(cv2.resize(tmp, (24, 24)))

        #进行预测
        output = self.RNet.predict(input_buf)
        # output[0] box
        # output[1] 脸

        # 过滤分数小于阈值的
        passed = np.where(output[1][:, 1] > self.threshold[1])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[1][passed, 1].reshape((-1,))
        reg = output[0][passed]

        # nms
        pick = self.nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick]
        total_boxes = self.calibrate_box(total_boxes, reg[pick])
        total_boxes = self.convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        return total_boxes


    def third_stage(self, img, second_boxes):

        #############################################
        # third stage ONet
        #############################################

        total_boxes = second_boxes
        height, width, _ = img.shape

        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, width, height)
        # ONet input shape (3, 48, 48)
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i]+1, dx[i]:edx[i]+1, :] = img[y[i]:ey[i]+1, x[i]:ex[i]+1, :]
            input_buf[i, :, :, :] = self.adjust_input(cv2.resize(tmp, (48, 48)))

        output = self.ONet.predict(input_buf)

        # output[0] landmark
        # output[1] box
        # output[2] 脸

        # 过滤分数小于阈值的
        passed = np.where(output[2][:, 1] > self.threshold[2])
        total_boxes = total_boxes[passed]

        if total_boxes.size == 0:
            return None

        total_boxes[:, 4] = output[2][passed, 1].reshape((-1,))

        reg = output[1][passed]
        points = output[0][passed]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = self.calibrate_box(total_boxes, reg)
        pick = self.nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[pick]
        points = points[pick]

        return total_boxes, points

    def extend_stage(self, img, third_boxes, third_points):

        #############################################
        # extended stage
        #############################################

        total_boxes = third_boxes
        points = third_points

        num_box = total_boxes.shape[0]
        height, width, _ = img.shape

        patchw = np.maximum(total_boxes[:, 2]-total_boxes[:, 0]+1, total_boxes[:, 3]-total_boxes[:, 1]+1)
        patchw = np.round(patchw*0.25)

        # make it even
        patchw[np.where(np.mod(patchw,2) == 1)] += 1

        input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.float32)
        for i in range(5):
            x, y = points[:, i], points[:, i+5]
            x, y = np.round(x-0.5*patchw), np.round(y-0.5*patchw)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(np.vstack([x, y, x+patchw-1, y+patchw-1]).T, width, height)
            for j in range(num_box):
                tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
                tmpim[dy[j]:edy[j]+1, dx[j]:edx[j]+1, :] = img[y[j]:ey[j]+1, x[j]:ex[j]+1, :]
                input_buf[j, i*3:i*3+3, :, :] = self.adjust_input(cv2.resize(tmpim, (24, 24)))

        output = self.LNet.predict(input_buf)

        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            tmp_index = np.where(np.abs(output[k]-0.5) > 0.35)
            output[k][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(points[:, k] - 0.5*patchw) + output[k][:, 0]*patchw
            pointy[:, k] = np.round(points[:, k+5] - 0.5*patchw) + output[k][:, 1]*patchw

        points = np.hstack([pointx, pointy])
        points = points.astype(np.int32)

        return points







