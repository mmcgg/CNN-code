# coding: utf-8
# YuanYang
import math
import cv2
import numpy as np


def nms(boxes, overlap_threshold, mode='Union'):
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

def adjust_input(in_data):
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

def generate_bbox(map, reg, scale, threshold):
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


def detect_first_stage(img, net, scale, threshold):
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
    input_buf = adjust_input(im_data)
    #预测
    output = net.predict(input_buf)
    # output[0] box
    # output[1] face,output[1][0,1,:,:] 是脸的可能性

    #将现在得到的位置转换为原来尺寸上的位置，前4列为原来尺寸的位置，5为分数，后4列为对应位置上的数值
    boxes = generate_bbox(output[1][0,1,:,:], output[0], scale, threshold)

    if boxes.size == 0:
        return None

    # 对位置进行去重叠
    pick = nms(boxes[:,0:5], 0.5, mode='Union')
    boxes = boxes[pick]

    return boxes
