# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    '''
    Return 满足条件需要保留的bbox
    
    非极大值抑制算法
    将scores排序，选取得分最高的框，其他与被选中框有明显重叠的框被抑制，不断递归应用于其余检测框

    Parameters
    -----------
    dets: ndarray
        记录边界框信息：[x1, y1, x2, y2, scores]
    thresh: float
        设定的阈值，重叠部分小于该阈值的框被丢弃

    Returns
    -------
    keep: list
        记录非极大值抑制后需要保留的检测框的索引
    '''
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)   # 框面积
    order = scores.argsort()[::-1]  # [::-1]表示逆序，order为从大到小排序

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 形如x1，将x1中比x1[i]小的全部替换为x1[i]
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # 修改后的宽度，若为负则取零
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h   # list，记录重叠区域面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]   # inds记录ovr < thresh的位置.[0]表示将其从tuple转成array
        order = order[inds + 1] # ovr记录时去掉了order中的第一个元素元素i，所以这里+1读取

    return keep
