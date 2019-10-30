# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# 边框回归补偿量
def bbox_transform(ex_rois, gt_rois):
    # ex_rois:内部anchor; gt_rois:与anchor最匹配的GT
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    # 得到ex_rois的宽度、高度
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    # 得到ex_rois的宽度、高度的中心

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    # 得到gt_rois的宽度、高度，并在gt_rois上进行变换得到gt_rois的宽度、高度中心

    '''
    边框回归的偏移量：
    target_dx = (x - xa) / wa ; target_dy = (y - ya) / ha
    target_dw = log(w / wa)   ; target_dh = log(h / ha)
    '''
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


# 边框回归，返回回归后的顶点坐标
def bbox_transform_inv(boxes, deltas):
    """
    Return pred_boxes' vertex coordinates

    Parameter
    ---------
    boxes: ndarray
        记录原始box的顶点坐标, [x1, y1, x2, y2]
    deltas: ndarray
        记录偏移量

    Return
    ------
    pred_boxes: ndarray
        记录原始box经回归后加上偏移量的顶点坐标,[x1, y1, x2, y2]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    # 计算输入boxes的宽、高、中心
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]    # 0::4表示从0开始每隔4个取一个
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    '''
    计算回归后的x,y,w,h
    为bbox_transform的返过程
    pred_x = xa + wa * target_dx
    pred_y = ya + ha * target_dy
    w = e^(target_w) * wa
    h = e^(target_h) * ha
    '''
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

# 将proposals的边界限制在图片内
def clip_boxes(boxes, im_shape):
    """
    Return clipped boxes

    Parameter
    ---------
    boxes: ndarray
        boxes before clip, [x1, y1, x2, y2]
    im_shape: list
        image shape, [height, width]

    Return
    ------
    boxes: ndarray
        boxes after clip
    """
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
