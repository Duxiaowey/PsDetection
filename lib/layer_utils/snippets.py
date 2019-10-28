# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ 
    Return anchors and its length.

    A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
    将anchor从特征图上映射到原图上

    Parameters
    ----------
    height, width: int32
        anchor的高度、宽度;

    feat_stride: list
        放大倍数,VGG16时,feat_stride=[16, ];

    anchor_scales, anchor_ratios: float
        anchor大小和宽高比的设定值

    Returns
    -------
    anchors : ndarray
        映射到原图上的anchor，shape=((number of anchors) x width x height, 4)
    length : int32
        anchors.shape[0] = (number of anchors) x width x height
        
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    # A =  anchor的数量 = len(anchor_ritio) * len(anchor_scale) = 9

    # 将特征图的宽、高以16倍延申至原图：经VGG16后得到的特征图大小缩小了16倍，故延申以还原
    shift_x = np.arange(0, width) * feat_stride
    # shift_x = [0, 1*16, ……, width*16], shape=[width, ]
    shift_y = np.arange(0, height) * feat_stride
    # shift_y = [0, 1*16, ……, height*16], shape=[height, ]

    # 生成原图网格点
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 将shift_x, shift_y生成形状相同的矩阵, 形为(height, width)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # 将shift_x, shift_y拉平后堆叠，然后转置
    # shift_x(), shift_y.ravel(): (width*height, ); shifts: (4, width*height)^T = (width*height, 4)
    K = shifts.shape[0]
    # K = width x height
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])
    # length = (number of anchors) x width x height 

    return anchors, length
