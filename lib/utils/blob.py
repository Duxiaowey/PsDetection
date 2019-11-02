# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


# 将图片转换为适合网络输入的形式
def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


# 将图片减掉均值后resize为统一尺寸
def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """
    Mean subtract and scale an image for use in a blob.

    Returns
    -------
    im: ndarray
        im = im - mean
    im_scale: float
        target_size/im_size_min 或 max_size/im_size_max
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2]) # 图片维度的最大值, 如shape=[3,6,2], 则im_size_min=2
    im_size_max = np.max(im_shape[0:2]) # 图片维度的最小值, 如shape=[3,6,2], 则im_size_min=6
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
