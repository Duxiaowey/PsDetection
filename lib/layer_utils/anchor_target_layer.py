# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
    # rpn_cls_score: rpn分类得分
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors / num_anchors
    # 统计平均每个anchor有几个框被选取
    im_info = im_info[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    # 只保留图像范围内的box，过滤掉不在图像范围内的box    
    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # 打标签，首先全贴上-1，即don't care
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # 计算anchor和ground trueth的重叠率
    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # 读取每一行重叠率的最大值
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # 返回与gt重合率最大的索引
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # 若参数为False：将满足负样本阈值的anchor全部标记为0
    # 若参数为True： 将满足负样本阈值且不满足正样本阈值的anchor标签设为0
    if not cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels first so that positive labels can clobber them
        # first set the negatives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # fg label: for each gt, anchor with highest overlap
    # 前景1：对每一个gt框，重叠率最大得检测框标签设为1，即前景
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    # 前景标签2：满足重叠率阈值的检测结果标签打为1
    labels[max_overlaps >= cfg.FLAGS.rpn_positive_overlap] = 1

    if cfg.FLAGS.rpn_clobber_positives:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.FLAGS.rpn_negative_overlap] = 0

    # 如果正样本过多则重采样，使正负样本均衡
    # subsample positive labels if we have too many
    num_fg = int(cfg.FLAGS.rpn_fg_fraction * cfg.FLAGS.rpn_batchsize)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # 对负样本进行同样操作
    # subsample negative labels if we have too many
    num_bg = cfg.FLAGS.rpn_batchsize - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 计算检测RoI与真实RoI的偏移
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # only the positive ones have regression targets
    bbox_inside_weights[labels == 1, :] = np.array(cfg.FLAGS2["bbox_inside_weights"])

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.FLAGS.rpn_positive_weight < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.FLAGS.rpn_positive_weight > 0) &
                (cfg.FLAGS.rpn_positive_weight < 1))
        positive_weights = (cfg.FLAGS.rpn_positive_weight /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.FLAGS.rpn_positive_weight) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # labels
    # 改变label形状
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # 计算bbox大小参数，给出每一个anchor框的输入权重、输出权重
    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


# 反映射，对子集中的item找到原始item，进行参数填充         
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

# 返回偏移量,(target_dx, target_dy, target_dw, target_dh)
def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
