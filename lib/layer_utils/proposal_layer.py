# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms

# 利用训练好的rpn网络生成区域候选框
def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    '''
    cfg_key: Train/Test;
    _feat_stride: [16, ]
    anchors: generate_anchors_pre(height, width, feat_stride, anchor_scales(面积大小)=(8, 16, 32), anchor_ratios(宽高比)=(0.5, 1, 2))
    '''
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    # 训练阶段
    if cfg_key == "TRAIN":
        # Number of top scoring boxes to keep before apply NMS to RPN proposals
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n # 12000
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n # 2000
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh # 0.7
    # 测试阶段
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n # 6000
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n # 300
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh # 0.7

    im_info = im_info[0]
    # Get the scores and bounding boxes
    # 得到RPN预测框属于前景的分数(前num_anchor个是属于背景的概率，后num_anchor个是属于前景的概率)
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
