# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse


#将config，imdb，voc_eval以模块形式导入。
from lib.config import config as cfg
from lib.datasets.imdb import imdb
from .voc_eval import voc_eval


class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        # data_path: self._devkit_path + VOC2007
        # Example：VOCdevkit2007/VOC2007/
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        # _class_to_ind:字典，记录每个类对应的图片数量
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._seg_index = self._load_seg_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    #输入i，调用load_image_set_index函数和image_path_from_index函数，返回第i张图片的路径
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    #输入Index(图片文件名），输出该index对应图片的路径
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example path to image_path:(index=0000432)
        # self._devkit_path + /VOCdevkit2007/VOC2007/JPEGImages/0000432.jpg
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    #输入i，调用load_seg_set_index函数和seg_path_from_index函数，返回第i张图片的分割图片路径
    def seg_path_at(self, i):
        """
        Return the absolute path to image i segmentation ground truth.
        """
        return self.seg_path_from_index(self._seg_index[i])

    def seg_path_from_index(self, index):
        """
        Construct an segmentation ground truth image path from the image's "index" identifier.
        """
        #Example path to image_path:(index=0000432)
        #self._devkit_path + /VOCdevkit2007/VOC2007/SegmentationObject/0000432.png
        image_path = os.path.join(self._data_path, 'SegmentationObject',
                                  index + '.png')
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    #输出image_index，为列表，以列表元素形式分别记录该class输入的数据集中每张图片的图片信息
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt
        # 该Example文件中记录的为某一子数据集
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    #输出seg_index，为列表，记录每张图片的分割信息
    def _load_seg_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Segmentation/trainval.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Segmentation',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            seg_index = [x.strip() for x in f.readlines()]
        return seg_index

    #返回路径：Image_manipulation_detection-master/VOCdevkitxxxx，xxxx为年份
    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.FLAGS2["data_dir"], 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # Example path to cache_file:
        # Image_manipulation_detection-master + self.name(imdb中输入)/_gt_roidb.pkl

        # 返回gt_roidb，列表，记录每个index对应的分割信息。分割信息为字典，在函数self._load_pascal_annotation中定义，记录'boxes'，'gt_classes'，'gt_overlaps'，'flipped'，'seg_areas'。
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    # def rpn_roidb(self):
    #     if int(self._year) == 2007 or self._image_set != 'test':
    #         gt_roidb = self.gt_roidb()
    #         rpn_roidb = self._load_rpn_roidb(gt_roidb)
    #         roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    #     else:
    #         roidb = self._load_rpn_roidb(None)
    #
    #     return roidb

    # def _load_rpn_roidb(self, gt_roidb):
    #     filename = self.config['rpn_file']
    #     print('loading {}'.format(filename))
    #     assert os.path.exists(filename), \
    #         'rpn data not found at: {}'.format(filename)
    #     with open(filename, 'rb') as f:
    #         box_list = pickle.load(f)
    #     return self.create_roidb_from_box_list(box_list, gt_roidb)

    #返回字典
    #{"boxes":      零矩阵，shape=(num_obj，4)
    # "gt_classes": 矩阵，shape=(num_obj，1)，记录类名称对应的数量
    # "gt_overlaps":矩阵，shape=(num_obj, 类别个数)，在(Object_index, 该Index对应的数量)处为1， 其他为0
    # "flipped":    False
    # "seg_area":   矩阵，shape=(num_obj，1)，记录分割区域面积大小}
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            # non_diff_objs列表记录difficult为0的obj
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))

            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    # def _get_comp_id(self):
    #     comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
    #                else self._comp_id)
    #     return comp_id

    # def _get_voc_results_file_template(self):
    #     # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    #     filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #     path = os.path.join(
    #         self._devkit_path,
    #         'results',
    #         'VOC' + self._year,
    #         'Main',
    #         filename)
    #     return path

    # def _write_voc_results_file(self, all_boxes):
    #     for cls_ind, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         print('Writing {} VOC results file'.format(cls))
    #         filename = self._get_voc_results_file_template().format(cls)
    #         with open(filename, 'wt') as f:
    #             for im_ind, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in range(dets.shape[0]):
    #                     f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                             format(index, dets[k, -1],
    #                                    dets[k, 0] + 1, dets[k, 1] + 1,
    #                                    dets[k, 2] + 1, dets[k, 3] + 1))

    # def _do_python_eval(self, output_dir='output'):
    #     annopath = self._devkit_path + '\\VOC' + self._year + '\\Annotations\\' + '{:s}.xml'
    #     imagesetfile = os.path.join(
    #         self._devkit_path,
    #         'VOC' + self._year,
    #         'ImageSets',
    #         'Main',
    #         self._image_set + '.txt')
    #     cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    #     aps = []
    #     # The PASCAL VOC metric changed in 2010
    #     use_07_metric = True if int(self._year) < 2010 else False
    #     print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    #     if not os.path.isdir(output_dir):
    #         os.mkdir(output_dir)
    #     for i, cls in enumerate(self._classes):
    #         if cls == '__background__':
    #             continue
    #         filename = self._get_voc_results_file_template().format(cls)
    #         rec, prec, ap = voc_eval(
    #             filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
    #             use_07_metric=use_07_metric)
    #         aps += [ap]
    #         print(('AP for {} = {:.4f}'.format(cls, ap)))
    #         with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
    #             pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #     print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    #     print('~~~~~~~~')
    #     print('Results:')
    #     for ap in aps:
    #         print(('{:.3f}'.format(ap)))
    #     print(('{:.3f}'.format(np.mean(aps))))
    #     print('~~~~~~~~')
    #     print('')
    #     print('--------------------------------------------------------------')
    #     print('Results computed with the **unofficial** Python eval code.')
    #     print('Results should be very close to the official MATLAB eval code.')
    #     print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    #     print('-- Thanks, The Management')
    #     print('--------------------------------------------------------------')

    # def _do_matlab_eval(self, output_dir='output'):
    #     print('-----------------------------------------------------')
    #     print('Computing results with the official MATLAB eval code.')
    #     print('-----------------------------------------------------')
    #     path = os.path.join(cfg.FLAGS2["root_dir"], 'lib', 'datasets',
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    #         .format(self._devkit_path, self._get_comp_id(),
    #                 self._image_set, output_dir)
    #     print(('Running:\n{}'.format(cmd)))
    #     status = subprocess.call(cmd, shell=True)

    # def evaluate_detections(self, all_boxes, output_dir):
    #     self._write_voc_results_file(all_boxes)
    #     self._do_python_eval(output_dir)
    #     if self.config['matlab_eval']:
    #         self._do_matlab_eval(output_dir)
    #     if self.config['cleanup']:
    #         for cls in self._classes:
    #             if cls == '__background__':
    #                 continue
    #             filename = self._get_voc_results_file_template().format(cls)
    #             os.remove(filename)

    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc

    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
