# --------------------------------------------------------
# PASCAL VOC Image manipulation detection dataset generator
# Licensed under The MIT License [see LICENSE for details]
# Written by Hangyan Jiang
# --------------------------------------------------------
import os
from random import randint
from PIL import Image
import numpy as np
from lib.datasets.factory import get_imdb
from lib.datasets.xml_op import *
import xml.etree.ElementTree as ET
from shutil import copyfile

DATASET_SIZE = 5011

# 读取图片文件夹，读取图片路径、图片注释信息路径
dataset_path = os.sep.join(['data', 'VOCdevkit2007', 'VOC2007'])
images_path = os.sep.join([dataset_path, 'JPEGImages'])
image_annotation_path = os.sep.join([dataset_path, 'Annotations'])

# 设定生成图片保存的文件夹，设定生成图片路径、生成图片注释信息路径
save_path = os.sep.join(['data', 'DIY_dataset', 'VOC2007'])
save_imgage_path = os.sep.join([save_path, 'JPEGImages'])
save_annotation_path = os.sep.join([save_path, 'Annotations'])

# 读取voc_2007_trainval和其RoI信息，记录其索引
imdb = get_imdb("voc_2007_trainval")
roidb = imdb.roidb

image_index = imdb._load_image_set_index()
seg_index = imdb._load_seg_set_index()


# 生成分割和图片的匹配字典map，关键字为分割索引idx1，值为图片索引idx2
def generate_seg_img_map():
    map = {}
    idx1 = 0
    for i in seg_index:
        idx2 = 0
        for j in image_index:
            if i == j:
                map[idx1] = idx2
            idx2 += 1
        idx1 += 1
    return map


# 生成随机数，范围 0~分割索引最大值
def random_seg_idx():
    return randint(0, len(seg_index)-1)


# 生成随机数，范围 (1, len(s)-2)
def random_obj_idx(s):
    return randint(1, len(s)-2)


# 生成两个随机数，长度分别为高度差和宽度差
def random_obj_loc(img_h, img_w, obj_h, obj_w):
    return randint(0, img_h - obj_h), randint(0, img_w - obj_w)


# 返回不为0的行和列的起始终止值
def find_obj_vertex(mask):
    hor = np.where(np.sum(mask, axis=0) > 0)
    ver = np.where(np.sum(mask, axis=1) > 0)
    return hor[0][0], hor[0][-1], ver[0][0], ver[0][-1]


# 复制filename到savefile，以树的形式记录xmin, ymin, xmax, ymax
def modify_xml(filename, savefile, xmin, ymin, xmax, ymax):

    # 定义实例element，text属性为content
    def create_node(tag, property_map, content):
        element = Element(tag, property_map)
        element.text = content
        return element

    copyfile(filename, savefile)
    tree = ET.parse(savefile)
    root = tree.getroot()
    # 在root中删除object
    for obj in root.findall('object'):
        root.remove(obj)

    # new_obj: tag = 'object', attribute  = {}
    new_obj = Element('object', {})
    new_obj.append(create_node('name', {}, 'tampered'))
    # bndbox: tag = 'bndbox', attribute = {}
    bndbox = Element('bndbox', {})
    bndbox.append(create_node('xmin', {}, str(xmin)))
    bndbox.append(create_node('ymin', {}, str(ymin)))
    bndbox.append(create_node('xmax', {}, str(xmax)))
    bndbox.append(create_node('ymax', {}, str(ymax)))
    new_obj.append(bndbox)
    root.append(new_obj)
    tree.write(savefile)


if __name__ == '__main__':
    map = generate_seg_img_map()
    count = 0
    while count < DATASET_SIZE:
        if count % 100 == 0:
            print('>>> %d / %d' % (count, DATASET_SIZE))
        img_idx = count % len(image_index)  # count % 类中图片数量
        seg_idx = random_seg_idx()
        # img：图片
        # seg：随机到的分割图片
        # seg_img：图片img匹配的分割图片
        img = Image.open(imdb.image_path_at(img_idx))   # base img
        seg = Image.open(imdb.seg_path_at(seg_idx)).convert('P')    # add-on object seg img picked randomly
        seg_img = Image.open(imdb.image_path_at(map[seg_idx]))  # corresponding add-on object original img

        seg_np = np.asarray(seg)
        obj_idx = random_obj_idx(set(seg_np.flatten()))  # randomly pick an obj from seg img
        mask2 = (seg_np == obj_idx)
        min_x, max_x, min_y, max_y = find_obj_vertex(mask2)
        loop_counter = 0
        while(max_x - min_x) * (max_y - min_y) < img.size[0] * img.size[1] * 0.005 or \
                (max_x - min_x) * (max_y - min_y) > img.size[0] * img.size[1] * 0.3 or \
                max_x - min_x >= img.size[0] or max_y - min_y >= img.size[1] or loop_counter > 1000:
            loop_counter += 1
            seg_idx = random_seg_idx()
            seg = Image.open(imdb.seg_path_at(seg_idx)).convert('P')
            seg_img = Image.open(imdb.image_path_at(map[seg_idx]))
            seg_np = np.asarray(seg)
            obj_idx = random_obj_idx(set(seg_np.flatten()))
            mask2 = (seg_np == obj_idx)
            min_x, max_x, min_y, max_y = find_obj_vertex(mask2)
        if loop_counter > 1000:
            continue
        mask2 = mask2[min_y:max_y, min_x:max_x]
        mask = np.stack((mask2, mask2, mask2), axis=2)
        seg_img_np = np.asarray(seg_img).copy()[min_y:max_y, min_x:max_x, :]
        img_np = np.asarray(img).copy()
        loc_y, loc_x = random_obj_loc(img.size[1], img.size[0], max_y - min_y, max_x - min_x)
        img_np[loc_y:loc_y+max_y - min_y, loc_x:loc_x+max_x - min_x, :] = img_np[loc_y:loc_y+max_y - min_y, loc_x:loc_x+max_x - min_x, :] * (1-mask) + seg_img_np * mask
        # seg_img_np *= mask
        new_img = Image.fromarray(img_np, mode='RGB')
        # img.paste(seg_img.resize((100, 100)), (0, 0))
        # img.show()
        # new_img.show()
        new_img.save(os.sep.join([save_imgage_path, image_index[img_idx] + '.jpg']))  # save
        modify_xml(os.sep.join([image_annotation_path, image_index[img_idx] + '.xml']),
                   os.sep.join([save_annotation_path, image_index[img_idx] + '.xml']),
                   loc_x+1, loc_y+1, loc_x+max_x - min_x, loc_y+max_y - min_y)
        count += 1
