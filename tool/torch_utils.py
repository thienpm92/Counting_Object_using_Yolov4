import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils 





def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def detect_yolo(model, img, conf_thresh, nms_thresh, object_type,class_names=None, use_cuda=True):
    model.eval()
    img_cv = img.copy()
    t0 = time.time()
    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img = torch.as_tensor(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        # img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        img = torch.as_tensor(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    
    t1 = time.time()

    output = model(img)

    t2 = time.time()

    # print('-----------------------------------')
    # print('           Preprocess : %f' % (t1 - t0))
    # print('      Model Inference : %f' % (t2 - t1))
    # print('-----------------------------------')

    boxes =  utils.post_processing(img, conf_thresh, nms_thresh, output)
    boxes = boxes[0]

    width = img_cv.shape[1]
    height = img_cv.shape[0]
    objects = []
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            if class_names[cls_id]==object_type:
                box = [x1, y1, x2 - x1, y2 - y1]
                objects.append(box)
    return objects
