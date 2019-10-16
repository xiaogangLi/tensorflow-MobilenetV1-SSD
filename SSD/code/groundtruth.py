# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import parameters as para
from parse import parse_size
from parse import parse_object
from onehotcode import onehotencode


def get_groundtruth(xml):
    size_dict = parse_size(xml)
    rw = 1.0*para.INPUT_SIZE[0]/size_dict['width']
    rh = 1.0*para.INPUT_SIZE[1]/size_dict['height']
    
    gt = np.zeros([para.MAX_NUM_GT,4+para.NUM_CLASSESS],dtype=np.float32)
    object_list = parse_object(xml)
    j = 0
    for box in object_list:
        box_class = box['classes']
        xmin =  box['xmin']*rw
        ymin =  box['ymin']*rh
        xmax =  box['xmax']*rw
        ymax =  box['ymax']*rh
        
        cx = (xmin + (xmax-xmin)*0.5)/para.INPUT_SIZE[0]
        cy = (ymin + (ymax-ymin)*0.5)/para.INPUT_SIZE[1]
        w = 1.0*(xmax-xmin)/(para.INPUT_SIZE[0])
        h = 1.0*(ymax-ymin)/(para.INPUT_SIZE[1])
    
        class_onehotcode = np.squeeze(onehotencode([box_class+'_*']))
        box = np.hstack((np.array([cx,cy,w,h],dtype=np.float32),class_onehotcode))
                 
        gt[j,:] = box
        j = j + 1
        if j == para.MAX_NUM_GT:break
    
    for i in range(j,para.MAX_NUM_GT):
        if i == para.MAX_NUM_GT:break
        gt[i,:] = box
    return {'groundtruth':gt}
