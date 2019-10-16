# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import tensorflow as tf
import parameters as para
from onehotcode import onehotencode


anchors = np.array(para.ANCHORS,dtype=np.float32)
anchors_tensor = tf.convert_to_tensor(anchors,dtype=tf.float32)

def calculate_iou(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
    w = tf.maximum(0.0, tf.minimum(xmax0, xmax1) - tf.maximum(xmin0, xmin1))
    h = tf.maximum(0.0, tf.minimum(ymax0, ymax1) - tf.maximum(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection
    iou = tf.reduce_max([0.0,intersection/union])
    return iou


def smooth_l1(x):
    l1 = (0.5*(x**2))*tf.cast(tf.less(tf.abs(x),1.0),dtype=tf.float32)
    l2 = (tf.abs(x)-0.5)*tf.cast(tf.greater(tf.abs(x),1.0),dtype=tf.float32)
    return l1+l2


def smooth_l1_loss(pred,anchor_cx,anchor_cy,anchor_w,anchor_h,gt):
    g_cx = ((gt[0]-anchor_cx)/anchor_w)*para.X_SCALE
    g_cy = ((gt[1]-anchor_cy)/anchor_h)*para.Y_SCALE
    g_w = tf.log(gt[2]/anchor_w)*para.W_SCALE
    g_h = tf.log(gt[3]/anchor_h)*para.H_SCALE

    pred_cx,pred_cy,pred_w,pred_h = pred[0],pred[1],pred[2],pred[3]
    loss = smooth_l1(pred_cx-g_cx)+smooth_l1(pred_cy-g_cy)+smooth_l1(pred_w-g_w)+smooth_l1(pred_h-g_h)
    return loss
    
    
def merge_loss(pos_loss,conf_loss,num_pos):
    num_neg = tf.cast(para.RATIO*num_pos,tf.int32)
    top_k_loss,_ = tf.nn.top_k(conf_loss,k=num_neg)
    neg_loss = tf.reduce_sum(top_k_loss)
    loss = (pos_loss+neg_loss)/num_pos
    return loss


def ssd_loss(loc,cls,grondtruth):
    losses = []
    num_anchors = len(anchors)
    gt_cls_bg = np.squeeze(onehotencode([para.LABELS.Class_name[para.NUM_CLASSESS-1]+'_*']))
    
    for b in range(para.BATCH_SIZE):
        i = 0
        pos_loss = 0.0
        loc_b = loc[b,:,:]
        cls_b = cls[b,:,:]
        grondtruth_b = grondtruth[b,:,:]
        
        def cond(i,pos_loss,conf_loss):
            boolean = tf.less(i,num_anchors)
            return boolean

        def body(i,pos_loss,conf_loss):
            pos_l = 0.0
            pred_loc = loc_b[i,:]
            pred_conf = cls_b[i,:]
            anchor = anchors_tensor[i,:]
            
            anchor_cx,anchor_cy,anchor_w,anchor_h = anchor[0],anchor[1],anchor[2],anchor[3]
            anchor_xmin = anchor_cx - anchor_w*0.5
            anchor_ymin = anchor_cy - anchor_h*0.5
            anchor_xmax = anchor_cx + anchor_w*0.5
            anchor_ymax = anchor_cy + anchor_h*0.5
        
            for j in range(para.MAX_NUM_GT):
                gt_loc = grondtruth_b[j,0:4]
                gt_cls = grondtruth_b[j,4::]
                gt_xmin = gt_loc[0] - gt_loc[2]*0.5
                gt_ymin = gt_loc[1] - gt_loc[3]*0.5
                gt_xmax = gt_loc[0] + gt_loc[2]*0.5
                gt_ymax = gt_loc[1] + gt_loc[3]*0.5
                
                iou = calculate_iou(anchor_xmin,anchor_ymin,anchor_xmax,anchor_ymax,gt_xmin,gt_ymin,gt_xmax,gt_ymax) # iou  
                loc_l = smooth_l1_loss(pred_loc,anchor_cx,anchor_cy,anchor_w,anchor_h,gt_loc)           # localization loss(positive)
                conf_l = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls,logits=pred_conf)        # confidence loss(positive)
                mask = tf.cast(tf.greater(iou,para.MATCH_IOU),dtype=tf.float32)
                pos_l = pos_l + mask*(conf_l + para.ALPHA*loc_l)
                
            flag = tf.cast(tf.equal(pos_l,0.0),dtype=tf.float32)
            loss = flag*tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls_bg,logits=pred_conf)      # confidence loss(negative)
            conf_loss = conf_loss.write(i,loss)
            pos_loss =  pos_loss + pos_l
            return [i+1,pos_loss,conf_loss]
        
        conf_loss = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)        
        [i,pos_loss,conf_loss] = tf.while_loop(cond,body,loop_vars=[i,pos_loss,conf_loss], parallel_iterations=10,back_prop=True,swap_memory=True)
        conf_loss = conf_loss.stack()
        
        num_pos = tf.cast((num_anchors-tf.count_nonzero(conf_loss, dtype=tf.int32)),tf.float32)
        loss = tf.cond(tf.equal(num_pos,0),lambda:0.0,lambda:merge_loss(pos_loss,conf_loss,num_pos))
        losses.append(loss)
    return tf.reduce_mean(losses)
