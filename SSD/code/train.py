# -*- coding: utf-8 -*-
"""
@author: LiXiaoGang        
"""
from __future__ import division

import os
import sys
import shutil
import tensorflow as tf
import parameters as para
from readbatch import mini_batch
from SSDLoss import ssd_loss
from SSDNet import SSDmobilenetv1
from postprocessing import nms,box_decode,save_instance


def net_placeholder(batch_size=None):
    Input = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size,para.INPUT_SIZE[0],para.INPUT_SIZE[1],para.CHANNEL],name='Input')
    
    groundtruth = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size,para.MAX_NUM_GT,4+para.NUM_CLASSESS],name='Label')
    
    isTraining = tf.placeholder(tf.bool,name='Batch_norm')
    return Input,groundtruth,isTraining


def training_net():
    image,groundtruth,isTraining = net_placeholder(batch_size=None)
    loc,cls = SSDmobilenetv1(image,isTraining)
    loss = ssd_loss(loc,cls,groundtruth)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):        
        train_step = tf.train.AdamOptimizer(para.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=5)
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter(os.path.join(para.PATH,'model'), sess.graph)
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        # restore model 
        if para.RESTORE_MODEL:
            if not os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH):
                print('Model does not exist！')
                sys.exit()
            ckpt = tf.train.get_checkpoint_state(para.CHECKPOINT_MODEL_SAVE_PATH)
            model = ckpt.model_checkpoint_path.split('\\')[-1]
            Saver.restore(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,'.\\'+ model))
            print('Successfully restore model:',model)
        
        for i in range(para.TRAIN_STEPS):
            batch = mini_batch(i,para.BATCH_SIZE,'train')
            feed_dict = {image:batch['image'], groundtruth:batch['groundtruth'], isTraining:True}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            print('===>Step %d: loss = %g' % (i,loss_))

            # evaluate and save checkpoint
            if i % 250 == 0:
                write_instance_dir = os.path.join(para.PATH,'pic')
                if not os.path.exists(write_instance_dir):os.mkdir(write_instance_dir)
                j = 0
                while True:
                    
                    batch = mini_batch(j,1,'val')
                    feed_dict = {image:batch['image'],isTraining:False}
                    location,confidence = sess.run([loc,cls],feed_dict=feed_dict)
                    predictions = {'location':location,'confidence':confidence}
                    pred_output = box_decode(predictions,batch['image_name'])
                    pred_output = nms(pred_output,para.NMS_THRESHOLD)
                    
                    if j < min(10,batch['image_num']):save_instance(pred_output)
                    if j == batch['image_num']-1:break
                    j += 1
                
                if os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                    shutil.rmtree(para.CHECKPOINT_MODEL_SAVE_PATH)
                Saver.save(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,para.MODEL_NAME))
            

def main():
    training_net()
     
if __name__ == '__main__':
    main()