# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import parameters as para


alpha = para.ALPHA
def depthwise_separable_conv2d(Input,stride,filters,alpha,mode,name):
    with tf.variable_scope(name):
        # depthwish layer
        net = tf.contrib.layers.separable_conv2d(Input,
                                                 num_outputs=None,
                                                 kernel_size=[3,3],
                                                 depth_multiplier=1,
                                                 stride=stride,
                                                 padding='SAME',activation_fn=None,biases_initializer=None)  
        net = tf.layers.batch_normalization(net,training=mode,name='batchnorm1')
        net = tf.nn.relu(net,name='relu1')
        
        # pointwise layer
        net = tf.layers.conv2d(net,int(filters*alpha),kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=False)
        net = tf.layers.batch_normalization(net,training=mode,name='batchnorm2')
        net = tf.nn.relu(net,name='relu2')                    
    return net


def conv2d(Input,filters,kernel_size,strides,alpha,mode,name):
    net = tf.layers.conv2d(Input,int(filters*alpha),kernel_size,strides,padding='same',activation=None,use_bias=False,name='conv_'+name)
    net = tf.layers.batch_normalization(net,training=mode,name='batchnorm_'+name)
    net = tf.nn.relu(net,name='relu_'+name)
    return net
    
 
def SSDmobilenetv1(clip_X,mode):
    '''
    Implementation of Mobilenet-v1.
    Architecture: https://arxiv.org/abs/1704.04861;https://arxiv.org/abs/1512.02325
    '''
    with tf.variable_scope('FeatureExtractor'):
        with tf.variable_scope('MobilenetV1'):
            with tf.variable_scope('Standard_conv'):                
                net = conv2d(clip_X,filters=32,kernel_size=(3,3),strides=(2,2),alpha=alpha,mode=mode,name=str(0))
                
            with tf.variable_scope('Depthwise_separable_conv'):
                net = depthwise_separable_conv2d(net,[1,1],64,alpha,mode,name='DW1')
                net = depthwise_separable_conv2d(net,[2,2],128,alpha,mode,name='DW2')
                net = depthwise_separable_conv2d(net,[1,1],128,alpha,mode,name='DW3')
                net = depthwise_separable_conv2d(net,[2,2],256,alpha,mode,name='DW4')
                net = depthwise_separable_conv2d(net,[1,1],256,alpha,mode,name='DW5')
                net = depthwise_separable_conv2d(net,[2,2],512,alpha,mode,name='DW6')
                net = depthwise_separable_conv2d(net,[1,1],512,alpha,mode,name='DW7')
                net = depthwise_separable_conv2d(net,[1,1],512,alpha,mode,name='DW8')
                net = depthwise_separable_conv2d(net,[1,1],512,alpha,mode,name='DW9')
                net = depthwise_separable_conv2d(net,[1,1],512,alpha,mode,name='DW10')
                net1 = depthwise_separable_conv2d(net,[1,1],512,alpha,mode,name='DW11') # Output size:19 x 19
                net = depthwise_separable_conv2d(net1,[2,2],1024,alpha,mode,name='DW12')
                net2 = depthwise_separable_conv2d(net,[1,1],1024,alpha,mode,name='DW13') # Output size:10 x 10
                
        with tf.variable_scope('AdditionalLayers'):                
            net = conv2d(net2,filters=256,kernel_size=(1,1),strides=(1,1),alpha=alpha,mode=mode,name=str(1))
            net3 = conv2d(net,filters=512,kernel_size=(3,3),strides=(2,2),alpha=alpha,mode=mode,name=str(2)) # Output size:5 x 5
            net = conv2d(net3,filters=128,kernel_size=(1,1),strides=(1,1),alpha=alpha,mode=mode,name=str(3))
            net4 = conv2d(net,filters=256,kernel_size=(3,3),strides=(2,2),alpha=alpha,mode=mode,name=str(4)) # Output size:3 x 3
            net = conv2d(net4,filters=128,kernel_size=(1,1),strides=(1,1),alpha=alpha,mode=mode,name=str(5))
            net5 = conv2d(net,filters=256,kernel_size=(3,3),strides=(2,2),alpha=alpha,mode=mode,name=str(6)) # Output size:2 x 2
            net = conv2d(net5,filters=64,kernel_size=(1,1),strides=(1,1),alpha=alpha,mode=mode,name=str(7))
            net6 = conv2d(net,filters=128,kernel_size=(3,3),strides=(2,2),alpha=alpha,mode=mode,name=str(8)) # Output size:1 x 1
                
    with tf.variable_scope('ObjectDetector'):
        with tf.variable_scope('Detetctor1'):
            with tf.variable_scope('Location'):
                filters = 4*6
                loc1 = tf.layers.conv2d(net1,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                loc1 = tf.reshape(loc1,shape=[-1,para.FEATURE_MAPS[0][0]*para.FEATURE_MAPS[0][1]*6,4])
            with tf.variable_scope('Classification'):
                filters = (para.NUM_CLASSESS)*6
                cls1 = tf.layers.conv2d(net1,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                cls1 = tf.reshape(cls1,shape=[-1,para.FEATURE_MAPS[0][0]*para.FEATURE_MAPS[0][1]*6,(para.NUM_CLASSESS)])
        with tf.variable_scope('Detetctor2'):
            with tf.variable_scope('Location'):
                filters = 4*6
                loc2 = tf.layers.conv2d(net2,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                loc2 = tf.reshape(loc2,shape=[-1,para.FEATURE_MAPS[1][0]*para.FEATURE_MAPS[1][1]*6,4])
            with tf.variable_scope('Classification'):
                filters = (para.NUM_CLASSESS)*6
                cls2 = tf.layers.conv2d(net2,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                cls2 = tf.reshape(cls2,shape=[-1,para.FEATURE_MAPS[1][0]*para.FEATURE_MAPS[1][1]*6,(para.NUM_CLASSESS)])
        with tf.variable_scope('Detetctor3'):
            with tf.variable_scope('Location'):
                filters = 4*6
                loc3 = tf.layers.conv2d(net3,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                loc3 = tf.reshape(loc3,shape=[-1,para.FEATURE_MAPS[2][0]*para.FEATURE_MAPS[2][1]*6,4])
            with tf.variable_scope('Classification'):
                filters = (para.NUM_CLASSESS)*6
                cls3 = tf.layers.conv2d(net3,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                cls3 = tf.reshape(cls3,shape=[-1,para.FEATURE_MAPS[2][0]*para.FEATURE_MAPS[2][1]*6,(para.NUM_CLASSESS)])
        with tf.variable_scope('Detetctor4'):
            with tf.variable_scope('Location'):
                filters = 4*6
                loc4 = tf.layers.conv2d(net4,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                loc4 = tf.reshape(loc4,shape=[-1,para.FEATURE_MAPS[3][0]*para.FEATURE_MAPS[3][1]*6,4])
            with tf.variable_scope('Classification'):
                filters = (para.NUM_CLASSESS)*6
                cls4 = tf.layers.conv2d(net4,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                cls4 = tf.reshape(cls4,shape=[-1,para.FEATURE_MAPS[3][0]*para.FEATURE_MAPS[3][1]*6,(para.NUM_CLASSESS)])
        with tf.variable_scope('Detetctor5'):
            with tf.variable_scope('Location'):
                filters = 4*6
                loc5 = tf.layers.conv2d(net5,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                loc5 = tf.reshape(loc5,shape=[-1,para.FEATURE_MAPS[4][0]*para.FEATURE_MAPS[4][1]*6,4])
            with tf.variable_scope('Classification'):
                filters = (para.NUM_CLASSESS)*6
                cls5 = tf.layers.conv2d(net5,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                cls5 = tf.reshape(cls5,shape=[-1,para.FEATURE_MAPS[4][0]*para.FEATURE_MAPS[4][1]*6,(para.NUM_CLASSESS)])
        with tf.variable_scope('Detetctor6'):
            with tf.variable_scope('Location'):
                filters = 4*5
                loc6 = tf.layers.conv2d(net6,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                loc6 = tf.reshape(loc6,shape=[-1,para.FEATURE_MAPS[5][0]*para.FEATURE_MAPS[5][1]*5,4])
            with tf.variable_scope('Classification'):
                filters = (para.NUM_CLASSESS)*5
                cls6 = tf.layers.conv2d(net6,filters,kernel_size=(1,1),strides=(1,1),padding='same',activation=None,use_bias=True)
                cls6 = tf.reshape(cls6,shape=[-1,para.FEATURE_MAPS[5][0]*para.FEATURE_MAPS[5][1]*5,(para.NUM_CLASSESS)])
        
        loc = tf.concat([loc1,loc2,loc3,loc4,loc5,loc6],axis=1,name='localization')
        cls = tf.concat([cls1,cls2,cls3,cls4,cls5,cls6],axis=1,name='confidence')
    return loc,cls
