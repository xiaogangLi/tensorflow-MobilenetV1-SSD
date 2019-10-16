# -*- coding: utf-8 -*-

from __future__ import division

import os
import pandas as pd


# Training para
BATCH_SIZE = 3
LEARNING_RATE = 0.0001
TRAIN_STEPS = 500000
PIC_TYPE = 'jpg'             # the picture format of training images.
RESTORE_MODEL = False

MAX_NUM_GT = 5              # suppose the number of objects per image is at most 5, which can be modified.
NMS_THRESHOLD = 0.45

# Mobilenetv1 para
ALPHA = 1                                   # width multiplier with typical settings of 1,0.75,0.5,0.25.
RHO = 1                                     # resolution multiplier,0 < RHO <= 1.
INPUT_SIZE = (int(300*RHO),int(300*RHO))    # (height,width)
CHANNEL = 3

# Anchcor para
FEATURE_MAPS = [(19,19),(10,10),(5,5),(3,3),(2,2),(1,1)] # when RHO = 1
MIN_SCALE = 0.2
MAX_SCALE = 0.9
ASPECT_RATIOS = [1,2,3,1./2,1./3]
MATCH_IOU = 0.5        # when iou between defalut boxes and groundtruth > 0.5, the default boxes will be seen as positives 
ALPHA = 1
RATIO = 3/1           # the ratio between the negatives and positives(hard negative mining)

# Box_coder para
Y_SCALE = 10
X_SCALE = 10
H_SCALE = 5
W_SCALE = 5

PATH = os.path.dirname(os.getcwd())
ANCHORS = pd.read_csv(os.path.join(PATH,'anchor','anchor.txt'))
LABELS = pd.read_csv(os.path.join(PATH,'label','label.txt'))
NUM_CLASSESS = len(LABELS.Class_name)
MODEL_NAME = 'model.ckpt'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(PATH,'model','checkpoint')
