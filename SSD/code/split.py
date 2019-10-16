# -*- coding: utf-8 -*-

import os
import random

ratio = 0.25
path = os.path.dirname(os.getcwd())
xmldir = os.path.join(path,'data','annotation','xml')
xmlname = os.listdir(xmldir)

#random.seed(888)
random.shuffle(xmlname)
num_val_iamges = int(ratio*len(xmlname))
num_train_iamges = len(xmlname) - num_val_iamges

train_images = xmlname[0:num_train_iamges]
val_images = xmlname[num_train_iamges::]

# train
train_txt_path = os.path.join(path,'data','train','train.txt')
train_txt = open(train_txt_path,'w')

for image in train_images:
    train_txt.write(image[0:-4]+'\n')
train_txt.close()
                                                              
# val
val_txt_path = os.path.join(path,'data','val','val.txt')
val_txt = open(val_txt_path,'w')

for image in val_images:
    val_txt.write(image[0:-4]+'\n')
val_txt.close()
