# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:42:44 2018
@author: LiXiaoGang

Anchor Clustering using K-Means.

"""
from __future__ import division

import os
import numpy as np
import parameters as para


def generator():
    with open(os.path.join(para.PATH,'anchor','anchor.txt'),'w') as f:
        f.write('cx,cy,w,h\n')
        scale = []
        for k in range(len(para.FEATURE_MAPS)):
            scale.append(para.MIN_SCALE+(para.MAX_SCALE-para.MIN_SCALE)*(k)/(len(para.FEATURE_MAPS)-1))
        
        for k in range(len(para.FEATURE_MAPS)):
            s_k = scale[k]
            feature_map_size_h = para.FEATURE_MAPS[k][0]
            feature_map_size_w = para.FEATURE_MAPS[k][1]
            
            for h in range(feature_map_size_h):
                for w in range(feature_map_size_w):
                    for r in range(len(para.ASPECT_RATIOS)):
                        anchor_cy = (h + 0.5)/feature_map_size_h
                        anchor_cx = (w + 0.5)/feature_map_size_w
                        anchor_h = s_k/np.sqrt(para.ASPECT_RATIOS[r])
                        anchor_w = s_k*np.sqrt(para.ASPECT_RATIOS[r])
                        f.write(str(anchor_cx)+','+str(anchor_cy)+','+str(anchor_w)+','+str(anchor_h)+'\n') # cx,cy,w,h
                        
                        # add a default box,for the aspect ratio of 1
                        if (para.ASPECT_RATIOS[r] == 1) and (k<len(para.FEATURE_MAPS)-1):
                            s_k_1 = scale[k+1]
                            anchor_cy = (h + 0.5)/feature_map_size_h
                            anchor_cx = (w + 0.5)/feature_map_size_w
                            anchor_h = np.sqrt(s_k*s_k_1)/np.sqrt(para.ASPECT_RATIOS[r])
                            anchor_w = np.sqrt(s_k*s_k_1)*np.sqrt(para.ASPECT_RATIOS[r])
                            f.write(str(anchor_cx)+','+str(anchor_cy)+','+str(anchor_w)+','+str(anchor_h)+'\n') # cx,cy,w,h
        f.close()


if __name__ == '__main__':
    generator()