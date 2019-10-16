# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET


def parse_object(xmlFile):
    Tree = ET.parse(xmlFile) 
    root = Tree.getroot()
    object_set = root.findall('object')
    
    object_list = []
    for one in object_set:
        
        # get class
        name = one.find('name')
        classes = name.text
        
        # get coordinates of the box 
        bndbox = one.findall('bndbox')
        
        xmin= bndbox[0].find('xmin')
        xmin = int(xmin.text)
        
        ymin= bndbox[0].find('ymin')
        ymin = int(ymin.text)
        
        xmax= bndbox[0].find('xmax')
        xmax = int(xmax.text)
        
        ymax= bndbox[0].find('ymax')
        ymax = int(ymax.text)
        
        patch_info = {'classes':classes,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}
        object_list.append(patch_info)
        
    return object_list


def parse_size(xmlFile):
    Tree = ET.parse(xmlFile) 
    root = Tree.getroot()
    size = root.findall('size')[0]
    
    width = size.find('width')
    width = int(width.text)
    
    height = size.find('height')
    height = int(height.text)
    
    depth = size.find('depth')
    depth = int(depth.text)  
    
    size = {'width':width,'height':height,'depth':depth}
        
    return size
