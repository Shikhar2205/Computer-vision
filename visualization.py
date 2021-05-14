# -*- coding: utf-8 -*-
"""
Created on Thu May 13 05:55:30 2021

@author: Shikhar Bajpai
"""

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import glob
import torch
import os
from PIL import Image, ImageDraw

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def visualization(image_path,label_path,output=False,resize_dim=1):
    '''

    Parameters
    ----------
    image_path : STR
        DESCRIPTION.
    label_path : STR
        DESCRIPTION.
    output : BOOL, optional
        DESCRIPTION. The default is False.
    resize_dim : INT, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.
    
    NOTE:
    This function is used to visualize any input or output image. Format of input and output file
    are different therefore there is a optional parameter as ouput which is set to False by default.
    Resize_dim takes the value as how much the image was resized. 
    '''
    image = cv.imread(image_path) 
    f1=open(label_path, 'r')
    if (not output):
        lines=f1.readlines()[2:]
    else:
        lines=f1.readlines()
    for i in range(len(lines)):
        split=lines[i].split(' ')
        if( not output):
            xmin=int(float(min(split[0],split[2],split[4],split[6])))
            xmax=int(float(max(split[0],split[2],split[4],split[6])))
            ymin=int(float(min(split[1],split[3],split[5],split[7])))
            ymax=int(float(max(split[1],split[3],split[5],split[7])))
        else:
            if(resize_dim==1):
                scale_y=1
                scale_x=1
            else:                
                y_,x_,c= image.shape 
                scale_y=resize_dim/y_
                scale_x=resize_dim/x_
            xmin=int(float(split[0])/scale_x)
            xmax=int(float(split[2])/scale_x)
            ymin=int(float(split[1])/scale_y)
            ymax=int(float(split[3])/scale_y)
        start_point = (xmin, ymin) 
        end_point = (xmax, ymax) 
        print (start_point,end_point)
        # Blue color in BGR 
        color = (255, 0, 0)   
        # Line thickness of 2 px 
        thickness = 2  
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        print (i)
        img_cov = cv.rectangle(image, start_point, end_point, color, thickness) 

    plt.imshow(img_cov)
    plt.show()

#Visualizing input image        
visualization('E:/DOTA DATASET/ships_val/P0887.png', 'E:\DOTA DATASET\labels_val\P0887.txt')

#Visualizing output image        
visualization('E:/DOTA DATASET/ships_val/P0887.png', 'E:\DOTA DATASET\FastRCNN-split-labels\P0887.txt',output=True)

#Visualizing resized output image        
visualization('E:/DOTA DATASET/ships_val/P0887.png', 'E:\DOTA DATASET\resized_labels_val\P0887.txt',ouput=True,resize_dim=2064)
