# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:27:15 2021

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
path=r"E:/DOTA DATASET/labels_val/*.txt"
path1=r"E:/DOTA DATASET/ships_val/*.png"
filelist=glob.glob(path)
filelist1=glob.glob(path1)
dataset=pd.DataFrame({'imagelabels':filelist,'image':filelist1})
dataset['countShips']=0

## testing script
image = cv.imread(r'E:/DOTA DATASET/ships_val/P0019.png') 
f1=open('E:\DOTA DATASET\P0019.txt', 'r')
lines=f1.readlines()
for i in range(len(lines)):
    split=lines[i].split(' ')
    y_,x_,c= image.shape 
    scale_y=2064/y_
    scale_x=2064/x_
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
