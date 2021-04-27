# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 23:57:48 2021

@author: Shikhar Bajpai
"""

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import glob
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path=r"E:/DOTA DATASET/labels/*.txt"
path1=r"E:/DOTA DATASET/ships/*.png"
filelist=glob.glob(path)
filelist1=glob.glob(path1)
dataset=pd.DataFrame({'imagelabels':filelist,'image':filelist1})
dataset['countShips']=0


image = cv.imread(r'E:/DOTA DATASET/ships/P0001.png') 
f1=open(filelist[0], 'r')
lines=f1.readlines()[2:]
for i in range(len(lines)):
    split=lines[i].split(' ')
    xmin=int(float(min(split[0],split[2],split[4],split[6])))
    xmax=int(float(max(split[0],split[2],split[4],split[6])))
    ymin=int(float(min(split[1],split[3],split[5],split[7])))
    ymax=int(float(max(split[1],split[3],split[5],split[7])))
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