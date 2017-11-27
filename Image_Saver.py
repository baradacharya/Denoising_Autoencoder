# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 02:13:37 2017

@author: anurag
"""
from Image_Loader import LoadImages
from os import listdir
from PIL import Image as PImage
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
def SaveImages(img_arr,path):
    # return array of images
    path = path + ".png"
    misc.imsave(path,img_arr)

'''
path1 = "train_modified\\"
path2 = "test_clean\\1"
imgs = LoadImages(path1)
x_train = np.array( imgs )
for i in range(x_train.shape[0]):
    path2 = "test_clean\\" + str(i)
    SaveImages(x_train[i].reshape(420, 540),path2)     
'''
