# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:20:45 2017

@author: 7
"""

from os import listdir
from PIL import Image as PImage
from scipy import misc
import numpy as np
from Image_loader import LoadImages
"""
def LoadImages(path):
    # return array of images
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img =  misc.imread(path + image)
        loadedImages.append(img)
    return loadedImages
"""


def ModifyImages(path,path1):
    # modify images to same scale

    imagesList = listdir(path)
    for image in imagesList:
        old_img = PImage.open(path + image)
        old_size = old_img.size
        new_size = (540,420)
        new_img =  PImage.new("L", new_size)   
        new_img.paste(old_img,((new_size[0]-old_size[0])//2,(new_size[1]-old_size[1])//2))
        new_img.save(path1 + image)

"""
path = "train\\"
path1 = "train_modified\\"
ModifyImages(path,path1)
imgs = LoadImages(path1)
a = np.array( imgs )
print (a.shape)
print("finished")


path = "test\\"
path1 = "test_modified\\"

ModifyImages(path,path1)
imgs = LoadImages(path1)
a = np.array( imgs )
print (a.shape)
print("finished")

path = "train_cleaned\\"
path1 = "train_cleaned_modified\\"

ModifyImages(path,path1)
imgs = LoadImages(path1)
a = np.array( imgs )
print (a.shape)
print("finished")
"""