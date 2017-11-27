# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:54:04 2017

@author: 7
"""

from os import listdir
from PIL import Image as PImage
from scipy import misc
import numpy as np

def LoadImages(path):
    # return array of images
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img =  misc.imread(path + image)
        loadedImages.append(img)
    return loadedImages