from glob import glob
import os
import sys
sys.path.append("..")

import torch
import pandas as pd
import numpy as np
import pydicom
import cv2
from random import randrange
from torch.utils.data import Dataset, DataLoader

from utils import get_windowing, window_image
from PIL import Image

import random
random.seed(3108)
#-------------------------------------------------------------------------------------------------#
class RsnaRIT(Dataset):

    def __init__(self, dataPartition, dataPath, dataFrame, randArray, transforms=None):
        self.dataPath = os.path.join(dataPath, dataPartition)
        self.dataList = os.listdir(self.dataPath)
        self.dataPartition = dataPartition
        self.dataFrame = dataFrame
        self.randArray = randArray
        self.transform = transforms
        #self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.randArray)

    def __getitem__(self, img_id):
        imgID = np.random.choice(self.randArray[img_id])
        imagePath = os.path.join(self.dataPath, imgID + '.dcm')
        
        data = pydicom.dcmread(imagePath)
        window_center, window_width, intercept, slope = get_windowing(data)
        img = pydicom.read_file(imagePath).pixel_array
        img = img.astype(np.float32)
#        img = cv2.resize(img, (128, 128))

#        brainw = window_image(img, 40, 80, intercept, slope)    # Brain window
#        subdw  = window_image(img, 80, 200, intercept, slope)    # Subdural window
#        bonew  = window_image(img, 600, 2400, intercept, slope) # Bone window
        imgw = window_image(img, window_center, window_width, intercept, slope)
        
        imgw = (imgw * 255.)
        outImg = np.stack((imgw, imgw, imgw), axis = 2)
        outImg = Image.fromarray(outImg.astype(np.int8), mode = 'RGB')
        outImg = outImg.resize((128, 128), resample = Image.BICUBIC)
#                
        if self.dataPartition == 'train':
            labels = self.dataFrame[self.dataFrame['Image']==imgID]
            labels = labels.loc[:,'any':'subdural'].values
        elif self.dataPartition == 'test':
            labels = None
        
        if self.transform is not None:
            outImg = self.transform(outImg)
#        
        return outImg, labels#, imgID
    
    
class RsnaRITv2(Dataset):

    def __init__(self, dataPartition, dataPath, dataFrame, randArray, transforms=None):
        self.dataPath = os.path.join(dataPath, dataPartition)
        self.dataList = os.listdir(self.dataPath)
        self.dataPartition = dataPartition
        self.dataFrame = dataFrame
        self.randArray = randArray
        self.transform = transforms
        #self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.randArray)

    def __getitem__(self, img_id):
        imgID = np.random.choice(self.randArray[img_id])
        imagePath = os.path.join(self.dataPath, imgID + '.dcm')
        
        data = pydicom.dcmread(imagePath)
        window_center, window_width, intercept, slope = get_windowing(data)
        img = pydicom.read_file(imagePath).pixel_array
        img = img.astype(np.float32)
#        img = cv2.resize(img, (128, 128))

        brainw = window_image(img, 40, 80, intercept, slope)    # Brain window
        subdw  = window_image(img, 80, 200, intercept, slope)    # Subdural window
        softtw  = window_image(img, 40, 380, intercept, slope) # Bone window
#        imgw = window_image(img, window_center, window_width, intercept, slope)
        
        brainw = brainw * 255.
        subdw = subdw * 255.
        softtw = softtw * 255.
        
        outImg = np.stack((brainw, subdw, softtw), axis = 2)
        outImg = Image.fromarray(outImg.astype(np.int8), mode = 'RGB')
        outImg = outImg.resize((128, 128), resample = Image.BICUBIC)
#                
        if self.dataPartition == 'train':
            labels = self.dataFrame[self.dataFrame['Image']==imgID]
            labels = labels.loc[:,'any':'subdural'].values
        elif self.dataPartition == 'test':
            labels = None
        
        if self.transform is not None:
            outImg = self.transform(outImg)
#        
        return outImg, labels#, imgID