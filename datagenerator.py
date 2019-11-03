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

import random
random.seed(3108)
#-------------------------------------------------------------------------------------------------#
class RsnaRIT(Dataset):

    def __init__(self, dataPartition, dataPath, dataFrame, randArray, transform=None):
        self.dataPath = os.path.join(dataPath, dataPartition)
        self.dataList = os.listdir(self.dataPath)
        self.dataPartition = dataPartition
        self.dataFrame = dataFrame
        self.randArray = randArray
        self.transform = transform
        #self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.randArray)

    def __getitem__(self, img_id):
        imgID = np.random.choice(self.randArray[img_id])
        imagePath = os.path.join(self.dataPath, imgID + '.dcm')
        
        data = pydicom.dcmread(imagePath)
        window_center , window_width, intercept, slope = get_windowing(data)
        img = pydicom.read_file(imagePath).pixel_array
        img = img.astype(np.float32)
        img = cv2.resize(img, (256, 256))

        brainw = window_image(img, 40, 80, intercept, slope)    # Brain window
        subdw  = window_image(img, 80, 200, intercept, slope)    # Subdural window
        bonew  = window_image(img, 600, 2400, intercept, slope) # Bone window
        
        outImg = np.stack((brainw, subdw, bonew))
        
        if self.dataPartition == 'train':
            labels = self.dataFrame[self.dataFrame['Image']==imgID]
            labels = labels.loc[:,'any':'subdural'].values
        elif self.dataPartition == 'test':
            labels = None
        
        return outImg, labels