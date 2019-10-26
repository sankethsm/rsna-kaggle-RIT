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

#-------------------------------------------------------------------------------------------------#
class RsnaRIT(Dataset):

    def __init__(self, dataPartition, dataPath, dataFrame, transform=None):
        self.dataPath = os.path.join(dataPath, dataPartition)
        self.dataList = os.listdir(self.dataPath)
        self.dataPartition = dataPartition
        self.dataFrame = pd.read_pickle(dataFrame)
        self.transform = transform
        #self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, img_id):
        imgID = self.dataList[img_id]
        imagePath = os.path.join(self.dataPath, imgID)
        
        data = pydicom.dcmread(imagePath)
        window_center , window_width, intercept, slope = get_windowing(data)
        img = pydicom.read_file(imagePath).pixel_array
        #img = window_image(img, window_center, window_width, intercept, slope)
        shp = img.shape
        intImg = np.zeros((3,shp[0],shp[1]))

        intImg[0,:,:] = window_image(img, 40, 80, intercept, slope)     # Brain window
        intImg[0,:,:] = np.divide((intImg[0,:,:]-intImg[0,:,:].min()), (intImg[0,:,:].max() - intImg[0,:,:].min()))
        intImg[1,:,:] = window_image(img, 80, 200, intercept, slope)    # Subdural window
        intImg[1,:,:] = np.divide((intImg[1,:,:]-intImg[1,:,:].min()), (intImg[1,:,:].max() - intImg[1,:,:].min()))
        intImg[2,:,:] = window_image(img, 600, 2000, intercept, slope)  # Bone window
        #intImg[2,:,:] = window_image(img, 80, 200, intercept, slope)  
        intImg[2,:,:] = np.divide((intImg[2,:,:]-intImg[2,:,:].min()), (intImg[2,:,:].max() - intImg[2,:,:].min()))
        
        if self.dataPartition == 'train':
            labels = np.array(self.dataFrame.loc[imgID,:].tolist())
        elif self.dataPartition == 'test':
            labels = None
        
        #minIm = intImg.min()
        #maxIm = intImg.max()
        #intImg = np.divide((intImg-minIm), (maxIm - minIm))

        finImg = np.zeros((3,256,256))
        for i in range(3):
            im = intImg[i,:,:].squeeze()
            finImg[i,:,:] = cv2.resize(im, (256,256))
        finImg = finImg * 255.0
        finImg = finImg.astype(np.uint8)
        #img = img.astype(np.uint8())
        #img = cv2.equalizeHist(img)

        if self.transform:
            finImg = self.transform(finImg)

        return finImg, labels