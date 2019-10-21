import os
import pandas as pd
import pydicom
import numpy as np
import time
import cv2
import torch
import shutil

#-----------------------------------------------------------------------------------#
def prepareDataframe(path, targets, train=False, nrows=None):
    """
    Prepare Pandas DataFrame for fitting neural network models
    Returns a Dataframe with two columns
    ImageID and Labels (list of all labels for an image)
    """ 
    df = pd.read_csv(path, nrows=nrows)
    if train:
        # Duplicates found from this kernel:
        # https://www.kaggle.com/akensert/resnet50-keras-baseline-model
        removeDuplicates = [1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
                                312468,  312469,  312470,  312471,  312472,  312473,
                                2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
                                3032994, 3032995, 3032996, 3032997, 3032998, 3032999]  
        df = df.drop(index=removeDuplicates).reset_index(drop=True)
    
    # Get ImageID for using with generator
    df['ImageID'] = df['ID'].str.rsplit('_', 1).map(lambda x: x[0]) + '.dcm'
    # Get labels for each image
    labelList = df.groupby('ImageID')['Label'].apply(list)
    
    # A clean DataFrame with a column for ImageID and columns for each label
    newDf = pd.DataFrame({'ImageID': df['ImageID'].unique(), 
                           'Labels': labelList}).set_index('ImageID').reset_index()
    newDf[targets] = pd.DataFrame(newDf['Labels'].values.tolist(), index= newDf.index)
    newDf = newDf.drop('Labels', axis=1)
    return newDf

#-----------------------------------------------------------------------------------#
def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img

#-----------------------------------------------------------------------------------#
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

#-----------------------------------------------------------------------------------#
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

#-----------------------------------------------------------------------------------#
def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, 
                                    now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

#-----------------------------------------------------------------------------------#
def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    if os.path.exists(os.path.join(path, 'checkpoints'))==False:
        os.mkdir(os.path.join(path, 'checkpoints'))

    prefix_save = os.path.join(path, 'checkpoints', prefix)
    #print(prefix_save)
    epochNum = state['epoch']
    name = prefix_save + '_' + 'latest_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

#-----------------------------------------------------------------------------------#
def adjust_opt(optAlg, optimizer, epoch, lr):
    if optAlg == 'adam':
        if epoch < 40:
            lr = lr
        elif epoch == 40:
            lr = lr*1e-1
        elif epoch == 80:
            lr = lr*1e-1

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(param_group['lr'])

#-----------------------------------------------------------------------------------#