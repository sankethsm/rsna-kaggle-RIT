#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:13:07 2019

@author: aneesh
"""

import argparse
import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import parse_args

from torchvision import models
from datagenerator import RsnaRIT

torch.manual_seed(3108)
parser = argparse.ArgumentParser(description='Training RSNA')
parser.add_argument('--config-file', type=str, default='configs/default.yaml')
parser.add_argument('--datasetpath', type=str, default=None)
parser.add_argument('--picklefile', type=str, default=None)
parser.add_argument('--network', type=str, default=None)
parser.add_argument('--pretrained-flag', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=1e-2)
parser.add_argument('--beta1', type=float, default=1e-2)
parser.add_argument('--beta2', type=float, default=1e-2)
parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam'))
parser.add_argument('--model-name', default = 'model.pt', help = 'What name to save for ?')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parse_args(parser)
print(args)

if torch.cuda.is_available() and args.disable_cuda is False:
    device = 'cuda'
else:
    device = 'cpu'
    
if __name__ == "__main__":
    
    ## Change pickle file to read files properly
    trainDf = pd.read_pickle(args.picklefile)
    df0 = trainDf.loc[trainDf['any']==0]
    df1 = trainDf.loc[trainDf['any']==1]

    df0Names = df0.sample(n=98000*5)
    df0Names = np.array(df0Names['Image'])
    df0Names = df0Names.reshape((98000,5))
    
    df1Names = np.array(df1['Image'])
    df1Names = np.array([df1Names,]*5).transpose()
    
    intArr = np.concatenate((df0Names,df1Names), axis=0)
    
    finIdx = np.arange(intArr.shape[0])
    np.random.shuffle(finIdx)
    finArr = intArr[finIdx]
    
    tx = transforms.Compose([transforms.ToTensor(),])
    
    trainset = RsnaRIT(dataPartition='train', 
                       dataPath=args.datasetpath, 
                       dataFrame=trainDf,
                       randArray=finArr,
                       transforms = tx)
    trainloader = DataLoader(trainset, batch_size=10, shuffle = True)
    
    if args.network == 'resnet50':
        net = models.resnet50(pretrained = True)
        net.fc = nn.Linear(2048, 6)
    elif args.network == 'resnet101':
        net = models.resnet101(pretrained = True)
        net.fc = nn.Linear(2048, 6)
        
    net.load_state_dict(torch.load('checkpoint/resnetsimple_batchupdate.pt'))
    net.eval()
    net = net.to(device)
    net = nn.DataParallel(net)
    
    z = nn.Sigmoid()
    with torch.no_grad():
        imgs, y = next(iter(trainloader))
        print(y, z(net(imgs.cuda())))
    
