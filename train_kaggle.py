#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:42:06 2019

@author: aneesh
"""

import argparse
import sys
sys.path.append("..")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import parse_args

from datagenerator import RsnaRIT
from models import resnet18, resnet50

np.random.seed(3108)
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

import warnings
warnings.filterwarnings("ignore")

def train(epoch = 0):
    """
    Training routine for the model
    """

    model.train()
    
    train_loss = 0
    optimizer.zero_grad() 

    for batch_idx, (data, targets) in enumerate(trainloader, 0):
#        print(batch_idx)
        optimizer.zero_grad()
        data[data != data] = 0
        
        data, targets = data.to(device), targets.type(torch.FloatTensor).to(device)
       
        output = model(data)
        
        loss = criterion(output, targets.squeeze(1))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1)%5 ==0:
#            optimizer.step()
#            optimizer.zero_grad()
            print("Batchidx: {} of {} Loss = {:.3f}".format((batch_idx +1), len(trainloader), train_loss/5))
            torch.save(model.state_dict(), './checkpoint/{}_batchupdate.pt'.format(args.model_name))
            train_loss = 0
        
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
    
    trainset = RsnaRIT(dataPartition='train', 
                       dataPath=args.datasetpath, 
                       dataFrame=trainDf,
                       randArray=finArr)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle = True)
    
    if args.network == 'resnet50':
        model = resnet50(pretrained = args.pretrained_flag)
    elif args.network == 'dilatedresnet50':
        model = resnet50_dilated(pretrained = args.pretrained_flag)
    elif args.network == 'resnet18':
        model = resnet18(pretrained = args.pretrained_flag)
    else:
        raise NameError ("Model not found")
        
    criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([1,1,1,1,1,2]).cuda())
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma=0.1, last_epoch=-1)
        
    if args.disable_cuda is not True and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    model = model.to(device)
    model = nn.DataParallel(model)
    
    for epoch in range(args.epochs):
        train(epoch)
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/{}_epoch{}.pt'.format(args.model_name, epoch))
        scheduler.step()