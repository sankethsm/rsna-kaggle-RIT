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
from networks_resnet import resnet50, resnet101, resnext50_32x4d, dilated_resnext50_32x4d

np.random.seed(3108)
torch.manual_seed(3108)

parser = argparse.ArgumentParser(description='Training RSNA')
parser.add_argument('--config-file', type=str, default='configs/dresnextv2_gcblock.yaml')
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
parser.add_argument('--drop_prob', type=float, default=None)
parser.add_argument('--dropblock_size', type=float, default=None)
parser.add_argument('--dropblock_steps', type=float, default=None)
parser.add_argument('--gcblock3', type=bool, default=False)
parser.add_argument('--gcblock4', type=bool, default=False)
parser.add_argument('--gcblock5', type=bool, default=False)


args = parse_args(parser)
print(args)

import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available() and args.disable_cuda is False:
    device = 'cuda'
else:
    device = 'cpu'

def train(epoch = 0):
    """
    Training routine for the model
    """

    net.train()
    
    train_loss = 0
    optimizer.zero_grad() 

    for batch_idx, (data, targets) in enumerate(trainloader, 0):
#        print(batch_idx)
        optimizer.zero_grad()
        data[data != data] = 0
        
        data, targets = data.to(device), targets.type(torch.FloatTensor).to(device)
       
        output = net(data)
        
        loss = criterion(output, targets.squeeze(1))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1)%5 ==0:
#            optimizer.step()print(z(
#            optimizer.zero_grad()
            print("Batchidx: {} of {} Loss = {:.3f}".format((batch_idx +1), len(trainloader), train_loss/5))
            torch.save(net.state_dict(), './checkpoint/{}_batchupdate.pt'.format(args.model_name))
            train_loss = 0
        
if __name__ == "__main__":

    ## Change pickle file to read files properly
    trainDf = pd.read_pickle(args.picklefile)
    trainDf = trainDf[trainDf['Image'] != 'ID_6431af929']
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
    
    tx = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                             transforms.RandomVerticalFlip(0.5),
                             transforms.ColorJitter(brightness=0.08, contrast=0.08, ),
                             transforms.RandomApply(
                                     [transforms.RandomRotation(degrees = 20)], p=0.5),
                             transforms.ToTensor(),])
    
    trainset = RsnaRITv2(dataPartition='train', 
                       dataPath=args.datasetpath, 
                       dataFrame=trainDf,
                       randArray=finArr,
                       transforms = tx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle = True)
    
    if args.network == 'resnet50':
        net = resnet50(pretrained = True, drop_prob = args.drop_prob, dropblock_size = args.dropblock_size
                       , dropblock_steps = args.dropblock_steps, 
                       isgcb = [args.gcblock3, args.gcblock4, args.gcblock5])
        net.fc = nn.Linear(2048, 6)
    elif args.network == 'resnet101':
        net = resnet101(pretrained = True, drop_prob = args.drop_prob, dropblock_size = args.dropblock_size
                       , dropblock_steps = args.dropblock_steps, 
                       isgcb = [args.gcblock3, args.gcblock4, args.gcblock5])
        net.fc = nn.Linear(2048, 6)
    elif args.network == 'resnext50':
        net = resnext50_32x4d(pretrained = True, drop_prob = args.drop_prob, dropblock_size = args.dropblock_size
                       , dropblock_steps = args.dropblock_steps, 
                       isgcb = [args.gcblock3, args.gcblock4, args.gcblock5])
        net.fc = nn.Linear(2048, 6)
    elif args.network == 'dilatedresnext50':
        net = dilated_resnext50_32x4d(pretrained = True, drop_prob = args.drop_prob, dropblock_size = args.dropblock_size
                       , dropblock_steps = args.dropblock_steps, 
                       isgcb = [args.gcblock3, args.gcblock4, args.gcblock5])
        net.fc = nn.Linear(2048, 6)
        
    net = net.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=args.momentum)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma=0.1, last_epoch=-1)
        
    if args.disable_cuda is not True and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    for epoch in range(args.epochs):
        train(epoch)
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/{}_epoch{}.pt'.format(args.model_name, epoch))
        scheduler.step()
