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
from models import resnet50, resnet50_dilated
from models2 import se_resnext50_32x4d

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

def train(epoch = 0):
    """
    Training routine for the model
    """

    model.train()
    
    train_loss = 0
    
    for batch_idx, (data, targets) in enumerate(trainloader, 0):
#        print(batch_idx)
        data[data != data] = 0
        
        data, targets = data.to(device), targets.type(torch.FloatTensor).to(device)
       
        optimizer.zero_grad() 
        output = model(data)
        
        loss = criterion(output, targets)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1)%10 ==0:
            print("Batchidx: {} of {} Loss = {:.3f}".format((batch_idx +1), len(trainloader), train_loss/10))
            train_loss = 0
        
if __name__ == "__main__":
    
    tx = transforms.Compose([transforms.ToPILImage(),
                             transforms.RandomHorizontalFlip(p = 0.5),
                             transforms.RandomRotation(degrees = 5),
                             transforms.ColorJitter(brightness = 0.05, contrast = 0.05),
                             transforms.ToTensor(),])
#    tx = None
    trainset = RsnaRIT(dataPartition='train', 
                       dataPath=args.datasetpath, 
                       dataFrame=args.picklefile,
                       transform=tx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle = True, num_workers=8)
    
    if args.network == 'resnet50':
        model = resnet50(pretrained = args.pretrained_flag)
    elif args.network == 'dilatedresnet50':
        model = resnet50_dilated(pretrained = args.pretrained_flag)
    elif args.network == 'seresnext50':
        model = se_resnext50_32x4d(num_classes=6)
    else:
        raise NameError ("Model not found")
        
    criterion = nn.BCEWithLogitsLoss()
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.1, last_epoch=-1)
        
    if args.disable_cuda is not True and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    model = model.to(device)
    
    for epoch in range(args.epochs):
        train(epoch)
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(model.state_dict(), './checkpoint/{}_epoch{}.pt'.format(args.model_name, epoch))
        scheduler.step()