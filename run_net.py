"""
Written by Sanketh S. Moudgalya for RSNA kaggle competition

Run script
"""
import os
import datetime
import time
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import resnet50, resnet101
from tensorboardX import SummaryWriter

from utils_.utils import adjust_opt, datestr, save_checkpoint, prepareDataframe
from datagen.datagenerator import RsnaRIT, ToTensor#, Augmentations
from scripts.train import trainModel
from scripts.valid import validModel
from scripts.arg_parser import argParser

#-------------------------------------------------------------------------------#

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = "cuda"
device = torch.device(dev)
#device = torch.device("cpu")
start_time = datetime.datetime.now()

#-------------------------------------------------------------------------------#
rootDir = "D:\\kaggle\\rsna" # Change for unix systems
#rootDir = "G:\\My Drive\\Kaggle" # Change for unix systems

#-------------------------------------------------------------------------------#
## Fill these first
modelTypeFlag = 'resnet101'
#randSeed = 705
debugFlag = False # Set to True if you do not want logs to be created during debugging
optims = ['adam']
lrs = [0.0001] #np.linspace(0.001, 0.00001, 10)
bsze = [32]
mm = 0.9 # If using SGD. Momentum
notes = modelTypeFlag +" Run. "+"" # Add whatever notes to the run

#-------------------------------------------------------------------------------#

def main():
    parser = argParser()
    args = parser.parse_args()

    bestPred = 0.15
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('Using CUDA?: ', args.cuda)

#-------------------------------------------------------------------------------#
    train = trainModel
    #valid = validModel

#-------------------------------------------------------------------------------#  

    print('Loading labels into dataframe')
    '''
    trainSheet = os.path.join(rootDir, 'stage_1_train.csv')

    # All labels that we have to predict in this competition
    targets = ['epidural', 'intraparenchymal', 
            'intraventricular', 'subarachnoid', 
            'subdural', 'any']

    trainDf = prepareDataframe(trainSheet, targets, train=True)
    trainDf = trainDf.set_index("ImageID", drop=True)
    trainDf.to_pickle(os.path.join(rootDir, 'pd_sheet.pkl'))
    '''
    trainDf = pd.read_pickle(os.path.join(rootDir, "pd_sheet.pkl"))
    print('Dataframe loaded')


#-------------------------------------------------------------------------------#
# Initialize optimizer and write out results and checkpoints
    for opts in optims:
        for l in lrs:
            for bs in bsze:
                torch.manual_seed(args.seed)
                if args.cuda:
                    torch.cuda.manual_seed(args.seed)
                
                if l==0.000001:
                    weight_decay = args.wd*0.1
                else:
                    weight_decay = args.wd
                print("build model")
                
                ## Add models as needed
                # Resnet 50 model. 
                if modelTypeFlag == 'resnet50':
                    model = resnet50(pretrained=False)
                    #for param in model.parameters():
                    #    param.requires_grad = False
                    model.fc = nn.Linear(2048, 6) # 6th class. 'any' will be added during testing
                    #exit(0)

                elif modelTypeFlag == 'resnet101':
                    model = resnet101(pretrained=False)
                    #for param in model.parameters():
                    #    param.requires_grad = False
                    model.fc = nn.Linear(2048, 6) # 6th class. 'any' will be added during testing
                    #exit(0)
                
                if args.resume:
                    if os.path.isfile(args.resume):
                        print("=> loading checkpoint '{}'".format(args.resume))
                        checkpoint = torch.load(args.resume)
                        args.start_epoch = checkpoint['epoch']
                        bestPred = checkpoint['bestPred']
                        model.load_state_dict(checkpoint['state_dict'])
                        print("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.evaluate, checkpoint['epoch']))
                    else:
                        print("=> no checkpoint found at '{}'".format(args.resume))

                print('  + Number of params: {}'.format(
                        sum([p.data.nelement() for p in model.parameters()])))

                if dev=="cuda":
                    print('Using cuda')
                    model = model.cuda()
                elif dev=="cpu":
                    print('Using cpu')
                    model = model.cpu()
                
                #-------------------------------------------------------------------#
                if not debugFlag:
                    writer = SummaryWriter(logdir='.\\logs\\classify_{}_{}_{}_{}'.format(datestr(), opts, bs, str(l)))
                    print(opts, l, bs)

                #-------------------------------------------------------------------#
                    sav_fol = args.save or '.\\torchRuns\\classify_{}_{}_{}_{}'.format(datestr(), opts, bs, str(l))

                    if os.path.exists(sav_fol)==False:
                        os.mkdir(sav_fol)
                #-------------------------------------------------------------------#

                print("loading dataset")
                trainSet = RsnaRIT('train', rootDir, trainDf, 
                            transform=transforms.Compose([ToTensor()]))
                
                #-------------------------------------------------------------------#
                
                #trainLoader = DataLoader(trainSet, batch_size=bs, shuffle=True)

                # dataSz = len(trainSet)
                # validSplit = 0.2
                # indxs = list(range(dataSz))
                # split = int(np.floor(validSplit * dataSz))
                # shuffleFlag = True

                # if shuffleFlag:
                #     np.random.seed(randSeed)
                #     np.random.shuffle(indxs)
                #     trainIndxs, validIndxs = indxs[split:], indxs[:split]
                
                # Creating PT data samplers and loaders:
                #trainSampler = SubsetRandomSampler(trainIndxs)
                #validSampler = SubsetRandomSampler(validIndxs)

                #trainLoader = DataLoader(trainSet, batch_size=bs, 
                                            #sampler=trainSampler)
                trainLoader = DataLoader(trainSet, batch_size=bs, 
                                            shuffle=True)

                #validLoader = DataLoader(trainSet, batch_size=1,
                                            #sampler=validSampler)

                #-------------------------------------------------------------------#
                
                if not debugFlag:
                    with open(os.path.join(sav_fol, "hyperparams_.csv"), 'w+') as wfil:
                        wfil.write("Train classifier for RSNA-RIT dataset\n")
                        wfil.write("optimizer," + str(opts) + "\n")
                        wfil.write("loss func," + "CrossEntropy" + '\n')
                        wfil.write("learning rate," + str(l) + '\n')
                        wfil.write("train batch size," + str(bs) + '\n')
                        wfil.write("validation batch size, Not validating" + str(1) + '\n')
                        wfil.write("momentum if SGD," + str(mm) + '\n')
                        wfil.write("total epochs," + str(args.nEpochs) + '\n')
                        wfil.write("augmentation type," + 
                        str('Add if needed') + '\n')
                        wfil.write("Weight decay, " + str(weight_decay) + '\n')
                        wfil.write("start time," + str(start_time) + '\n')
                        wfil.write("dataset," + 'RSNA' + '\n')
                        wfil.write("Model,"+str(modelTypeFlag)+'\n')
                        wfil.write("Notes: " + notes + '\n')
                    trainF = open(os.path.join(sav_fol, 'train.csv'), 'w')
                    #validF = open(os.path.join(sav_fol, 'valid.csv'), 'w')
                else:
                    trainF = None
                    #validF = None

                #-------------------------------------------------------------------#
                if opts == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=l, momentum=mm,
                    weight_decay=weight_decay, nesterov=True)
                elif opts == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=l, 
                    weight_decay=weight_decay, amsgrad=True)
                
                for epoch in range(0, args.nEpochs + 1):
                    # Uncomment if changing learning rate is needed. Change LR values in subroutine
                    #adjust_opt(opts, optimizer, epoch, l)
                    
                    t0 = time.time()
                    
                    lossTrain = train(epoch, model, trainLoader, optimizer, device, debugFlag, trainF)
                    
                    #accuValid, lossValid = valid(epoch, model, validLoader, device, debugFlag, validF)

                    print('time_elapsed {} seconds'.format(time.time()-t0)) 

                    # if not debugFlag:
                    #     writer.add_scalars('Accuracy', {'valid Accuracy':accuValid}, global_step=epoch)

                    if epoch%10 == 0:
                        for name, param in model.named_parameters():
                            if not debugFlag:
                                writer.add_histogram(name, param, global_step=epoch)

                    if not debugFlag:
                        # writer.add_scalars('Loss', {'train_loss':lossTrain,
                        #                         'valid_loss': lossValid}, global_step=epoch)
                        writer.add_scalars('Loss', {'train_loss':lossTrain}, global_step=epoch)

                    is_best = False
                    if (lossTrain) < bestPred:
                        is_best = True
                        bestPred = lossTrain
                    
                    save_checkpoint({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'bestPred': bestPred},
                                    is_best, sav_fol, "vnet")
                
                if not debugFlag:
                    trainF.close()
                    #validF.close()

                    writer.close()

#-------------------------------------------------------------------------------#    

if __name__ == '__main__':
    
    main()