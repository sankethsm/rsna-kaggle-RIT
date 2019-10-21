import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from apex import amp

#-------------------------------------------------------------------------------#

def trainModel(epoch, model, trainLoader, optimizer, device, debugFlag, trainF=None):
    """
    Training routine for the model
    """

    model.train()
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    finalLoss = 0
    nTotal = len(trainLoader)

    for batch_idx, dataDict in tqdm(enumerate(trainLoader), total=nTotal):
        data = dataDict['image']
        target = dataDict['label']
        data[data != data] = 0
        #target = tgt[0][0:5]
        
        data, target = data.to(device), target.type(torch.FloatTensor).to(device)
        #data, target = data.to(device), target.to(device)
        #target = target.type(torch.LongTensor)
       
        optimizer.zero_grad() 
        output = model(data)
        #print(output.shape, target.shape)
        
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        
        '''
        if torch.sum(target)==0:
            tgt = torch.max(target, 1)[1]+5
            loss = criterion(output, tgt)
        else:
            tgt = torch.max(target, 1)[1]
            loss = criterion(output, tgt)
        '''
        #print(target, tgt)
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
                
        optimizer.step()
        
        finalLoss += loss.item()
        
    #finalLoss = finalLoss.cpu().numpy()
    print('Train Epoch: {} \tTrain Loss: {:.8f}'.format(epoch, finalLoss/nTotal))
    
    nTotal = len(trainLoader)
    if not debugFlag:
        trainF.write('{},{}\n'.format(epoch, finalLoss/nTotal))
        trainF.flush()
    
    return finalLoss/nTotal