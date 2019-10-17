import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

#-------------------------------------------------------------------------------#

def trainModel(epoch, model, trainLoader, optimizer, device, debugFlag, trainF=None):
    """
    Training routine for the model
    """

    model.train()
    finalLoss = 0
    nTotal = len(trainLoader)

    for batch_idx, dataDict in tqdm(enumerate(trainLoader), total=nTotal):
        data = dataDict['image']
        target = dataDict['label']
        
        data, target = data.to(device), target.type(torch.LongTensor).to(device)
        #target = target.type(torch.LongTensor)
       
        optimizer.zero_grad() 
        output = model(data)
        #print(output.shape, target.shape)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, torch.max(target, 1)[1])
        
        loss.backward()
                
        optimizer.step()
        
        finalLoss += loss.data
        
    finalLoss = finalLoss.cpu().numpy()
    print('Train Epoch: {} \tTrain Loss: {:.8f}'.format(epoch, finalLoss/nTotal))
    
    nTotal = len(trainLoader)
    if not debugFlag:
        trainF.write('{},{},{}\n'.format(epoch, finalLoss/nTotal))
        trainF.flush()
    
    return finalLoss/nTotal