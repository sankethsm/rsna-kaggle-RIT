import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
#-------------------------------------------------------------------------------#

def validModel(epoch, model, validLoader, device, debugFlag, validF=None):
    """
    Validation routine for the model
    """
    model.eval()
    validLoss = 0
    
    nTotal = len(validLoader)
    validAccu = np.zeros((len(validLoader) * 6, 1))

    with torch.no_grad():

        for i, dict_ in enumerate(tqdm(validLoader, total=nTotal)):
            data = dict_['image']
            target = dict_['label']
            #target2 = deepcopy(target)
            data[data != data] = 0

            data, target = data.to(device), target.type(torch.FloatTensor).to(device)
            
            output = model(data)

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
            validLoss += loss.item()

            validAccu[(i * 6):((i + 1) * 6)] = torch.sigmoid(output).detach().cpu().reshape((len(data) * 6, 1))
            #print(validAccu)
            # gt = np.zeros(6)
            # if tgt.cpu().numpy()[0]!=5:
            #     gt[tgt.cpu().numpy()[0]] = 1
            #     gt[5] = 1
            # pd = output.cpu().numpy()
            # validAccu += np.sqrt(np.sum(np.square(gt - pd)))            
            

    #validLoss = validLoss.cpu().numpy()
    validAccu = np.sum(validAccu)
    print('Valid Epoch: {} \tValid Loss: {:.8f}\tValid Accuracy: {:.8f}'.format(
            epoch, validLoss/nTotal, validAccu/nTotal))

    if not debugFlag:
        validF.write('{},{},{}\n'.format(epoch, validLoss/nTotal, validAccu/nTotal))
        validF.flush()

    return 100*validAccu/nTotal, validLoss/nTotal