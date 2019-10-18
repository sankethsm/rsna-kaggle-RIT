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
    validAccu = 0
    nTotal = len(validLoader)
    with torch.no_grad():

        for dict_ in tqdm(validLoader, total=nTotal):
            data = dict_['image']
            target = dict_['label']
            target2 = deepcopy(target)
            target = target[0:5]


            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            
            output = model(data)

            criterion = nn.CrossEntropyLoss()

            if torch.sum(target)==0:
                tgt = torch.max(target, 1)[1]+5
                loss = criterion(output, tgt)
            else:
                tgt = torch.max(target, 1)[1]
                loss = criterion(output, tgt)

            validLoss += loss.data

            gt = np.zeros(6)
            if tgt.cpu().numpy()[0]!=5:
                gt[tgt.cpu().numpy()[0]] = 1
            else:
                gt[tgt.cpu().numpy()[0]] = 1
                gt[5] = 1
            pd = output.cpu().numpy()
            validAccu += np.sqrt(np.sum(np.square(gt - pd)))            
            

    validLoss = validLoss.cpu().numpy()
    print('Valid Epoch: {} \tValid Loss: {:.8f}\tValid Accuracy: {:.8f}'.format(
            epoch, validLoss/nTotal, 100*validAccu/nTotal))

    if not debugFlag:
        validF.write('{},{},{}\n'.format(epoch, validLoss/nTotal, validAccu/nTotal))
        validF.flush()

    return 100*validAccu/nTotal, validLoss/nTotal