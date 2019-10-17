import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
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

            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            
            output = model(data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, torch.max(target, 1)[1])
            validLoss += loss.data

            pred = output.argmax(dim=1, keepdim=True)
            validAccu += pred.eq(target.view_as(pred)).sum().item()
            

    validLoss = validLoss.cpu().numpy()
    print('Valid Epoch: {} \tValid Loss: {:.8f}\tValid Accuracy: {:.8f}'.format(
            epoch, validLoss/nTotal, 100*validAccu/nTotal))

    if not debugFlag:
        validF.write('{},{},{}\n'.format(epoch, validLoss/nTotal, validAccu/nTotal))
        validF.flush()

    return 100*validAccu/nTotal, validLoss/nTotal