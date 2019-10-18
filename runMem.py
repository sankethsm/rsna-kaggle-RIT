# Define a model
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision.models import resnet50
from utils_.memoryEstimator import SizeEstimator

'''
class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

    def forward(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        return h

model = Model()
'''

model = resnet50(pretrained=False)
model.fc = nn.Linear(2048, 6)


# Estimate Size

se = SizeEstimator(model, input_size=(1,3,256,256))
print(se.estimate_size())

# Returns
# (size in megabytes, size in bits)
# (408.2833251953125, 3424928768)

print(se.param_bits) # bits taken up by parameters
print(se.forward_backward_bits) # bits stored for forward and backward
print(se.input_bits) # bits for input