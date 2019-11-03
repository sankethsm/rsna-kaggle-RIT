#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:42:34 2019

@author: aneesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Linear Scheduler courtesy of https://github.com/miguelvr/dropblock
class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.keep_prob = 1.0 - self.drop_values[self.i]
        self.i += 1
        
class DropBlock(nn.Module):
    def __init__(self, keep_prob = 1.0, dropblock_size = 3):
        super(DropBlock, self).__init__()
        self.keep_prob = keep_prob
        self.dropblock_size = dropblock_size
    
    def forward(self, x):
        if not self.training or self.keep_prob == 1.0:
            return x
        else:
            assert x.dim() == 4, \
            "Expected input with 4 dimensions (B, C, H, W)"
            
            numbatch, channel, height, width = x.shape
            assert height == width, \
            " Input tensor with width! = height not supported"
            
            dropblock_size = min(self.dropblock_size, width)
            seed_drop_rate = (1.0 - self.keep_prob) * width**2 / \
            dropblock_size**2 / (width - dropblock_size + 1)**2
                    
            w_i, h_i = torch.meshgrid(torch.arange(width), torch.arange(width))
            valid_block_center = (w_i >= int(dropblock_size // 2)) \
                                  & (w_i < width - (dropblock_size - 1) // 2) \
                                  & (h_i >= int(dropblock_size // 2)) \
                                  & (h_i < width - (dropblock_size - 1) // 2)
                                  
            valid_block_center = torch.unsqueeze(valid_block_center, 0)
            valid_block_center = torch.unsqueeze(valid_block_center, 0)
            valid_block_center = valid_block_center.float()
            
            randnoise = torch.rand(x.shape).float()
            
            block_pattern = (1 - valid_block_center + 
                             (1 - seed_drop_rate) + randnoise) >= 1
            block_pattern = block_pattern.float()
            
            if dropblock_size == width:
                block_pattern = block_pattern.view(numbatch, channel, height*width)
                bcz, _ = torch.min(block_pattern, dim = 2, keepdim = True)
                block_pattern = bcz.view(numbatch, channel, 1, 1)
                
            else:
                block_pattern = -F.max_pool2d(
                        input = -block_pattern, 
                        kernel_size = (dropblock_size, dropblock_size), 
                        stride = (1,1),
                        padding = (dropblock_size//2, dropblock_size//2))
            
            block_pattern = block_pattern.to(x.device)
            percent_ones = block_pattern.sum()/block_pattern.numel()
            x = x / percent_ones * block_pattern
            
            #To prevent channel dropout to go NaNs
            x[x!=x] = 0.
            
            return x
    
if __name__ == "__main__":
    device = 'cuda'
    net = DropBlock(keep_prob=1.0, dropblock_size=3).to(device)
#    net.eval()
    x = torch.ones((1, 4, 6, 6)).to(device)
    y = net(x)
    print(y)
