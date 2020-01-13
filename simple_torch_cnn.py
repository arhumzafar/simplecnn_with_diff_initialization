import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        #find padding using below equation or in attached image!
        #  (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w +k)/2

        # 28x28x1 -> 28x28x4
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size = (3,3), stride = (1,1), padding = 1)

        # 28x28x4 -> 14x14x4
        self.pool_1 = torch.nn.MaxPool2d(kernel_size(2,2), stride=(2,2), padding =0)

        # 14x14x4 -> 14x14x8
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size = (3,3), stride = (1,1), padding = 1)

        # 28x28x4 -> 7x7x8
        self.pool_2 = torch.nn.MaxPool2d(kernel_size(2,2), stride=(2,2), padding =0)

        self.linear_1 = torch.nn.Linear(7*7*8, num_classes)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)
        
        logits = self.linear_1(out.view(-1, 7*7*8))
        probas = F.softmax(logits, dim=1)
        return logits, probas


