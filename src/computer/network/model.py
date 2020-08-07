"""Model of the network. 

Raises:
    NotImplementedError: Raised if the forward function  is not implemented.
"""
import torch
import os
if os.name == 'nt':
    # https://discuss.pytorch.org/t/torch-cat-runtimeerror-error-in-loadlibrarya/71188/9
    import ctypes
    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from abc import abstractmethod

class Net(nn.Module):
    def __init__(self, num_classes=7, num_conditions=3):
        """Inits the model. The network consists of a pretrained resnet50 followed by
            a linear layer and multiple similar neural networks after that. One for each condition.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 7.
            num_conditions (int, optional): Number of conditions. Defaults to 3. (forward, left,right)
        """        
        super().__init__()

        gpu = torch.device('cuda')

        # freeze weights of the vgg layers
        self.resnet_model = models.resnet50(pretrained=True, progress=True)

        ct = 0 
        for child in self.resnet_model.children():
            ct += 1
            if ct < 7: #freeze the first 6 children
                for param in child.parameters():
                    param.requires_grad = False

        self.resnet_model.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc = nn.ModuleList()

        for i in range(num_conditions):
            self.fc.append(nn.Sequential(
                nn.Linear(in_features=1024, out_features=512, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(512, num_classes)))

        self.to(gpu)

    @abstractmethod
    def forward(self):
        """Forward method of Network

        Raises:
            NotImplementedError: Method should be implemented
        """        
        raise NotImplementedError()





