#搭建神经网络
import torch
from torch import nn


class Neuro(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(64,10)

        )

    def forward(self,x):
        x=self.model(x)
        return x




