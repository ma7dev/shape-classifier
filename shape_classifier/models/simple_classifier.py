import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models 

class Model(nn.Module):
    def __init__(self, num_classes: int):
        super(Model, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.classifier=nn.Sequential(
            nn.Linear(9216,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor()) -> torch.Tensor():
        return self.model(x)