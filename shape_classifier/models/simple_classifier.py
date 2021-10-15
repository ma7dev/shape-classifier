import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models 

class Model(nn.Module):
    def __init__(self, num_classes: int):
        super(Model, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.classifier=nn.Sequential(
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor()) -> torch.Tensor():
        x = self.model(x)
        x = self.model.classifier(x)
        return x