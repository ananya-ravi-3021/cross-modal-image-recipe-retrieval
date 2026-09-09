import torch
import torch.nn as nn
from torchvision import models as models

# creating a simple CNN model
class CNN_Class(nn.Module):

    def __init__(self):
        super(CNN_Class, self).__init__()
        self.model = models.resnet101(pretrained=True)
        
        fc_in_features = self.model.fc.in_features
        
        self.model.fc = nn.Identity()
        
        self.new_fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fc_in_features, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.new_fc(self.model(x))
        return out







    
