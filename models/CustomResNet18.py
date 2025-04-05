import torch
import torch.nn as nn
from torchvision import  models
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=3):  # Example: 3 classes for shoe/sandal/boot
        super(CustomResNet18, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze all layers except final layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


