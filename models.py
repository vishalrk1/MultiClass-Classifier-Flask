import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm

# ResNet50 Model
class ResNet(nn.Module):
    def __init__(self, num_classes, is_freeze=True):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.is_freeze = is_freeze
        self.base_model = timm.create_model('resnet50', pretrained=True)

        if self.is_freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
      
        self.base_model.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x

# EfficientNet Model
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()

        self.num_classes = num_classes
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.base_model.classifier = nn.Linear(1280, self.num_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x

# BaseLine Model
class BaseLine(nn.Module):
    def __init__(self, num_classes):
        super(BaseLine, self).__init__()

        self.Conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)
        self.Conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.Conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.Conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.Linear1 = nn.Linear(2304, 512)
        self.Linear3 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.Conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.Conv2(x)
        x = self.maxpool(x)

        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.Linear3(x)
        return x