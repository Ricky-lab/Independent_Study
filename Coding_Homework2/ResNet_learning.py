# -*- Coding: utf-8 -*-
# Author: Yuehui Ruan
# Date: 10/4/22
# Using torch.nn.module to implement the ResNet18
# DataSet: CIFAR-10
import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# First of all, device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the dataset
train_data = torchvision.datasets.CIFAR10(root="../dataset", train= True, transform= torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train= False, transform= torchvision.transforms.ToTensor(),
                                          download=True)

# Get the size of dataset
train_data_size = len(train_data)
test_data_size = len(test_data)
print("The size of training set is: {}".format(train_data_size))
print("The size of testing set is: {}".format(test_data_size))

# Dateloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Build the neural network for training
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()
        self.out_channels = out_channels
        #If there is shortcut, we define conv1 and shortcut
        if downsample:
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        else:
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

# Build the ResNet18
# Reference:
# https://www.google.com/search?q=resNet18&source=lnms&tbm=isch&sa=X&ved=2ahUKEwj4z_SB7sv6AhWaFlkFHSdEARYQ_AUoAXoECAMQAw&biw=1633&bih=794&dpr=1.8#imgrc=JZveLIKqPlJGTM
class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input)
        input = self.fc(input)

        return input



#Training and testing
#Testing Output ****
if __name__ == '__main__':
    net = ResNet18().to(device)
    input = torch.rand(1, 3, 64, 64)
    output = net(input)
    print(output.shape)


