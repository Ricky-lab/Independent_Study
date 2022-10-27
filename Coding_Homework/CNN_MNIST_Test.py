# -*- Coding: utf-8 -*-
# Author: Yuehui Ruan
# Date: 9/22/22
# -*- Coding: utf-8 -*-
# Author: Yuehui Ruan
# Date: 9/4/22
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torchvision.datasets
from torch.utils.data import DataLoader
# reference the model.py
from model import *

# Prepare the dateset
train_data = torchvision.datasets.MNIST(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.MNIST(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# to check the size of dataset
train_data_size = len(train_data)
test_data_size = len(test_data)
print("The size of training set is: {}".format(train_data_size))
print("The size of testing set is: {}".format(test_data_size))

# Use Dataloader to load the dataset
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# build the model of MyNetwork
myNetwork = MyNetwork()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(myNetwork.parameters(), lr=learning_rate)

# Start training, set up parameters
total_train_steps = 0
total_test_steps = 0
epoch = 10


for i in range(epoch):
    print("----------- The {} round starts -----------".format(i+1))
    # start training
    for data in train_dataloader:
        imgs, targets = data
        outputs = myNetwork(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # finished this round training
        total_train_steps = total_train_steps + 1
        if total_train_steps % 100 == 0:
            print("Training round: {}, Loss: {}".format(total_train_steps, loss.item()))

    # How you know the accomplishment is achieved? Use test dataset
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = myNetwork(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item() # sum the total loss of this testing
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("The total loss of this time testing: {}".format(total_test_loss))
    print(("The total accuracy is: {}".format(total_accuracy/test_data_size)))

    # TenserBorad's implementation?
