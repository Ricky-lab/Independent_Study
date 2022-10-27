# -*- Coding: utf-8 -*-
# Author: Yuehui Ruan
# Date: 10/18/22
import time

import torch.cuda
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import res_net_model
# Import the resNet model, and implement the instance of ResNet18
resNet18_1 = res_net_model.ResNet18()
resNet18_2 = res_net_model.ResNet18()
resNet18_3 = res_net_model.ResNet18()
resNet18_4 = res_net_model.ResNet18()
resNet18_5 = res_net_model.ResNet18()

#prepare the dataset from train and test
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=False)

#data size
train_data_size = len(train_data)
test_data_size = len(test_data)
print("The size of training set is: {}".format(train_data_size))
print("The size of testing set is: {}".format(test_data_size))

#data loader
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)


#Selecting device:
if torch.cuda.is_available():
    resNet18_1 = resNet18_1.cuda()
    resNet18_2 = resNet18_2.cuda()
    resNet18_3 = resNet18_3.cuda()
    resNet18_4 = resNet18_4.cuda()
    resNet18_5 = resNet18_5.cuda()

#Optimizer:
    #1. learning rate
learning_rate = 1e-2
    #2. Choosing different gradiant descend algorithm
optimizer1 = torch.optim.SGD    (resNet18_1.parameters(), lr=learning_rate)
optimizer2 = torch.optim.ASGD   (resNet18_2.parameters(), lr=learning_rate)
optimizer3 = torch.optim.Adagrad(resNet18_3.parameters(), lr=learning_rate)
optimizer4 = torch.optim.RMSprop(resNet18_4.parameters(), lr=learning_rate)
optimizer5 = torch.optim.Adam   (resNet18_5.parameters(), lr=learning_rate)

#loss function
# Cross entropy
loss_func = nn.CrossEntropyLoss()
# hinge loss
loss_func1 = nn.HingeEmbeddingLoss()
# MSE loss
loss_func2 = nn.MSELoss()
# MAE
loss_func3 = nn.MSELoss()
# Huber Loss
loss_func4 = nn.HuberLoss()

if torch.cuda.is_available():
    loss_func = loss_func.cuda()
    #loss_func1 = loss_func1.cuda()
    #loss_func2 = loss_func2.cuda()
    #loss_func3 = loss_func3.cuda()
    #loss_func4 = loss_func4.cuda()

#Setting some feature
total_train_step = 0
total_test_step = 0
epoch = 100

plt.xlabel("X: Epcho that it is trained")
plt.ylabel("Y: The loss value")
for i in range(epoch):
    print("\n---- The {}th round training starts".format(i + 1))

    #start training
    resNet18_1.train()
    start_time = time.time()
    for data in train_data_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = resNet18_1(imgs)
        loss = loss_func(outputs, targets)

        #Different loss_funcs with same Optimizer: grad_decn
        optimizer1.zero_grad()

        loss.backward()

        optimizer1.step()

        total_train_step += 1
        if total_train_step %200 == 0:
            end_time = time.time()
            print("---- Time used: {}".format(end_time - start_time))
            print("---- Training times: {}, loss: {}".format(total_train_step, loss.item()))
        #end training in this epoch

    #start testing
    resNet18_1.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = resNet18_1(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

            #count the accuracy
            correct = (outputs.argmax(1) == targets).sum()
            total_correct = correct + total_correct

    print("Total loss of whole dataset is : {}".format(total_test_loss))
    accur_rate = total_correct / test_data_size
    plt.scatter(i, accur_rate.item(), color = "red")
    print("The accuracy of the testing dataset is : {}".format(total_correct / test_data_size))

    total_test_step = total_test_step + 1

plt.scatter()

