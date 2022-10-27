# -*- Coding: utf-8 -*-
# Author: Yuehui Ruan
# Date: 9/5/22
# establish the NN
import torch
from torch import nn
import torch.nn.functional as F

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3) # 26 * 26 * 10
        self.conv2 = torch.nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3) # 24 * 24 * 20
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3) # 22 * 22 * 40
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2) # 11 * 11 * 40
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2) # 5 * 5 * 40
        self.pooling3 = torch.nn.MaxPool2d(kernel_size=2) # 2 * 2 * 40
        self.linear1 = torch.nn.Linear(40,32)  #想确定40这个值？是和
        self.linear2 = torch.nn.Linear(32,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pooling3(x)
        x = x.view(x.size(0), -1)  # Flatten 改变张量形状
        #print(x.size(-1))
        # 此时 x.sixe() [64,40] 对应liner1中的40，具体linear1的40读者可以算出来，也可以采用偷懒的方法，运行代码，由print(x.size(-1))确定
        x = self.linear1(x)
        x = self.linear2(x)
        return x #最后一层不做激活，因为下一步输入到交叉损失函数中，交叉熵包含了激活层

# test the correctness
if __name__ == '__main__':
    myNetwork = MyNetwork()
    input = torch.ones((64, 3, 28, 28))
    output = myNetwork(input)
    print(output.shape)

