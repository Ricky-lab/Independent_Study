# -*- Coding: utf-8 -*-
# Author: Yuehui Ruan
# Date: 9/13/22
'''
注：梯度下降算法的loss函数并不是随着迭代次数的增加而变小，而且loss函数（损失函数）的大小与初始值k，b，lr的选取有很大关系，
所以梯度下降算法只适用于loss函数为凸函数的情况。
'''
import numpy as np
import matplotlib.pyplot as plt

# 1. 导入数据（两列）
data=np.genfromtxt('\jupyter\data\Algorithm data\linear regression data.csv',delimiter=',')
x_data=data[:,0]   # 冒号前后：从第0行取到最后一行，逗号后面0：只要第一列
y_data=data[:,1]
plt.scatter(x_data,y_data)  #  散点图
plt.show()
# 学习率learning rate（步长）、截距、斜率、最大迭代次数
lr=0.0005
b=1
k=0
epochs=5000
# 2. Loss Function（最小二乘法）:该函数只返回一个值
def compute_error(b,k,x_data,y_data):
    totalError=0
    for i in range(0,len(x_data)):
        totalError+=(y_data[i]-(k*x_data[i]+b))**2
    return totalError/float(len(x_data))
# 3. 梯度下降算法函数
def gradient_descent_runner(x_data,y_data,b,k,lr,epochs):
    # 总数据量
    m=float(len(x_data))
    # 迭代epochs次
    for i in range(epochs):
        b_grad=0
        k_grad=0
        # 计算每个样本对应的真实值和预估值的差距（也就是误差）
        for j in range(0,len(x_data)):
            b_grad+= -(1/m)*(y_data[j]-((k*x_data[j])+b))
            k_grad+= -(1/m)*x_data[j]*(y_data[j]-((k*x_data[j])+b))
        # 更新b和k
        b= b-(lr*b_grad) # theta 0
        k= k-(lr*b_grad) # theta 1
        # 每迭代500次，输出一次图像和数据
        if i%500 ==0:
            print('epochs：',i)
            print('b:',b,'k:',k)
            print('error:',compute_error(b,k,x_data,y_data),)
            plt.plot(x_data,y_data,'b')
            plt.plot(x_data,k*x_data+b,'r')
            plt.show()
    return b,k
gradient_descent_runner(x_data,y_data,b,k,lr,epochs)