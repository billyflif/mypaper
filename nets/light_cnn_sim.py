'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out





class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)),
                              requires_grad=True)  # nn.Parameter将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.func = nn.Softmax()  # 二分类

    def forward(self, x, s=64, m=0.5):  # s=64, m=222.5为超参数m为弧度
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)  # 传入的参数nn.Parameter在0维上进行标准化

        cosa = torch.matmul(x_norm, w_norm) / s  # torch.matmul二维的点成，高维的矩阵乘法
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))  # 代码实现公式

        return arcsoftmax





class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 保证kernel_size至少为3
        t = max(int(abs((math.log(channels, 2) + b) / gamma)), 3)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.bn = nn.BatchNorm1d(1)  # 新增BN层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.bn(y)  # 添加BN稳定数值
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v2, self).__init__()
        self.num_classes = num_classes
        self.conv1    = mfm(3, 48, 5, 1, 2)
        self.eca1 = ECALayer(48)
        self.block1   = self._make_layer(block, layers[0], 48, 48)
        self.group1   = group(48, 96, 3, 1, 1)
        self.eca2 = ECALayer(96)
        self.block2   = self._make_layer(block, layers[1], 96, 96)
        self.group2   = group(96, 192, 3, 1, 1)
        self.eca3 = ECALayer(192)
        self.block3   = self._make_layer(block, layers[2], 192, 192)
        self.group3   = group(192, 128, 3, 1, 1)
        self.eca4 = ECALayer(128)
        self.block4   = self._make_layer(block, layers[3], 128, 128)
        self.group4   = group(128, 128, 3, 1, 1)
        self.fc       = nn.Linear(8*8*128, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-05)
        # self.fc2 = nn.Linear(256, 512, bias=False)
        # self.fc2      = ArcMarginProduct(256,num_classes, s=32.0, m=0.1)
        # self.fc2 = Arcsoftmax(512, num_classes)
        self._initialize_weights()


    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d):  # 新增1D卷积初始化
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'filter'):  # 初始化MFM模块内的卷积
                if isinstance(m.filter, nn.Conv2d) or isinstance(m.filter, nn.Linear):
                    nn.init.kaiming_uniform_(m.filter.weight)

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected in input")
            raise ValueError
        x = self.conv1(x)
        x = self.eca1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = self.eca2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = self.eca3(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.eca4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        # if label is None:
        #     return fc,torch.tensor(np.zeros((1,self.num_classes),dtype=np.float32))
        # else:
        #     out = self.fc2(x,label)
        #     return fc,out
        out = self.features(x)
        return out


def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model





