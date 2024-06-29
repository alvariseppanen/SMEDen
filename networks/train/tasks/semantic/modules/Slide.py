# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiLaBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(LiLaBlock, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(7, 3), stride=1, padding=(3, 1))
        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv3 = nn.Conv2d(in_filters, out_filters, kernel_size=(2, 2), dilation=2, stride=1, padding=(1, 1))
        self.conv4 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 7), stride=1, padding=(1, 3))
        self.conv5 = nn.Conv2d(out_filters*4, out_filters, kernel_size=(1, 1), stride=1)
        self.bn5 = nn.BatchNorm2d(out_filters)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv1 = self.act(conv1)
        conv2 = self.conv2(x)
        conv2 = self.act(conv2)
        conv3 = self.conv3(x)
        conv3 = self.act(conv3)
        conv4 = self.conv4(x)
        conv4 = self.act(conv4)

        x = torch.cat((conv1, conv2, conv3, conv4), dim=1)

        x = self.conv5(x)
        x = self.act(x)
        x = self.bn5(x)

        return x


class RDNet(nn.Module):
    def __init__(self, nclasses, params):
        super(RDNet, self).__init__()
        self.nclasses = nclasses

        self.lilaBlock1 = LiLaBlock(2, 64)
        self.lilaBlock2 = LiLaBlock(64, 64)
        self.lilaBlock3 = LiLaBlock(64, 64)
        self.lilaBlock4 = LiLaBlock(64, 1)

    def forward(self, x):
        
        x = (x[:, [0,4], ...].clone()) # range, intensity
        x = self.lilaBlock1(x)
        x = self.lilaBlock2(x)
        x = self.lilaBlock3(x)
        x = self.lilaBlock4(x)
        
        return x
    

class PRNet(nn.Module):
    def __init__(self, nclasses, params):
        super(PRNet, self).__init__()
        self.nclasses = nclasses

        self.lilaBlock1 = LiLaBlock(2, 64)
        self.lilaBlock2 = LiLaBlock(64, 64)
        self.lilaBlock3 = LiLaBlock(64, 64)
        self.lilaBlock4 = LiLaBlock(64, 1)
        self.lilaBlock5 = LiLaBlock(64, 64)
        self.lilaBlock6 = LiLaBlock(64, 1)
        self.lilaBlock7 = LiLaBlock(64, 64)
        self.lilaBlock8 = LiLaBlock(64, 1)

    def forward(self, x, binary_mask):

        x = (x[:, [0,4], ...].clone()) # range, intensity
        x = self.lilaBlock1(x * ~binary_mask)
        x = self.lilaBlock2(x)

        # 3 hypotheses produced best results
        x1 = self.lilaBlock3(x)
        x1 = self.lilaBlock4(x1)
        x2 = self.lilaBlock5(x)
        x2 = self.lilaBlock6(x2)
        x3 = self.lilaBlock7(x)
        x3 = self.lilaBlock8(x3)

        return torch.cat((x1[:,None,:,:,:], x2[:,None,:,:,:], x3[:,None,:,:,:]), dim=1)
