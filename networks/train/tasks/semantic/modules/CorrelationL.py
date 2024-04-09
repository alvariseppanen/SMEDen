# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
from traceback import print_tb
#from matplotlib import test
import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import time

class MultiEchoNeighborBlock(nn.Module):

    def __init__(self, in_chans=3, stem_size=32, n_echoes=2):
        super(MultiEchoNeighborBlock, self).__init__()
        self.n_echoes = n_echoes
        kernel_knn_size = 3 # 3
        self.search = 7 # 5
        self.range_weight = Parameter(torch.Tensor(in_chans, stem_size, kernel_knn_size*kernel_knn_size*n_echoes+n_echoes))
        init.kaiming_uniform_(self.range_weight, a = math.sqrt(5))
        self.out_channels = stem_size
        self.knn_nr = kernel_knn_size ** 2
        self.act1 = nn.LeakyReLU(inplace=True)

    def Neighbor_conv(self, inputs):
        B, H, W = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]
        search_dim = self.search ** 2
        pad = int((self.search - 1) / 2)

        first_range = (inputs[:, 0:1, ...].clone())
        first_points = (inputs[:, self.n_echoes:self.n_echoes+3, ...].clone())
        
        first_unfold_range = F.unfold(first_range,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))

        first_unfold_points = F.unfold(first_points,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))

        first_unfold_points = first_unfold_points.view(B, 3, search_dim, H*W)

        unfold_range = torch.zeros((B, 0, H*W)).cuda()
        
        for echo in range(self.n_echoes):
            n_range = (inputs[:, echo:echo+1, ...].clone())
            n_points = (inputs[:, self.n_echoes+echo*3:self.n_echoes+echo*3+3, ...].clone())

            n_points = n_points.flatten(start_dim=2).unsqueeze(dim=2)
            n_distance = torch.linalg.norm(n_points - first_unfold_points, dim=1)
            _, n_knn_index = n_distance.topk(self.knn_nr, dim=1, largest=False)
            n_unfold_range = torch.gather(input=first_unfold_range, dim=1, index=n_knn_index)
            unfold_range = torch.cat((unfold_range, n_unfold_range, n_range.flatten(start_dim=2)), dim=1)

        output = torch.matmul(self.range_weight, unfold_range).view(B, self.out_channels, H, W)

        return output

    def forward(self, x):
        x = self.Neighbor_conv(x)
        x = self.act1(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class CorrL(nn.Module):
    def __init__(self, nclasses, params, n_echoes=2):
        super(CorrL, self).__init__()
        
        self.MultiEchoNeighborBlock = MultiEchoNeighborBlock(1, 2 * 32, n_echoes)
        self.resBlock1 = ResBlock(2 * 32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=False)
        self.upBlock = UpBlock(2 * 2 * 32, 32, 0.2, drop_out=False)
        self.logits = nn.Conv2d(32, n_echoes, kernel_size=(1, 1))

    def forward(self, x):
        MultiEchoNeighborBlock = self.MultiEchoNeighborBlock(x)
        down, down_skip = self.resBlock1(MultiEchoNeighborBlock)
        bottom = self.resBlock2(down)
        up = self.upBlock(bottom, down_skip)
        logits = self.logits(up)

        return logits
