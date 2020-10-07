"""
Jason Stranne
"""
import numpy as np
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import random

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Relative_Positioning import RelPosNet
import torch


class Custom_CPC_Dataset(torch.utils.data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, path, total_points, Nc, Np):
        # 'Initialization'
        datapath = path + "_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath = path + "_Windowed_StartTime.npy"
        self.start_times = np.load(datapath)
        print(self.data.shape)
        self.total_windows = len(self.data)

        #set Nc and Np
        self.Nc = Nc
        self.Np = Np
        #get starting points
        self.Xc_starts = self.getXc_starts(total_points, self.Nc + self.Np)


    def __len__(self):
        'Denotes the total number of samples'

        return len(self.Xc_starts)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        # X1 = torch.from_numpy(self.data[self.pairs[index, 0], :, :]).float()
        # X2 = torch.from_numpy(self.data[self.pairs[index, 1], :, :]).float()
        # y = torch.from_numpy(np.array([self.labels[index]])).float()
        Xc = torch.from_numpy(self.data[self.Xc_starts[index]:self.Xc_starts[index]+self.Nc, :, :]).float()
        Xp = torch.from_numpy(self.data[self.Xc_starts[index] + self.Nc:self.Xc_starts[index] + self.Nc + self.Np, :, :]).float()
        Xb = []

        for i in range(self.Xc_starts[index]+self.Nc, self.Xc_starts[index] + self.Nc+self.Nb):
            Xb.append(self.generate_negative_sample_list(self.Xc_starts[index]))

        Xb = torch.from_numpy(np.array(Xb)).float()
        return Xc, Xp, Xb


    def getXc_starts(self, total_points, buffer_needed):
        startList = []
        for i in range(total_points):
            startList.append(np.random.randInt(low=0, high=self.total_windows-buffer_needed))
        return np.array(startList)


    def generate_negative_sample_list(self, xc_start):
        toReturn = []
        for i in range(self.Nb):
            toReturn.append(self.random_Nb_Sample(xc_start))
        return toReturn


    def random_Nb_Sample(self, xcStart):
        num = xcStart # will cause while loop to run
        count = 0
        while xcStart <= num <= (xcStart + self.Nc + self.Np):
            num = np.random.random(low=0, high=self.total_windows)
            count += 1
            if count > 1000:
                raise Exception("impossible to find a valid Nb sample, need to debug")

        if count == 0:
            raise Exception("never found an Nb, need to debug")
        # cant be in the range (start+Nc+Np)
        return self.data[num, :, :]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Stager_net_pratice import StagerNet


class CPC_Net(nn.Module):
    def __init__(self,):
        super(CPC_Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 2, (2,1), stride=(1,1))

        # we want 2 filters?
        self.stagenet = StagerNet()
        self.linear = nn.Linear(100, 1)
        Np=10
        h_dim=100
        ct_dim=100
        self.NpList = []
        for i in range(Np):
            self.NpList.append(nn.Bilinear(in1_features=h_dim, in2_features=ct_dim, out_features=1, bias=False))

        self.logsoftmax = nn.LogSoftmax()


    def forward(self, Xc, Xp, Xb_array):
        x1 = self.stagenet(x1)
        x2 = self.stagenet(x2)

        # the torch.abs() is able to emulate the grp
        x1 = self.linear(torch.abs(x1 - x2))
        return x1


total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
nce += torch.sum(torch.diag(self.logsoftmax(total)))  # nce is a tensor

import torch
import torch.nn as nn
import numpy as np


def InfoLoss():
    lsoft = nn.LogSoftmax(dim=1)
    a = torch.from_numpy(np.array([[1, 100, 100], [0.2, 1000, 3], [0.2, 6, 55]]))
    b = -lsoft(a)
    c = np.sum(np.diag(b))