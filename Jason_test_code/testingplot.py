#!/usr/bin/env python3
"""
Jason Stranne
"""
import numpy as np
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import gc
import random
from matplotlib import pyplot as plt 

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Relative_Positioning import RelPosNet
import torch

data = np.load(".."+os.sep+"training"+os.sep+"tr03-0078"+os.sep+"tr03-0078_Windowed_Preprocess.npy")
sample = data[0]
print("sample",sample.shape)
samplechannel=sample[:,0]
print("sample channel", samplechannel.shape)
plt.plot(samplechannel)
