#!/usr/bin/env python3
"""
Jason Stranne

"""
import numpy as np
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import gc
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
 
from Relative_Positioning import RelPosNet


import torch



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, tpos=0, tneg=0):
        'Initialization'
        xpath=path+"_RPdata.npy"
        ypath=path+"_RPlabels.npy"
        self.data = np.load(xpath)
        self.labels = np.load(ypath)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X1 = torch.from_numpy(self.data[index,0,:,:]).float()
        X2 = torch.from_numpy(self.data[index,0,:,:]).float()
        y = torch.from_numpy(self.labels[index]).float()

        return X1, X2, y
    
    
root = os.path.join("..","training", "")
recordName="tr03-0078"
data_file=root+recordName+os.sep+recordName

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 2

# Generators
print('Loading Data')
training_set = Dataset(data_file)
training_generator = torch.utils.data.DataLoader(training_set, **params)

print("len of the dataloader is:",len(training_generator))

# cuda setup if allowed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = RelPosNet().to(device)

#defining training parameters
loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
learning_rate = 5e-4
beta_vals = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

# Notes:
#     We need to train 2000 per file
#     tneg and tpos as a hyperparam

print("Start Training")
loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
learning_rate = 5e-4
beta_vals = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

# t_neg=0
# t_pos=0
for epoch in range(20):
    running_loss=0
    for X1,X2, y in training_generator:
        # Transfer to GPU
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        #print(X1.shape)
        y_pred = model(X1, X2)
        loss = loss_fn(y_pred, y)
        print("batch:", loss.item())
        
        #zero gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        running_loss+=loss.item()
    print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss/len(training_generator)))