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
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Temporal_Shuffling import TemporalShufflingNet
import torch


class Custom_RP_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, path, total_points, tpos, tneg, windowSize, sfreq):
        #'Initialization'
        datapath=path+"_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath=path+"_Windowed_StartTime.npy"
        self.start_times = np.load(datapath)
        print(self.data.shape)
        self.total_windows = len(self.data)
        self.trios, self.labels = self.get_pairs_and_labels(size=total_points, tpos=tpos, tneg=tneg, windowSize=windowSize)
    def __len__(self):
        'Denotes the total number of samples'
        
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X1 = torch.from_numpy(self.data[self.trios[index,0],:,:]).float()
        X2 = torch.from_numpy(self.data[self.trios[index,1],:,:]).float()
        X3 = torch.from_numpy(self.data[self.trios[index,2],:,:]).float()
        y = torch.from_numpy(np.array([self.labels[index]])).float()

        return X1, X2, X3, y
    
    def get_pairs_and_labels(self, size, tpos, tneg, windowSize):
        # pairs=np.zeros((size,3),dtype=int)
        #label=np.zeros(size)
        
        pairs = []
        labels = []
        
        # order is t, t', t''
        
        for i in range(size):
            tempval=np.random.randint(low=0, high=self.total_windows)
            
            secondval = self.return_pos_index(index=tempval, tpos=tpos, windowSize=windowSize)
            
            if(np.abs(self.start_times[tempval]-self.start_times[secondval])>tpos):
                    print("skipping bad label")
                    continue
            
            if random.random() < 0.5:
                outval = 1
                # we need to check if its impossible to return a pos label
                if(np.abs(secondval-tempval)<=1):
                    print("skipping since it is impossible to return a positive label")
                    continue
                print("lowval", tempval)
                print("highval", secondval)
                unknown_val=np.random.randint(low=tempval+1, high=secondval)
                    
            else:
                outval = -1
                unknown_val = self.return_neg_index(tempval, tneg, windowSize)
                # print("neg",tempval, secondval)
                # No need to check for mistakes since we cant return a bad negative window, still check
                if(np.abs(self.start_times[tempval]-self.start_times[unknown_val])<tneg):
                    print("ERROR, messed up neg label")
                    continue
                
            
            pairs.append([tempval, unknown_val, secondval])
            labels.append(outval)
            pairs.append([secondval, unknown_val, tempval])
            labels.append(outval)
            
        
        pairs = np.array(pairs)
        labels=np.array(labels)
        print(labels.shape)
        return pairs, labels
    
    def return_pos_index(self, index, tpos, windowSize):
        # tpos and windowSize in seconds
        #print("windowsize", windowSize)
        #print("tpos", tpos)
        # minimum = max(0,index-(tpos//windowSize))
        maximum = min(len(self.data),index+(tpos//windowSize)+1) #since non inclusive
        #print("min", minimum)
        #print("max", maximum)
        return np.random.randint(index, maximum)
    
    def return_neg_index(self, index, tneg, windowSize):
        midlow=max(0,index-(tneg//windowSize))
        midhigh =  min(len(self.data)-1,index+(tneg//windowSize))
        # print("midlow", midlow)
        # print("midhigh", midhigh)
        assert (midlow>0 or midhigh<len(self.data))
        # check if it is even possible to return a negative index
        trial = np.random.randint(0, len(self.data))
        while(trial >= midlow and trial <= midhigh):
            # keep trying
            trial = np.random.randint(0, len(self.data))
        return trial
           
        
def num_correct(ypred, ytrue):
    return ((ypred* ytrue) > 0).float().sum().item()

    
root = os.path.join("..","training", "")

datasets_list=[]
print('Loading Data')
f=open(os.path.join("..","training_names.txt"),'r')
lines = f.readlines()
for line in lines:
    recordName=line.strip()
    print('Processing', recordName)
    data_file=root+recordName+os.sep+recordName
    datasets_list.append(Custom_RP_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=30, sfreq=100))
f.close()



# recordName="tr03-0078"
# data_file=root+recordName+os.sep+recordName
# training_set=Custom_RP_Dataset(path=data_file, total_points=2000, tpos=120, tneg=300, windowSize=30, sfreq=100)

training_set = torch.utils.data.ConcatDataset(datasets_list)

print("one dataset is", len(datasets_list[0]))

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100
training_generator = torch.utils.data.DataLoader(training_set, **params)

print("len of the dataloader is:",len(training_generator))

# cuda setup if allowed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = TemporalShufflingNet().to(device)

#defining training parameters
print("Start Training")
loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
learning_rate = 5e-4
beta_vals = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)

# t_neg=0
# t_pos=0
for epoch in range(max_epochs):
    running_loss=0
    correct=0
    total=0
    for X1,X2,X3, y in training_generator:
        #print(X1.shape)
        #print(y.shape)
        # Transfer to GPU
        X1, X2, X3, y = X1.to(device), X2.to(device), X3.to(device), y.to(device)
        #print(X1.shape)
        y_pred = model(X1, X2, X3)
        loss = loss_fn(y_pred, y)
        
        #calculate accuracy
        correct += num_correct(y_pred,y)
        total += len(y)
        
        #print("batch:", loss.item())
        
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
    print('[Epoch %d] accuracy: %.3f' %
                      (epoch + 1, correct/total))
    
    
    

print(model.stagenet)
stagenet_save_path = os.path.join("..", "models", "TS_stagernet.pth")
torch.save(model.stagenet.state_dict(), stagenet_save_path)
