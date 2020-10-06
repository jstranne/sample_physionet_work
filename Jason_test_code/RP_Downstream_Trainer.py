#!/usr/bin/env python3
"""
Jason Stranne
"""
import numpy as np
import os
import sys
import gc
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from torchsummary import summary
from Stager_net_pratice import StagerNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownstreamNet(nn.Module):
    def __init__(self, trained_stager):
        super(DownstreamNet, self).__init__()
        self.stagenet=trained_stager
        self.linear = nn.Linear(100,5) # 5 labels
        
    def forward(self, x):
        x = self.stagenet(x)
        x = self.linear(x)
        return x

class Downstream_Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, path):
        datapath = path+"_Windowed_Preprocess.npy"
        self.data = np.load(datapath)
        datapath = path+"_Windowed_SleepLabel.npy"
        self.labels = np.load(datapath)
        
        #need to removed the -1 labels (unknown)
        unknown=np.where(self.labels<0)
        self.labels=np.delete(self.labels,unknown)
        self.data=np.delete(self.data,unknown, axis=0)
        print("labels shape", self.labels.shape)
        print("data shape", self.data.shape)
        print("removed", len(unknown[0]), "unknown entries")
        
        
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.from_numpy(self.data[index,:,:]).float()
        Y = torch.from_numpy(np.array(self.labels[index])).long()
        return X, Y

    
def print_class_counts(y_pred):
        zero = (torch.argmax(y_pred, dim=1)==0).float().sum()
        print("zero", zero)
        one = (torch.argmax(y_pred, dim=1)==1).float().sum()
        print("one", one)
        two = (torch.argmax(y_pred, dim=1)==2).float().sum()
        print("two", two)
        three = (torch.argmax(y_pred, dim=1)==3).float().sum()
        print("three", three)
        four = (torch.argmax(y_pred, dim=1)==4).float().sum()
        print("four", four)
    

def num_correct(ypred, ytrue):
    #print(ypred)
    #print(torch.argmax(ypred, dim=1))
    #torch.argmax(a, dim=1)
    return (torch.argmax(ypred, dim=1)==ytrue).float().sum().item()
    
    
root = os.path.join("..","training", "")

datasets_list=[]
print('Loading Data')
f=open(os.path.join("..","training_names.txt"),'r')
lines = f.readlines()
for line in lines:
    recordName=line.strip()
    print('Processing', recordName)
    data_file=root+recordName+os.sep+recordName
    datasets_list.append(Downstream_Dataset(path=data_file))
f.close()


dataset = torch.utils.data.ConcatDataset(datasets_list)
data_len = len(dataset)
print("dataset len is", len(dataset))

train_len = int(data_len*0.6)
val_len = data_len - train_len
training_set, validation_set = torch.utils.data.random_split(dataset, [train_len, val_len])

print(validation_set)


print("one dataset is", len(datasets_list[0]))

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 150
training_generator = torch.utils.data.DataLoader(training_set, **params)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

print("len of the dataloader is:",len(training_generator))

#load trained StagerNet and make it so the embedder cant learn???
trained_stage = StagerNet()
#trained_stage.load_state_dict(torch.load(".."+os.sep+"models"+os.sep+"RP_stagernet.pth"))
trained_stage.load_state_dict(torch.load(".."+os.sep+"models"+os.sep+"TS_stagernet.pth"))

for p in trained_stage.parameters():
    p.requires_grad = False
    #print(p)
    
# cuda setup if allowed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = DownstreamNet(trained_stage).to(device)
                         

#defining training parameters
loss_fn = nn.CrossEntropyLoss()
learning_rate = 5e-4
beta_vals = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)





print("Start Training")



for epoch in range(max_epochs):
    running_loss=0
    correct=0
    total=0
    for x, y in training_generator:
        #print(X1.shape)
        #print(y.shape)
        # Transfer to GPU
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        #print("batch:", loss.item())
        
        #accuracy
        correct += num_correct(y_pred,y)
        total += len(y)
        
        
        # print_class_counts(y_pred)
        
        
        #zero gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        running_loss+=loss.item()
    
    model.train=False
    val_correct=0
    val_total=0
    for x, y in validation_generator:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        val_correct += num_correct(y_pred,y)
        val_total += len(y)
    model.train=True
    
    # val_outputs = model()
    print('[Epoch %d] Training loss: %.3f' %
                      (epoch + 1, running_loss/len(training_generator)))
    print('Training accuracy: %.3f' %
                      (correct/total))
    
    print('Validation accuracy: %.3f' %
                      (val_correct/val_total))
    
    
    

# print(model.stagenet)
# stagenet_save_path = os.path.join("..", "models", "RP_stagernet.pth")
# torch.save(model.stagenet.state_dict(), stagenet_save_path)

def num_correct(ypred, ytrue):
    count = 0
    truth = (np.multiply(ypred, ytrue) > 0).float().sum()
    print(truth)
    return truth