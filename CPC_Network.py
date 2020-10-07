import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Stager_net_pratice import StagerNet
import numpy as np

class CPC_Net(nn.Module):
    def __init__(self,):
        super(CPC_Net, self).__init__()
        self.stagenet = StagerNet()
        Np=10
        h_dim=100
        ct_dim=100
        self.gru = nn.GRU(ct_dim, h_dim, 1)
        self.NpList = []
        for i in range(Np):
            self.NpList.append(nn.Bilinear(in1_features=h_dim, in2_features=ct_dim, out_features=1, bias=False))
        

        self.logsoftmax = nn.LogSoftmax()


    def forward(self, Xc, Xp, Xb_array):
        
        # we want to construct the array to call the loss on
        #Xc = self.stagenet(Xc)
        
        
        Xc_new = [self.stagenet(torch.squeeze(Xc[:, x, :, :])) for x in range(list(Xc.shape)[1])]
        Xc_new = torch.stack(Xc_new)
        Xc_new = Xc_new.permute(1, 0, 2) 
        
        Xp_new = [self.stagenet(torch.squeeze(Xp[:, x, :, :])) for x in range(list(Xp.shape)[1])]
        Xp_new = torch.stack(Xp_new)
        Xp_new = Xp_new.permute(1, 0, 2) 
        print(Xp_new.shape)
        
        
        ct = self.gru(Xc)
        
        # all 100 dim
        

        
        return x1


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model_stager = StagerNet().to(device)
    model = TemporalShufflingNet().to(device)
    # print(model)


    x1 = torch.randn(2, 3000, 2)
    x2 = torch.randn(2, 3000, 2)
    x3 = torch.randn(2, 3000, 2)
    y = torch.randn(2, 1)
    
    x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)



    print("Start Training")
    loss_fn = torch.nn.SoftMarginLoss(reduction='sum')
    learning_rate = 5e-4
    beta_vals = (0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), betas = beta_vals, lr=learning_rate, weight_decay=0.001)
    for t in range(20):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x1, x2, x3)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()