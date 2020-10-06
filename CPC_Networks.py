import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from Stager_net_pratice import StagerNet

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


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