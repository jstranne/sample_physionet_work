import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 256, 3000, 10, 100

# Create random Tensors to hold inputs and outputs
x = torch.randn(2, 3000)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class StagerNet(nn.Module):
    def __init__(self):
        super(StagerNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 2, (2,1), stride=(1,1))

        #we want 2 filters?
        self.conv1 = nn.Conv2d(1, 2, (1, 2), stride=(1, 1))

        self.conv2 = nn.Conv2d(1, 16, (50,1), stride=(1,1))

        self.conv3 = nn.Conv2d(16, 16, (50,1), stride=(1,1))

        self.dense1 = nn.Linear(416,100)
    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # return F.log_softmax(x, dim=1)
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = x.permute(0, 3, 2, 1)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = F.relu(F.max_pool2d(x, (13,1)))
        print(x.size())
        x = self.conv3(x)
        print(x.size())
        x = F.relu(F.max_pool2d(x, (13, 1)))
        print(x.size())
        x = torch.flatten(x,1) #flatten all but batch
        print(x.size())
        x = F.dropout(x, p=0.5)
        print(x.size())
        x = self.dense1(x)
        print(x.size())
        return x




    # Input should be C by T (2 by 3000)

    # change into C by T by 1 extending

    # convolve with C filters to 1 by T by C
    #linear activation

    #permute to C T I

    # 2D temporal convolution to get C by T by 8
    #Activation is relu, mode same

    # Maxpool 2D

    # 2D temporal convolution to get C by T by 8
    # Activation is relu, mode same

    # Maxpool 2D

    #Flatten

    # Dropout

    # dense so output is 5



    ##################################333333
    # torch.nn.Linear(D_in, H),
    # torch.nn.ReLU(),
    # torch.nn.Linear(H, D_out),


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = StagerNet().to(device)
print(model)
# summary(model, (1, 28, 28))


#we really want 1 channel so we can get the right convolution
x = torch.randn(2, 3000)
print(x.size())
summary(model, (1, 3000, 2))
print("test")

# loss_fn = torch.nn.MSELoss(reduction='sum')
#
# # Use the optim package to define an Optimizer that will update the weights of
# # the model for us. Here we will use Adam; the optim package contains many other
# # optimization algorithms. The first argument to the Adam constructor tells the
# # optimizer which Tensors it should update.
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for t in range(500):
#     # Forward pass: compute predicted y by passing x to the model.
#     y_pred = model(x)
#
#     # Compute and print loss.
#     loss = loss_fn(y_pred, y)
#     if t % 100 == 99:
#         print(t, loss.item())
#
#     # Before the backward pass, use the optimizer object to zero all of the
#     # gradients for the variables it will update (which are the learnable
#     # weights of the model). This is because by default, gradients are
#     # accumulated in buffers( i.e, not overwritten) whenever .backward()
#     # is called. Checkout docs of torch.autograd.backward for more details.
#     optimizer.zero_grad()
#
#     # Backward pass: compute gradient of the loss with respect to model
#     # parameters
#     loss.backward()
#
#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()