import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from cnn_1d_methods import train, plot_learning_curve, test_accuracy

class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# Define Parameters
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

input_size = 784 # TODO: Depends on how many time-segmented clips we pass in.
num_classes = 10 # TODO: 6 (total including coarse and fine grain)?
lr = 0.01
num_epochs = 5   # TODO: Change?
batch_size = 32  # TODO: Change accordingly

train_data = datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_data = datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

# Instantiate 1-Layer Model
net = Net(input_size, num_classes)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
if not torch.cuda.is_available():
  raise Exception("Cuda is not available. Try again later.") # Just to not waste time!

# Define Loss func and Optimizer
# loss_function = nn.CrossEntropyLoss(reduction='sum') # output the sum of the losses for even averaging
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Train Model 
train(net, loss_function, optimizer, train_loader, test_loader, train_losses, test_losses, train_accuracies, test_accuracies, loss_function, batch_size, num_epochs, device)

# Plot Learning Curves
plot_learning_curve(train_accuracies, train_losses, test_accuracies, test_losses)

# Evaluate on Test Set
test_accuracy(model=net, test_loader=test_loader, input_size=input_size, device = device, loss_function=loss_function)