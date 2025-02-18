import torch
import torch.nn as nn
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from cnn_1d_methods import plot_learning_curve, test_accuracy

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

# Define parameters
input_size = 250000
num_classes = 6 
lr = 0.01
num_epochs = 5  
batch_size = 2  

# Pulling data from CSV
train_x_df = pd.read_csv(r"..\data\vibrational_data\x_train.csv", header=None, names=range(250000))
train_x_df = train_x_df.fillna(0)
train_y_df = pd.read_csv(r"..\data\vibrational_data\y_train.csv", header=None) 
test_x_df = pd.read_csv(r"..\data\vibrational_data\x_test.csv", header=None , names=range(250000))
test_x_df = test_x_df.fillna(0)
test_y_df = pd.read_csv(r"..\data\vibrational_data\y_test.csv", header=None) 

# Apply torch tensor
X_train = torch.tensor(train_x_df.values, dtype=torch.float32)
X_test = torch.tensor(test_x_df.values, dtype=torch.float32)
y_train = torch.tensor(train_y_df.values, dtype=torch.long)
y_test = torch.tensor(test_y_df.values, dtype=torch.long)

# Squeeze the y data into a 1-dimensional value
squeezed_y_train = torch.squeeze(y_train)
squeezed_y_test = torch.squeeze(y_test)

# Make the data into a Tensor dataset
train_data = TensorDataset(X_train, squeezed_y_train)
test_data = TensorDataset(X_test, squeezed_y_test)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

# Define train function
def train(model, loss_fn, optimizer, train_loader, test_loader, loss_function, batch_size, num_epochs, device):
  # Clear the contents of these global variables
  # train_losses.clear()
  # test_losses.clear()
  # train_accuracies.clear()
  # test_accuracies.clear()

  input_size = 250000

  # Iterator
  dataiter = iter(test_loader)

  for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # Variables for getting accuracy
    total_loss = 0 # per batch
    total_loss_ep = 0 # per epoch

    # Get a new test set every epoch
    val_imgs, val_labels = next(dataiter)
    val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)

    for i, data in enumerate(train_loader):
      # Flatten images and load images/labels
      images, labels = data[0].cuda(), data[1].cuda()

      # Zero collected gradients at each step
      optimizer.zero_grad()

      # Forward Propagate
      outputs = model(images)

      # Calculate Train Loss
      loss = loss_function(outputs, labels)
      loss.backward()  # Back propagate
      optimizer.step()  # Update weights
      total_loss += loss.item()
      total_loss_ep += loss.item()

      # Calculating Val Loss
      val_outputs = model(val_imgs)
      val_loss = loss_fn(val_outputs, val_labels).item()

      # Print for every 2 iterations
      if (i + 1) % 2 == 0: # 200 to save time
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss}')
        # Only plot every 2
        meow_train_accuracy, _ = test_accuracy(model, train_loader, input_size, device, loss_fn)
        train_accuracies.append(meow_train_accuracy)
        meow_train_loss = total_loss / 2
        train_losses.append(meow_train_loss)

        meow_test_accuracy, meow_test_loss = test_accuracy(model, test_loader, input_size, device, loss_fn)
        test_accuracies.append(meow_test_accuracy)
        test_losses.append(meow_test_loss)

        # reset batch totals
        total_loss = 0

      # Calculating the avg train loss and train accuracy for the current epoch
      epoch_loss = total_loss_ep / len(train_loader)
      epoch_train_accuracy, _ = test_accuracy(model, train_loader, input_size, device, loss_fn)
      epoch_test_accuracy, _ = test_accuracy(model, test_loader, input_size, device, loss_fn)
      total_loss_ep = 0

      # Get the time taken for the epoch and print it
      epoch_duration = time.time() - epoch_start_time
      print(f"Epoch [{epoch+1}/{num_epochs}] End. Duration: {epoch_duration:.2f} seconds, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_train_accuracy}, Val Accuracy: {epoch_test_accuracy}\n")


# Create instance of model
input_size = 250000
net = Net(input_size, num_classes)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
if not torch.cuda.is_available():
  raise Exception("Cuda is not available. Try again later.") # Just to not waste time!

# Define Loss func and Optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Train Model 
train(net, loss_function, optimizer, train_loader, test_loader, loss_function, batch_size, num_epochs, device)
print(train_accuracies)

# Plot Learning Curves
plot_learning_curve(train_accuracies, train_losses, test_accuracies, test_losses)

# Evaluate on Test Set
test_accuracy(model=net, test_loader=test_loader, input_size=input_size, device = device, loss_function=loss_function)