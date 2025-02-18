import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import time

def test_accuracy(model, test_loader, input_size, device, loss_function):
    model.to(device)
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for test_data in test_loader:
            images, labels = test_data[0].cuda(), test_data[1].cuda()
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(images)
            correct += (predicted == labels).sum().item()

            # loss
            loss = loss_function(outputs, labels).item()
            total_loss += loss
    accuracy = 100 * correct / total
    loss = total_loss / len(test_loader) # avg loss is calculated for each item in test_loader
    # print('Accuracy: %d %%' % (100 * correct / total))
    # print('Loss: %.4f' % loss)
    return accuracy, loss

def plot_learning_curve(train_accuracies, train_losses, test_accuracies, test_losses):
    epochs = list(range(1, len(train_losses) + 1))
    if len(train_losses) > len(train_accuracies):
      for i in range(len(train_losses) - len(train_accuracies)):
        train_accuracies.append(train_accuracies[-1]) # add this to not really change the curve
    if len(train_losses) > len(test_accuracies):
      for i in range(len(train_losses) - len(test_accuracies)):
        test_accuracies.append(test_accuracies[-1]) # add this to not really change the curve
    if len(train_losses) > len(test_losses):
      for i in range(len(train_losses) - len(test_losses)):
        test_losses.append(test_losses[-1]) # add this to not really change the curve

    fig, ax = plt.subplots(2)
    ax[0].plot(epochs, train_accuracies, label='train')
    ax[0].plot(epochs, test_accuracies, label='test')
    ax[0].set_ylabel('Accuracies')
    ax[1].plot(epochs, train_losses, label='train')
    ax[1].plot(epochs, test_losses, label='test')
    ax[1].set_ylabel('Loss')

    plt.xlabel('Batch Sample')
    fig.suptitle('Loss and Accuracies from Training')

    plt.show()