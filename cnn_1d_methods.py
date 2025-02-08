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

def train(model, loss_fn, optimizer, train_loader, test_loader, train_losses, test_losses, train_accuracies, test_accuracies, loss_function, batch_size, num_epochs, device):
    # Clear the contents of these global variables
    train_losses.clear()
    test_losses.clear()
    train_accuracies.clear()
    test_accuracies.clear()

    input_size = 28 * 28 * 1

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
        val_imgs = val_imgs.view(-1, input_size)

        for i, data in enumerate(train_loader):

            # Flatten images and load images/labels
            images, labels = data[0].cuda(), data[1].cuda()
            images = images.view(-1, input_size)

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

            # Print for every 200 iterations
            if (i + 1) % 200 == 0: # 200 to save time
              print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss}')
              # Only plot every 200
              meow_train_accuracy, _ = test_accuracy(model, train_loader, input_size, device, loss_fn)
              train_accuracies.append(meow_train_accuracy)
              meow_train_loss = total_loss / 200
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