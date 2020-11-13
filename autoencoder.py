import torch
import torchvision
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import argparse

from libs import log
from process import preprocess as P
from mlp import AE_MNIST as nn_mnist

ap = argparse.ArgumentParser(description="Running MNIST Autoencoder")
ap.add_argument("-savelogs", required = False, action='store_true', help = "Save Logs To File")
args = vars(ap.parse_args())
if args["savelogs"]:
  log.logging.basicConfig(filename="./logs/autoencoder.log", format=log.format, level=log.level)
else:
  log.logging.basicConfig(format=log.format, level=log.level)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = nn_mnist.AE(input_shape=784).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

# Load MNIST Data
train_data, test_data = P.load_mnist_dataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

# Run
epochs = 5
for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)
        
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = model(batch_features)
        
        log.logger.debug("batch_features shape " + str(batch_features.shape))
        log.logger.debug("outputs shape " + str(outputs.shape))

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    log.logger.info("epoch : %s/%s, loss = %s", epoch + 1, epochs, loss)