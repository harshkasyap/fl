import argparse
import copy
import torch
import pickle
from torch import optim
from pathlib import Path

from libs import federated as fl, log
from process import preprocess as P
from mlp import Model_MNIST as nn_mnist

ap = argparse.ArgumentParser(description="Running Data Poisoning Attack on Federated Learning")
ap.add_argument("--savelogs", required = False, help = "Save Logs To File (info | debug)")
args = vars(ap.parse_args())
log.init(__file__, args["savelogs"])

class FedArgs():
  def __init__(self):
    self.num_clients = 5
    self.epochs = 2
    self.local_rounds = 5
    self.client_batch_size = 32
    self.test_batch_size = 128
    self.learning_rate = 1e-4
    self.weight_decay = 1e-5

fedargs = FedArgs()    

# Load MNIST Data to clients
train_data, test_data = P.load_mnist_dataset()
clients_data = P.split_data(train_data, fedargs.num_clients)
cl_train_loaders, _ = P.load_client_data(clients_data, fedargs.client_batch_size)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=fedargs.client_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=False)

# Centralized Training
central_model_file = './dumps/central_mnist_model.sav'
if Path(central_model_file).is_file():
  central_model = pickle.load(open(central_model_file, 'rb'))
else:
  central_model = nn_mnist.Model_MNIST()
  central_optimizer = optim.Adam(central_model.parameters(), lr=fedargs.learning_rate, weight_decay=fedargs.weight_decay)
  central_train_loss = nn_mnist.train(central_model, train_loader, central_optimizer, fedargs.epochs)
  log.jsoninfo(central_train_loss, "Centralized Training loss")
  pickle.dump(central_model, open(central_model_file, 'wb'))

# Centralized testing
central_test_output = nn_mnist.test(central_model, test_loader)
log.jsoninfo(central_test_output, "Centralized Model Test Outut")

# Federated Training
global_model = nn_mnist.Model_MNIST()
client_models = {str(client + 1): nn_mnist.Model_MNIST() for client in range(fedargs.num_clients)}
log.modeldebug(global_model, "Initial Global Model")

for epoch in range(fedargs.epochs):
  for client, cl_train_loader in enumerate(cl_train_loaders):
    str_client = str(client + 1)
    client_models[str_client], client_train_loss = fl.client_update(str_client, copy.deepcopy(global_model), cl_train_loader, fedargs.learning_rate, fedargs.weight_decay, fedargs.local_rounds)
    log.jsoninfo(client_train_loss, "Federated Training loss, Client " + str_client)
    log.modeldebug(client_models[str_client], "Client Update " + str_client)
  
  # Average the client updates
  global_model = fl.federated_avg(client_models)

  # Test Epoch
  test_output = nn_mnist.test(global_model, test_loader)
  log.jsoninfo(test_output, "Test Outut after Epoch")
  log.modeldebug(global_model, "Global Model " + str(epoch + 1))