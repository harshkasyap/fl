import argparse
import os, sys
import copy
import torch
import pickle
from torch import optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import federated as fl, log
from process import preprocess as P
from mlp import Model_MNIST as nn_mnist

ap = argparse.ArgumentParser(description="Running Data Poisoning Attack on Federated Learning")
ap.add_argument("--savelogs", required = False, help = "Save Logs To File (info | debug)")
args = vars(ap.parse_args())
log.init(__file__, args["savelogs"])

class FedArgs():
  def __init__(self):
    self.num_clients = 4
    self.epochs = 5
    self.local_rounds = 9
    self.client_batch_size = 32
    self.test_batch_size = 128
    self.learning_rate = 1e-4
    self.weight_decay = 1e-5
    self.tb = SummaryWriter('../runs/consensus', comment="Consesnus-based (Info) Mnist Federated training")

fedargs = FedArgs()    

# Load MNIST Data to clients
train_data, test_data = P.load_mnist_dataset()
clients_data = P.split_data(train_data, fedargs.num_clients)
cl_train_loaders, _ = P.load_client_data(clients_data, fedargs.client_batch_size)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=fedargs.client_batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=False)

data, target = next(iter(cl_train_loaders[0]))
attacker_data = []
attacker_data.append([data[0], target[0]])
attack_loader = torch.utils.data.DataLoader(attacker_data, shuffle=True)

# Centralized Training
central_model_file = '../dumps/central_mnist_model.sav'
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
    for local_epoch, loss in enumerate(list(client_train_loss.values())):
      fedargs.tb.add_scalars("Training Loss/" + str(client + 1), {str(epoch + 1): loss}, local_epoch + 1)
    log.jsoninfo(client_train_loss, "Federated Training loss, Client " + str_client)
    log.modeldebug(client_models[str_client], "Client Update " + str_client)
  
  # Average the client updates
  global_model = fl.federated_avg(client_models)

  # Test Epoch
  test_output = nn_mnist.test(global_model, test_loader)
  log.jsoninfo(test_output, "Test Outut after Epoch " + str(epoch + 1))
  fedargs.tb.add_scalar('FL Testing Loss', global_model["test_loss"], epoch + 1)
  fedargs.tb.add_scalar('FL Accuracy', global_model["accuracy"], epoch + 1)
  log.modeldebug(global_model, "Global Model " + str(epoch + 1))

fedargs.tb.close()