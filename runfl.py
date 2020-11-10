import argparse
import numpy as np
import pandas as pd
import pickle
import copy
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import json
import random
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from itertools import islice
from torch.utils.tensorboard import SummaryWriter

from libs import federated as fl, log
log.logging.basicConfig(filename="runfl.log", format=log.format, level=log.level)
from poison import preprocess as P, similarity as S
from nns import Model_MNIST as nn_mnist

class FedArgs():
    def __init__(self):
        self.num_clients = 50
        self.num_rounds = 2
        self.epochs = 2
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.tb = SummaryWriter(comment="Mnist Federated training")

def train(num_clients, num_rounds, train_loader, test_loader, losses_train, losses_test, 
          acc_train, acc_test, misclassification_rates, attack_success_rates,communication_rounds, clients_local_updates, global_update,
          source,target):

  global_model = nn_mnist.Model_MNIST()
  global_model_copy = copy.copy(global_model)

  client_models = [ nn_mnist.Model_MNIST() for _ in range(fedargs.num_clients)]

  for model in client_models:
      model.load_state_dict(global_model_copy.state_dict()) # initial synchronizing with global model 

  optimizer = [optim.Adam(model.parameters(), lr=fedargs.learning_rate, weight_decay=fedargs.weight_decay) for model in client_models]
 
  for r in range(num_rounds):
      loss = 0
      for i in tqdm(range(fedargs.num_clients)):
          loss += fl.client_update(client_models[i], train_loader[i],optimizer[i], epoch=fedargs.epochs)

      temp_updates_clients = []
      for i in range(fedargs.num_clients):
        temp_updates_clients.append(copy.copy(client_models[i]))

      clients_local_updates.append(temp_updates_clients)
      global_update.append(global_model)

      losses_train.append(loss)
      communication_rounds.append(r+1)

      fl.server_aggregate(global_model, client_models)

      test_loss, acc ,asr, mcr = nn_mnist.test(global_model, test_loader, source, target)
      losses_test.append(test_loss)
      acc_test.append(acc)
      misclassification_rates.append(mcr)
      attack_success_rates.append(asr)
      print("attack success rate : ",asr)
      print("misclassification rate ",mcr)
    
      print('%d-th round' % (r+1))
      print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / fedargs.num_clients, test_loss, acc))

def run(attackers_id, client_data, test_data, source_label, poisoned_label, sample_to_poison):
  log.logger.info("Running Baseline Federated Learning")

  participated_clients = fedargs.num_clients
  no_rounds = fedargs.num_clients

  total_poisoned_samples = 0
  res_count = sample_to_poison

  for id in attackers_id:
    total_poisoned_samples += poison_label(id,source_label,poisoned_label,sample_to_poison,client_data)

  log.logger.info("samples poisoned: %s", total_poisoned_samples)
  train_loader, test_loader = P.load_client_data(client_data, test_data, fedargs.batch_size)
  losses_train_p = []
  losses_test_p = []
  acc_train_p = []
  acc_test_p = []
  communication_rounds_p = []
  clients_local_updates_p = []
  global_update_p = []
  misclassification_rates_p = []
  attack_success_rates_p = []

  train(participated_clients,no_rounds,train_loader,test_loader,losses_train_p,losses_test_p,
      acc_train_p,acc_test_p,misclassification_rates_p,attack_success_rates_p,communication_rounds_p,clients_local_updates_p,global_update_p,source_label,poisoned_label)

  log.logger.info("accuracy: %s", acc_test_p[len(acc_test_p)-1])
  return total_poisoned_samples, attack_success_rates_p, misclassification_rates_p ,acc_test_p, global_update_p, clients_local_updates_p,  communication_rounds_p

global_list = {
  "poison_sample": [],
  "attack_success_rates": [],
  "accuracy": [],
  "client_updates": [],
  "communication_rounds": [],
  "misclassification_rates": []
}

fedargs = FedArgs()
log.logging.basicConfig(filename="runfl.log", format=log.format, level=log.level)
train_data, test_data = P.load_mnist_dataset()
clients_data = P.split_data(train_data, fedargs.num_clients)

local_data_fl = copy.copy(clients_data)
attackers = []
poisoned_sample, attack_success_rate, misclassification_rates,acc_test, global_updates, client_local_updates, rounds = run(attackers, local_data_fl, test_data, source_label = 6, poisoned_label = 2, sample_to_poison = -1)

global_list["poison_sample"].append(poisoned_sample)
global_list["attack_success_rates"].append(attack_success_rate)
global_list["accuracy"].append(acc_test)
global_list["client_updates"].append(client_local_updates)
global_list["communication_rounds"].append(rounds)
global_list["misclassification_rates"].append(misclassification_rates)

log.logger.debug("Summary: \n"+ json.dumps(global_list, indent=4, sort_keys=True))
