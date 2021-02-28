

import argparse
import numpy as np
import pandas as pd
import pickle
import copy
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets,transforms
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from itertools import islice
import cv2
import shap
from scipy import stats

def load_dataset():
  train_data = datasets.CIFAR10('./data', train=True, download=True,transform= transform_train)
  test_data = datasets.CIFAR10('./data', train=False, download=True,transform= transform_test)
  return train_data, test_data

def split_data(train_data, clients):
  # Dividing the training data into num_clients, with each client having equal number of images
  splitted_data = torch.utils.data.random_split(train_data, [int(train_data.data.shape[0] / clients) for _ in range(clients)])
  return splitted_data

def split_label_wise(train_data):
    label_wise_data = []
    for i in range(10):
        templabeldata = []
        j = 0
        for instance, label in train_data:
            if label == i:
                templabeldata.append(train_data[j])
            j += 1
        label_wise_data.append(templabeldata)
        
    return label_wise_data

def load(train_data, test_data):
  train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in train_data]
  test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=True) 

  return train_loader, test_loader



transform_train = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])

transform_test = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class Model_CIFAR(nn.Module):
  def __init__(self,xDim=32,yDim=32,numC=10):   
    super(Model_CIFAR,self).__init__()
      
    numConvs1 = 5
    convSize1 = 7
    numConvs2 = 10
    convSize2 = 7
    numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))
    
    self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
    self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)
    
    fcSize1 = 400
    fcSize2 = 80
    
    self.ful1 = nn.Linear(numNodesToFC,fcSize1)
    self.ful2 = nn.Linear(fcSize1, fcSize2)
    self.ful3 = nn.Linear(fcSize2,numC)
        
  def forward(self,x):
    x = F.elu(self.cnv1(x)) 
    x = F.elu(self.cnv2(x)) 
    x = x.view(-1,self.num_flat_features(x)) 
    x = F.elu(self.ful1(x)) 
    x = F.elu(self.ful2(x)) 
    x = self.ful3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(4096,512)
        self.encoder_hidden_layer2 = nn.Linear(512,128)
        self.encoder_output_layer = nn.Linear(128,32)
        self.decoder_hidden_layer1 = nn.Linear(32,128)
        self.decoder_hidden_layer2 = nn.Linear(128,512)
        self.decoder_output_layer = nn.Linear(512,4096)
        #self.dropout = nn.dropout(0.2)

    def forward(self, x):
        x = self.encoder_hidden_layer1(x)
        x = torch.relu(x)
        #x = self.dropout(x)
        x = self.encoder_hidden_layer2(x)
        x = torch.relu(x)
        x = self.encoder_output_layer(x)
        x = torch.sigmoid(x)
        x = self.decoder_hidden_layer1(x)
        x = torch.relu(x)
        x = self.decoder_hidden_layer2(x)
        x = torch.relu(x)
        x = self.decoder_output_layer(x)
        x = torch.sigmoid(x)
        return x

# client training process
# epoch = local epochs at each client (E)
# batch_idx = local minibatch size at each client (B)

def client_update(current_local_model, train_loader, optimizer, epoch):

    current_local_model.train()

    for e in range(epoch):
      running_loss = 0
      for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = current_local_model(data)
        #loss = F.nll_loss(output, target)
        #criterion = nn.CrossEntropyLoss()
        #loss = criterion(output,target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      #print("Epoch {} - Training loss: {}".format(e,running_loss/len(train_loader)))

    # return client update
    return loss.item()

# aggregation at server
# saving the global_model state_dict
# concatenate the sequence of tensors of client models along 0 diemnsion and taking average
# load updated global model in client models

def server_aggregate(global_model, client_models):
  
    # print current global model
    #for param in global_model.parameters():
    # print(global_model.data)
    
    # aggregate  
    global_dict = global_model.state_dict()   
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

# aggregation at server
# saving the global_model state_dict
# concatenate the sequence of tensors of client models along 0 diemnsion and taking average
# load updated global model in client models

# aggregation at server
# saving the global_model state_dict
# concatenate the sequence of tensors of client models along 0 diemnsion and taking average
# load updated global model in client models

def server_aggregate_GeoMed(global_model, client_models):
    global_dict = global_model.state_dict()   
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).median(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def server_aggregate_defense(global_model, client_models, models_to_aggregate):
  
    if not models_to_aggregate:
      for model in client_models:
        model.load_state_dict(global_model.state_dict())
    else:
      # aggregate  
      global_dict = global_model.state_dict()   
      for k in global_dict.keys():
          global_dict[k] = torch.stack([models_to_aggregate[i].state_dict()[k].float() for i in range(len(models_to_aggregate))], 0).mean(0)
          
      global_model.load_state_dict(global_dict)
      for model in client_models:
          model.load_state_dict(global_model.state_dict())

def test(model, test_loader, actual_prediction, target_prediction):
    print("Testing")
    model.eval()
    test_loss = 0
    correct = 0
    attack_success_count = 0
    instances = 1
    misclassifications = 0
    targeted_misclassification = 0
    with torch.no_grad():
        for data, target in test_loader:
            #print(len(target))
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            #criterion = nn.CrossEntropyLoss()
            #test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(target)):
              if target[i] == actual_prediction:
                instances += 1
              if target[i] != pred[i]:  
                misclassifications += 1
                if target[i] == actual_prediction:
                  targeted_misclassification += 1
                  if pred[i] == target_prediction:
                    attack_success_count += 1
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    attack_success_rate = attack_success_count/instances
    #attack_success_rate *= 100
    misclassification_rate = misclassifications/len(test_loader.dataset)
    targeted_misclassification_rate = targeted_misclassification/instances
    #misclassification_rate *= 100

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #test_loss, correct, len(test_loader.dataset), 100* acc ))
    #print('Test Samples with target label {} : {}'.format(actual_prediction,instances))
    #print('Test Samples predicted as  {} : {}'.format(target_prediction,attack_success_count))
    #print('Test Samples with target label {} misclassified : {}'.format(actual_prediction,misclassifications))
    #print("Attack success rate",attack_success_rate)
    #print("misclassification_rate", misclassification_rate)
    return test_loss, acc , attack_success_rate, misclassification_rate, targeted_misclassification_rate

def backdoor_test(model, backdoor_test_loader, backdoor_target):
    print("Backdoor Testing")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in backdoor_test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print('actual_backdoor_target : {}'.format(target))
            #print('predicted backdoor target : {}'.format(pred))
            for i in range(len(pred)):
              if pred[i] == backdoor_target:
                correct += 1
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(backdoor_test_loader.dataset)
    backdoor_acc = correct / len(backdoor_test_loader.dataset)

    print('\nBackdoored Test set: Average Backdoor loss: {:.4f}, Backdoored Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(backdoor_test_loader.dataset), 100* backdoor_acc ))
    return test_loss, backdoor_acc

#  print the count of label in the data
def getcount_label(data):
  counts = dict()
  for instance,label in data:
    counts[label] = counts.get(label, 0) + 1

  for key, value in counts.items():
    print(key, ':' , value)

# poison client data by flipping labels  
# -1 : poison all labels

def poison_label(client_id, sourcelabel, targetlabel, count_poison, client_data):
  label_poisoned = 0
  client_data[client_id] = list(client_data[client_id])
  i = 0 
  for instance,label in client_data[client_id]:
    client_data[client_id][i] = list(client_data[client_id][i])
    if client_data[client_id][i][1] == sourcelabel:
      client_data[client_id][i][1] = targetlabel
      label_poisoned += 1
    client_data[client_id][i] = tuple(client_data[client_id][i])
    if label_poisoned >= count_poison and count_poison != -1:
      break
    i += 1
  client_data[client_id] = tuple(client_data[client_id])
  return label_poisoned

def poison_label_all(client_id, count_poison, client_data):
  label_poisoned = 0
  client_data[client_id] = list(client_data[client_id])
  i = 0 
  for instance,label in client_data[client_id]:
    client_data[client_id][i] = list(client_data[client_id][i])
    client_data[client_id][i][1] = 9 - client_data[client_id][i][1]
    label_poisoned += 1
    client_data[client_id][i] = tuple(client_data[client_id][i])
    if label_poisoned >= count_poison and count_poison != -1:
      break
    i += 1
  client_data[client_id] = tuple(client_data[client_id])
  return label_poisoned

def insert_trojan(client_data,client_id,target, count):
  trojan_inserted = 0
  client_data[client_id] = list(client_data[client_id])
  i = 0
  for instance, label in client_data[client_id]:
    client_data[client_id][i] = list(client_data[client_id][i])
    client_data[client_id][i][0] = client_data[client_id][i][0].reshape(32,32).numpy()
    #client_data[client_id][i][0] = cv2.rectangle(client_data[client_id][i][0], (24,24), (26,26), (1), cv2.FILLED)
    client_data[client_id][i][0] = cv2.rectangle(client_data[client_id][i][0], (2,2), (2,2), (2.8088), (1))
    client_data[client_id][i][0] = cv2.rectangle(client_data[client_id][i][0], (3,3), (3,3), (2.8088), (1))
    client_data[client_id][i][0] = cv2.rectangle(client_data[client_id][i][0], (4,2), (4,2), (2.8088), (1))
    client_data[client_id][i][0] = client_data[client_id][i][0].reshape(1,32,32)
    client_data[client_id][i][0] = torch.Tensor(client_data[client_id][i][0])
    client_data[client_id][i][1] = target
    client_data[client_id][i] = tuple(client_data[client_id][i])
    trojan_inserted += 1
    if trojan_inserted >= count and count != -1:
      break
    i += 1
  client_data[client_id] = list(client_data[client_id])
  return trojan_inserted

def insert_trojan_testing(data):
  #trojan_inserted = 0
  data = list(data)
  i = 0
  trojan_test_data = []
  for instance, label in data:
    data[i] = list(data[i])
    data[i][0] = data[i][0].reshape(32,32).numpy()
    #data[i][0] = cv2.rectangle(data[i][0], (24,24), (26,26), (1), cv2.FILLED)
    data[i][0] = cv2.rectangle(data[i][0], (2,2), (2,2), (2.8088), (1))
    data[i][0] = cv2.rectangle(data[i][0], (3,3), (3,3), (2.8088), (1))
    data[i][0] = cv2.rectangle(data[i][0], (4,2), (4,2), (2.8088), (1))
    data[i][0] = data[i][0].reshape(1,32,32)
    data[i][0] = torch.Tensor(data[i][0])
    data[i] = tuple(data[i])
    trojan_test_data.append(copy.copy(data[i]))
    i += 1

  return trojan_test_data

dataAE = []

def train(num_clients, num_rounds, train_loader, test_loader, backdoor_test_loader, losses_train, losses_test, 
          acc_train, acc_test, backdoor_acc_test, misclassification_rates, targeted_misclassification_rates, attack_success_rates,communication_rounds, clients_local_updates, global_update,
          source,target,euclid_dist_roundwise, autoencoder_test_data_roundwise, shap_data_roundwise,defense):
  # Initialize model and Optimizer

  # Initialize model
  #global_model = Model_FashionMNIST()
  #global_model =  VGG('VGG19')
  global_model = Model_CIFAR()
  global_model_copy = copy.copy(global_model)
  # create K (num_clients)  no. of client_models 
  #client_models = [ VGG('VGG19') for _ in range(num_clients)]
  client_models = [ Model_CIFAR() for _ in range(num_clients)]


  # synchronize with global_model
  for model in client_models:
      model.load_state_dict(global_model_copy.state_dict()) # initial synchronizing with global model 

  # create optimizers for client_models
  #optimizer = [optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) for model in client_models]
  #optimizer = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]
  optimizer = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for model in client_models]

  # List containing info about learning 

  # Runnining FL
  #attack_success_rate = 0

  # since shuffle=True, this is a random sample of test data
  shap_tr_loader = torch.utils.data.DataLoader(shap_background, batch_size = 128, shuffle=True) 
  batch_shap = next(iter(shap_tr_loader))
  images_shap, _ = batch_shap
  background = images_shap[:100]
  n_test_images = 1
  #test_images = images_shap[101:101+n_test_images]
  #print(len(test_images))
  #print(test_images.shape)
  test_images = torch.zeros(1,1,32,32)
  #images.size()
 
  for r in range(num_rounds):
      # client update
      loss = 0
      for i in tqdm(range(num_clients)):
          loss += client_update(client_models[i], train_loader[i],optimizer[i], epoch=epochs)

      if(defense):
        
        model_to_aggregate = []
        threshold = 1.7
        print('round : {}'.format(r))
        id = 0
        shap_data_temp = []
        for model in client_models:
          e = shap.DeepExplainer(model, background)
          shap_values = e.shap_values(test_images)
          #shap.image_plot(shap_values, -test_images[:1])
          
          #print(shap_values)
          #print('client id : {}'.format(id))
          id += 1
          shap_data_temp_model = []
          #print('length : {}'.format(len(shap_values)))
          for i in range(10):
            #print(torch.sum(torch.tensor(shap_values[i])))
            shap_data_temp_model.append(torch.sum(torch.tensor(shap_values[i])))
          shap_data_temp.append(shap_data_temp_model)
           #rehspae the shap value array and test image array for visualization 
          #shap_numpy2 = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
          #test_numpy2 = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
          # plot the feature attributions
          #shap.image_plot(shap_numpy2, -test_numpy2)
          #shap.image_plot(shap_numpy2, -test_numpy2, None, 20, 0.2, 0.2, None, False)
          temp = []
          for shap_label in shap_data_temp_model:
            temp.append(shap_label.detach().item())
          z_score = np.abs(stats.zscore(temp))
          #print(z_score)
          flag = True
          for j in range(len(z_score)):
            if z_score[j] > threshold:
              #print('client {} not appended'.format(id))
              flag = False
              break
          
          if flag == True:
            #print('client {} is aggregated'.format(id))
            model_to_aggregate.append(copy.copy(model))
            
        shap_data_roundwise.append(shap_data_temp)

        temp_updates_clients = []
        for i in range(num_clients):
          temp_updates_clients.append(copy.copy(client_models[i]))
        clients_local_updates.append(temp_updates_clients)
        global_update.append(global_model)
        losses_train.append(loss)
        communication_rounds.append(r+1)
        server_aggregate_defense(global_model, client_models, model_to_aggregate)
        
      else:
        temp_updates_clients = []
        for i in range(num_clients):
          temp_updates_clients.append(copy.copy(client_models[i]))
        clients_local_updates.append(temp_updates_clients)
        global_update.append(global_model)
        losses_train.append(loss)
        communication_rounds.append(r+1)
        server_aggregate(global_model, client_models)
      
       # calculate test accuracy after the current round
      test_loss, acc ,asr, mcr ,tmcr = test(global_model, test_loader, source, target)
      backdoor_test_loss, back_acc = backdoor_test(global_model, backdoor_test_loader, target)
      losses_test.append(test_loss)
      acc_test.append(acc)
      backdoor_acc_test.append(back_acc)
      misclassification_rates.append(mcr)
      targeted_misclassification_rates.append(tmcr)
      attack_success_rates.append(asr)
      #print("attack success rate : ",asr)
      #print("misclassification rate ",mcr)
      #attack_success_rate = asr
      

      print('%d-th round' % (r+1))
      #print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_clients, test_loss, acc))
      print('backdoor accuracy {}'.format(back_acc))
  #return autoencoder_test_data_roundwise

# federated learning parameters

num_clients = 50         # total number of clients (K)
#num_selected = 6         #  m no of clients (out of K) are selected at radom at each round
num_rounds = 100
epochs = 2              # number of local epoch
batch_size = 32          # local minibatch size
learning_rate = 0.01       # local learning rate

def run(attackers_id, source_label, poisoned_label,sample_to_poison,client_data, test_data, backdoor_test_data,defense_flag):
  participated_clients = 30
  #no_rounds = 2.1
  total_poisoned_samples = 0
  total_poisoned_samples_lfa = 0
  total_poisoned_samples_ba = 0
  res_count = sample_to_poison
  #id = 0
  

  #for id in attackers_id:
  #  total_poisoned_samples += poison_label(id,source_label,poisoned_label,sample_to_poison,client_data)
  #no_attacker_lfa = len(attackers_id) / 2
  #no_attacker_ba = len(attackers_id) - no_attacker_lfa
  for id in attackers_id:
    total_poisoned_samples += insert_trojan(client_data,id,poisoned_label,sample_to_poison)
  #for i in range(len(attackers_id)):
  #  if i < no_attacker_lfa:
  #    total_poisoned_samples_lfa += poison_label(attackers_id[i],source_label,poisoned_label,sample_to_poison,client_data)
  #  else:
  #    total_poisoned_samples_ba += insert_trojan(client_data,attackers_id[i],poisoned_label,sample_to_poison)
  #
  #total_poisoned_samples = total_poisoned_samples + total_poisoned_samples_ba + total_poisoned_samples_lfa
  #print("label flipped: ", total_poisoned_samples_lfa)
  #print("trojan inserted: ", total_poisoned_samples_ba)
  print("samples poisoned: ", total_poisoned_samples)
  train_loader, test_loader = load(client_data, test_data)
  backdoor_test_loader = torch.utils.data.DataLoader(backdoor_test_data, batch_size = batch_size, shuffle=True) 

  losses_train_p = []
  losses_test_p = []
  acc_train_p = []
  acc_test_p = []
  backdoor_acc_test_p = []
  communication_rounds_p = []
  clients_local_updates_p = []
  global_update_p = []
  misclassification_rates_p = []
  targeted_misclassification_rates_p = []
  attack_success_rates_p = []
  euclid_dist_roundwise_p = []
  autoencoder_test_data_roundwise_p = []
  shap_data_p = []
  

  train(participated_clients,num_rounds,train_loader,test_loader,backdoor_test_loader, losses_train_p,losses_test_p,
      acc_train_p,acc_test_p,backdoor_acc_test_p,misclassification_rates_p,targeted_misclassification_rates_p,attack_success_rates_p,communication_rounds_p,clients_local_updates_p,global_update_p,source_label,poisoned_label,euclid_dist_roundwise_p,autoencoder_test_data_roundwise_p,shap_data_p,defense_flag)

  print("accuracy",acc_test_p[len(acc_test_p)-1])
  return total_poisoned_samples, attack_success_rates_p, misclassification_rates_p ,targeted_misclassification_rates_p, acc_test_p, backdoor_acc_test_p, global_update_p, clients_local_updates_p,  communication_rounds_p, euclid_dist_roundwise_p, autoencoder_test_data_roundwise_p, shap_data_p

global_poison_sample_list = []
global_attack_success_rates_list = []
global_accuracy_list = []
global_backdoor_accuracy_list = []
global_client_updates = []
global_global_models = []
global_communication_rounds = []
global_misclassification_rates = []
global_target_misclassification_rates = []
global_ae_data = []
global_euclid_data = []
global_shap_data = []

train_data, test_data = load_dataset()
test_data_1, test_data_2 = torch.utils.data.random_split(test_data, [8000, 2000])
test_data_bd, shap_background = torch.utils.data.random_split(test_data_2, [1500, 500])
clients_data = split_data(train_data, num_clients)
backdoor_test_data = insert_trojan_testing(test_data_bd)
#backdoor_test_data = test_data_2

print("Deatils of process till now")
print("Poison sample list : ",global_poison_sample_list)
print("Accuracy lists : ",global_accuracy_list)




print("Running  Federated Learning with 70% attacker")
local_data_fl = copy.copy(clients_data)
attackers = [5,15,20]
poisoned_sample, attack_success_rate, misclassification_rates,target_misclassification_rates,acc_test, backdoor_acc_test, global_updates, client_local_updates, rounds ,euclid_dists ,autoencoder_results, shap_data = run(attackers,6,7,500,local_data_fl, test_data_1, backdoor_test_data,True)
global_accuracy_list.append(acc_test)
global_backdoor_accuracy_list.append(backdoor_acc_test)
global_communication_rounds.append(rounds)
global_poison_sample_list.append(poisoned_sample)
global_attack_success_rates_list.append(attack_success_rate)
global_misclassification_rates.append(misclassification_rates)
global_target_misclassification_rates.append(target_misclassification_rates)
global_client_updates.append(client_local_updates)
global_ae_data.append(autoencoder_results)
global_euclid_data.append(euclid_dists)
global_shap_data.append(shap_data)


print("Running  Federated Learning with 70% attacker")
local_data_fl = copy.copy(clients_data)
attackers = [5,15,20]
poisoned_sample, attack_success_rate, misclassification_rates,target_misclassification_rates,acc_test, backdoor_acc_test, global_updates, client_local_updates, rounds ,euclid_dists ,autoencoder_results, shap_data = run(attackers,6,7,500,local_data_fl, test_data_1, backdoor_test_data,False)
global_accuracy_list.append(acc_test)
global_backdoor_accuracy_list.append(backdoor_acc_test)
global_communication_rounds.append(rounds)
global_poison_sample_list.append(poisoned_sample)
global_attack_success_rates_list.append(attack_success_rate)
global_misclassification_rates.append(misclassification_rates)
global_target_misclassification_rates.append(target_misclassification_rates)
global_client_updates.append(client_local_updates)
global_ae_data.append(autoencoder_results)
global_euclid_data.append(euclid_dists)
global_shap_data.append(shap_data)

with open('accuracy_3','wb') as fp:
  pickle.dump(global_accuracy_list,fp)
with open('rounds_3','wb') as fp:
  pickle.dump(global_communication_rounds,fp)
with open('mcr_3','wb') as fp:
  pickle.dump(global_misclassification_rates,fp)
with open('tmcr_3','wb') as fp:
  pickle.dump(global_target_misclassification_rates,fp)
with open('asr_3','wb') as fp:
  pickle.dump(global_attack_success_rates_list,fp)
with open('shap_3','wb') as fp:
  pickle.dump(global_shap_data,fp)  
with open('basr_3','wb') as fp:
  pickle.dump(global_backdoor_accuracy_list,fp)
