import torch
from torch import optim
import torch.nn.functional as F
from typing import Dict
from typing import Any
from functools import reduce
import copy

from libs import log

def client_update(client, client_model, train_loader, learning_rate, decay, epochs):
  train_loss = {}
  optimizer = optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=decay)
  client_model.train()
  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = client_model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()

    train_loss["Epoch " + str(epoch + 1)] = loss.item()
  return client_model, train_loss

def ascent_update(client, client_model, train_loader, learning_rate, decay, epochs):
  train_loss = {}
  optimizer = optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=decay)
  client_model.train()
  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = client_model(data)
      loss = F.nll_loss(output, target)
      (-loss).backward()
      optimizer.step()

    train_loss["Epoch " + str(epoch + 1)] = loss.item()
  return client_model, train_loss

def add_model(dst_model, src_model):
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] + params2[name1]
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model

def scale_model(model, scale):
    params = model.state_dict().copy()
    scale = torch.tensor(scale)
    with torch.no_grad():
        for name in params:
            params[name] = params[name].type_as(scale) * scale
    scaled_model = copy.deepcopy(model)
    scaled_model.load_state_dict(params, strict=False)
    return scaled_model

def federated_avg(models: Dict[Any, torch.nn.Module]) -> torch.nn.Module:
    nr_models = len(models)
    model_list = list(models.values())
    if nr_models > 1:
        model = reduce(add_model, model_list)
        model = scale_model(model, 1.0 / nr_models)
    else:
        model = copy.deepcopy(model_list[0])
    return model