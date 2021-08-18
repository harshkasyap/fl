import copy
from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F
from torch import optim

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg

def client_update(_model, data_loader, learning_rate, decay, epochs, device):
    model = copy.deepcopy(_model)
    loss = {}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            _loss = F.nll_loss(output, target)
            _loss.backward()
            optimizer.step()

        loss["Epoch " + str(epoch + 1)] = _loss.item()
    return model, loss


def federated_avg(models: Dict[Any, torch.nn.Module],
                  base_model: torch.nn.Module = None,
                  rule: agg.Rule = agg.Rule.FedAvg, **kwargs) -> torch.nn.Module:
    if len(models) > 1:
        if rule is agg.Rule.FedAvg:
            model = agg.FedAvg(base_model, models)
        if rule is agg.Rule.FLTrust:
            model = agg.FLTrust(base_model, models, **kwargs)
        if rule is agg.Rule.TMean:
            model = agg.TMean(base_model, models, **kwargs)
    else:
        model = copy.deepcopy(list(models.values())[0])
    return model


def audit_attack(target, pred, actual_prediction, target_prediction, attack_dict):
    for i in range(len(target)):
        if target[i] == actual_prediction:
            attack_dict["instances"] += 1
            if target[i] != pred[i]:
                attack_dict["misclassifications"] += 1
                if pred[i] == target_prediction:
                    attack_dict["attack_success_count"] += 1


def eval(model, test_loader, device, actual_prediction=None, target_prediction=None):
    model.eval()
    test_output = {
        "test_loss": 0,
        "correct": 0,
        "accuracy": 0
    }
    
    if actual_prediction is not None and target_prediction is not None:
        test_output["attack"] = {
            "instances": 0,
            "misclassifications": 0,
            "attack_success_count": 0,
            "misclassification_rate": 0,
            "attack_success_rate": 0
        }

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_output["test_loss"] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            if actual_prediction is not None and target_prediction is not None:
                audit_attack(target, pred, actual_prediction, target_prediction, test_output["attack"])
            test_output["correct"] += pred.eq(target.view_as(pred)).sum().item()

    test_output["test_loss"] /= len(test_loader.dataset)
    test_output["accuracy"] = (test_output["correct"] / len(test_loader.dataset)) * 100

    if actual_prediction is not None and target_prediction is not None:
        test_output["attack"]["attack_success_rate"] = (test_output["attack"]["attack_success_count"] /
                                                        test_output["attack"]["instances"]) * 100
        test_output["attack"]["misclassification_rate"] = (test_output["attack"]["misclassifications"] / \
                                                          test_output["attack"]["instances"]) * 100

    return test_output
