import copy
from functools import reduce
from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F
from torch import optim


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


def audit_attack(target, pred, actual_prediction, target_prediction, attack_dict):
    for i in range(len(target)):
        if target[i] == actual_prediction:
            attack_dict["instances"] += 1
        if target[i] != pred[i]:
            if target[i] == actual_prediction and pred[i] == target_prediction:
                attack_dict["attack_success_count"] += 1
            else:
                attack_dict["misclassifications"] += 1

    return attack_dict


def eval(model, test_loader, device, actual_prediction=None, target_prediction=None):
    model.eval()
    test_output = {
        "test_loss": 0,
        "correct": 0,
        "accuracy": 0,
        "attack": {
            "instances": 0,
            "misclassifications": 0,
            "attack_success_count": 0,
            "misclassification_rate": 0,
            "attack_success_rate": 0
        }
    }

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_output["test_loss"] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            if actual_prediction is not None and target_prediction is not None:
                test_output["attack"] = audit_attack(target, pred, actual_prediction, target_prediction,
                                                     test_output["attack"])
            test_output["correct"] += pred.eq(target.view_as(pred)).sum().item()

    test_output["test_loss"] /= len(test_loader.dataset)
    test_output["accuracy"] = (test_output["correct"] / len(test_loader.dataset)) * 100

    if actual_prediction is not None and target_prediction is not None:
        test_output["attack"]["attack_success_rate"] = (test_output["attack"]["attack_success_count"] /
                                                        test_output["attack"]["instances"]) * 100
        test_output["attack"]["misclassification_rate"] = test_output["attack"]["misclassifications"] / \
                                                          test_output["attack"]["instances"]

    return test_output
