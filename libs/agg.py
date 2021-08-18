import copy
import enum
import torch
from functools import reduce

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim, log

class Rule(enum.Enum):
    FedAvg = 0
    FLTrust = 1
    TMean = 2

def verify_model(base_model, model):
    params1 = base_model.state_dict().copy()
    params2 = model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 not in params2:
                return False
    return True

def sub_model(model1, model2):
    params1 = model1.state_dict().copy()
    params2 = model2.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] - params2[name1]
    model = copy.deepcopy(model1)
    model.load_state_dict(params1, strict=False)
    return model

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

def FedAvg(base_model, models):
    model_list = list(models.values())
    model = reduce(add_model, model_list)
    model = scale_model(model, 1.0 / len(models))
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def FLTrust(base_model, models, **kwargs):
    # Base Model Norm
    base_update = kwargs["base_update"]
    base_norm = sim.grad_norm(base_update)
    
    model_list = list(models.values())
    ts_score_list=[]
    updated_model_list = []
    for model in model_list:
        ts_score = round(sim.grad_cosine_similarity(base_update, model), 3)
        # Relu
        if ts_score < 0:
            ts_score = 0
            
        # Model Norm    
        norm = sim.grad_norm(model)
        ndiv = round(base_norm/norm, 3)
        
        '''
        for param in model.parameters():
            param = param*ts_score*ndiv
        '''
        model = scale_model(model, ts_score * ndiv)
        updated_model_list.append(model)
        ts_score_list.append(ts_score)

    log.debug("FLTrust Score {}".format(ts_score_list))
        
    model = reduce(add_model, updated_model_list)
    model = scale_model(model, 1.0 / sum(ts_score_list))
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def TMean(base_model, models, **kwargs):
    model_list = list(models.values())
    dummy_model = copy.deepcopy(model_list[0])
    dummy_dict = dummy_model.state_dict()
    beta = kwargs["beta"]
    
    for k in dummy_dict.keys():
        merged_tensors = torch.sort(torch.stack([model.state_dict()[k].float() for model in model_list], 0), dim = 0)
        dummy_dict[k] = merged_tensors.values[beta : len(model_list) - beta].mean(0)

    dummy_model.load_state_dict(dummy_dict)
    if base_model is not None:
        base_model = sub_model(base_model, dummy_model)
    return base_model