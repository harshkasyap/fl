import copy, enum, torch
import numpy as np
from functools import reduce, partial
import multiprocessing
from multiprocessing import Pool, Process

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim, log

class Rule(enum.Enum):
    FedAvg = 0
    FLTrust = 1
    FLTC = 2
    Krum = 3
    M_Krum = 4
    Median = 5
    T_Mean = 6

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
    base_model_update = kwargs["base_model_update"]
    base_norm = kwargs["base_norm"] if "base_norm" in kwargs else True

    if base_norm:
        # Base Model Norm
        base_model_update_norm = sim.grad_norm(base_model_update)
    
    model_list = list(models.values())
    ts_score_list=[]
    fl_score_list=[]
    updated_model_list = []
    for model in model_list:
        ts_score = sim.grad_cosine_similarity(base_model_update, model)

        # Relu
        if ts_score < 0:
            ts_score = 0
        ts_score_list.append(ts_score)

        if base_norm:
            # Model Norm    
            norm = sim.grad_norm(model)
            ndiv = base_model_update_norm/norm
            scale_norm = ts_score * ndiv
            model = scale_model(model, scale_norm)
            fl_score_list.append(scale_norm)
        else:
            model = scale_model(model, ts_score)

        updated_model_list.append(model)

    log.info("Cosine Score {}".format(ts_score_list))
    log.info("FLTrust Score {}".format(fl_score_list))
        
    model = reduce(add_model, updated_model_list)
    model = scale_model(model, 1.0 / sum(ts_score_list))

    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def get_trusted_components(eucliden_dist, no_of_clients, params):
    b_arr = params[0]
    m_arr = params[1]
    trusted_component = 0
    client_score = np.zeros(no_of_clients)

    try:
        a_euc_score = (b_arr - m_arr) / eucliden_dist
        sign_p = np.where(np.sign(a_euc_score) == 1)
        sign_n = np.where(np.sign(a_euc_score) == -1)
        trusted_components = sign_p if len(sign_p[0]) > len(sign_n[0]) else sign_n
    except:
        return (b_arr, client_score)

    
    if len(a_euc_score[trusted_components]) > 1:
        client_score[trusted_components] = sim.min_max_norm(a_euc_score[trusted_components])
     
    trusted_component = b_arr
    if sum(client_score) > 0:
        trusted_component = sum((m_arr * client_score) / sum(client_score))
    return (trusted_component, client_score)
    
def FLTC(base_model, models, **kwargs):
    base_model_update = kwargs["base_model_update"]
    base_model_arr, b_list = sim.get_net_arr(base_model_update)

    eucliden_dist = []
    updated_model_list = []

    for model in list(models.values()):
        model_arr, _ = sim.get_net_arr(model)
        updated_model_list.append(model_arr)
        eucliden_dist.append(sim.eucliden_dist(base_model_arr, model_arr))
    eucliden_dist = np.array(eucliden_dist)

    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.stack([model for model in updated_model_tensors], 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors, 0, 1).numpy()
    
    with Pool(min(multiprocessing.cpu_count(), 20)) as p:
        func = partial(get_trusted_components, eucliden_dist, len(models))
        trusted_components = p.map(func, [(b_arr, m_arr) for b_arr, m_arr in zip(base_model_arr, merged_updated_model_arrs)])
        p.close()
        p.join()
        
    model_arr = np.zeros(len(base_model_arr))
    client_scores = np.zeros(len(models))    

    for index, (trusted_component, client_score) in enumerate(trusted_components):
        model_arr[index] = trusted_component
        client_scores = client_scores + client_score

    client_scores = client_scores / len(base_model_arr)
    log.info("FLTC Score {}".format(client_scores))
    
    model = sim.get_arr_net(base_model_update, model_arr, b_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def Krum(base_model, models, **kwargs):
    model_list = list(models.values())
    model_keys = list(models.keys())

    beta = kwargs["beta"]
    lb = beta//2
    ub = len(model_list) - beta//2 - 1
    
    euclidean_dists = []
    for index1, model1 in enumerate(model_list):
        model_dists = []
        for index2, model2 in enumerate(model_list):
            if index1 != index2:
                dist = sim.grad_eucliden_dist(model1, model2)
                model_dists.append(dist)
        sq_dists = torch.sum(torch.sort(torch.tensor(model_dists)).values[lb:ub])
        euclidean_dists.append(sq_dists)
    
    min_model_index = euclidean_dists.index(min(euclidean_dists)) 
    log.info("Krum Candidate is {}".format(model_keys[min_model_index]))

    model = model_list[min_model_index]
    if base_model is not None:
        model = sub_model(base_model, model_list[min_model_index])
    return model

def M_Krum(base_model, models, **kwargs):
    model_list = list(models.values())
    model_keys = list(models.keys())

    beta = kwargs["beta"]
    lb = beta//2
    ub = len(model_list) - beta//2 - 1

    euclidean_dists = []
    for index1, model1 in enumerate(model_list):
        model_dists = []
        for index2, model2 in enumerate(model_list):
            if index1 != index2:
                dist = sim.grad_eucliden_dist(model1, model2)
                model_dists.append(dist)
        sq_dists = torch.sum(torch.sort(torch.tensor(model_dists)).values[lb:ub])
        euclidean_dists.append(sq_dists)
            
    min_model_indices = np.argpartition(np.array(euclidean_dists), len(model_list) - 2*beta - 2)
    min_model_indices = min_model_indices[:len(model_list) - 2*beta - 2]
    log.info("M_Krum Candidates are {}".format([model_keys[index] for index in min_model_indices]))
    
    model_list = [model for index, model in enumerate(model_list) if index in min_model_indices]

    model = reduce(add_model, model_list)
    model = scale_model(model, 1.0 / len(models))

    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def Median(base_model, models, **kwargs):
    model_list = list(models.values())
    dummy_model = model_list[0]
    dummy_model_arr, d_list = sim.get_net_arr(dummy_model)

    beta = kwargs["beta"]
    lb = beta
    ub = len(model_list) - beta

    updated_model_list = []
    for model in model_list:
        model_arr, _ = sim.get_net_arr(model)
        updated_model_list.append(model_arr)
        
    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.sort(torch.stack([model for model in updated_model_tensors], 0), dim = 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors.values, 0, 1).numpy()
    merged_updated_model_indices = torch.transpose(merged_updated_model_tensors.indices, 0, 1).numpy()

    model_arr = np.zeros(len(dummy_model_arr))
    for index, arr in enumerate(merged_updated_model_arrs):
        model_arr[index] = np.median(arr)
    model = sim.get_arr_net(dummy_model, model_arr, d_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model

def T_Mean(base_model, models, **kwargs):
    model_list = list(models.values())
    dummy_model = model_list[0]
    dummy_model_arr, d_list = sim.get_net_arr(dummy_model)

    beta = kwargs["beta"]
    lb = beta
    ub = len(model_list) - beta

    updated_model_list = []
    for model in model_list:
        model_arr, _ = sim.get_net_arr(model)
        updated_model_list.append(model_arr)
        
    updated_model_tensors = torch.tensor(updated_model_list)
    merged_updated_model_tensors = torch.sort(torch.stack([model for model in updated_model_tensors], 0), dim = 0)
    merged_updated_model_arrs = torch.transpose(merged_updated_model_tensors.values, 0, 1).numpy()
    
    model_arr = np.zeros(len(dummy_model_arr))
    for index, arr in enumerate(merged_updated_model_arrs):
        model_arr[index] = arr[lb:ub].mean(0)
    model = sim.get_arr_net(dummy_model, model_arr, d_list)
    
    if base_model is not None:
        model = sub_model(base_model, model)
    return model