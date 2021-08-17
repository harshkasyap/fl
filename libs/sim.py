import numpy as nd
from mxnet import nd as mnd

def get_mx_net_arr(model):
    param_list = [param.data.numpy() for param in model.parameters()]
    _param_list = nd.array(param_list).squeeze()

    arr = nd.array([[]])
    for index, item in enumerate(_param_list):
        item = item.reshape((-1, 1))
        if index == 0:
            arr = item
        else:
            arr = nd.concatenate((arr, item), axis=0)

    arr = nd.array(arr).squeeze()
    arr = mnd.array(arr)
    return arr

def grad_norm(model):
    arr = get_mx_net_arr(model)
    return mnd.norm(arr).asnumpy()[0]

def grad_cosine_similarity(model1, model2):
    arr1 = get_mx_net_arr(model1)
    arr2 = get_mx_net_arr(model2)
    cs = mnd.dot(arr1, arr2) / (mnd.norm(arr1) + 1e-9) / (mnd.norm(arr2) + 1e-9)
    return cs.asnumpy()[0]

'''
import torch

def grad_norm(model, p=2):
    parameters = [param for param in model.parameters() if param.grad is not None and param.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in parameters]), p).item()

    return total_norm

def grad_cosine_similarity(model1, model2):
    cos_score=[]
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if len(param1.shape) > 1:
            cos_score.append(torch.nn.functional.cosine_similarity(param1, param2).mean().detach().numpy())

    return sum(cos_score)/len(cos_score)

def grad_cosine_similarity(model1, model2):
    cos_score=[]
    params1 = model1.state_dict().copy()
    params2 = model2.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                if len(params1[name1].shape) > 1:
                    cos_score.append(torch.nn.functional.cosine_similarity(params1[name1], params2[name1]).mean())

    return sum(cos_score)/len(cos_score)
'''