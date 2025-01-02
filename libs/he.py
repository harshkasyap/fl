import base64
import numpy as np

import asyncio, inspect, itertools, os, sys, time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, sim

argsdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

enc_type = "ckks"
split_arr_len = 4000
quant_factor = 1000000

def writeSerFile(enc_vec, filename):
    ser_vec = base64.b64encode(enc_vec)

    with open(filename, 'wb') as f:
        f.write(ser_vec)

def readSerFile(filename):
    with open(filename, 'rb') as f:
        ser_vec = f.read()
    
    return base64.b64decode(ser_vec)

import tenseal as ts

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale= 2**40

context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree = poly_modulus_degree,
            coeff_mod_bit_sizes = coeff_mod_bit_sizes
            )

if enc_type == "bfv":
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)

context.generate_galois_keys()
context.global_scale = global_scale

public_context = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)

def encrypt(arr, batch = True):
    if enc_type == "ckks":
        enc_arr = ts.ckks_tensor(context, arr, None, batch)
    if enc_type == "bfv":
        enc_arr = ts.bfv_tensor(context, arr, batch)
    return enc_arr

def decrypt(enc_arr):
    arr = np.array(enc_arr.decrypt().tolist())        
    if enc_type == "bfv":
        arr = arr / quant_factor
    return arr
    

def serialise(enc_arr, filename):
    writeSerFile(enc_querier.serialize(), "/../out/enc/"+ filename)

'''
def enc_model_update(model_params):
    for key in model_params.keys():
        prepared_tensor = (torch.flatten(model_params[key]))
        np_params[key] = ts.plain_tensor(prepared_tensor)

    enc_model_params = OrderedDict()
    for key in np_params.keys():
        enc_model_params[key] = encrypt(np_params[key])
    return enc_model_params
'''

def enc_model_update(model_params):
    start_time = time.time()
    arr, slist = sim.get_net_arr(model_params)
    
    #quantise for bfv enc
    if enc_type == "bfv":
        arr = (arr * quant_factor).astype(int)
    
    enc_arr = []
    
    split_arr = np.array_split(arr, len(arr)/split_arr_len)

    for each_arr in split_arr:
        enc_arr.append(encrypt(each_arr))
        
    #print(time.time() - start_time)    
    return enc_arr

def sub_model(base_model, model):
    for index, (global_arr, model_arr) in enumerate(zip(base_model, model)):
        base_model[index].sub_(model_arr)
    return base_model

def FedAvg(base_model, models, **kwargs):
    # init a template model
    model_list = list(models.values())
    n_clients = len(model_list)
    weight = 1/n_clients
    
    dummy_model = kwargs["dummy_model"]
    slist = kwargs["slist"]
  
    # Add
    agg_model = []
    for index1, each_model in enumerate(model_list):
        if index1 == 0:
            agg_model = model_list[0]
        else:
            for index2, each_arr in enumerate(each_model):
                agg_model[index2].add_(each_arr)


    # decryption and weighted avg
    dec_model = []
    for index, each_arr in enumerate(agg_model):
        dec_model.append(decrypt(each_arr) * weight)
    dec_model = np.array(list(itertools.chain.from_iterable(dec_model)))        

    # Sub    
    base_model_arr, _ = sim.get_net_arr(base_model)        
    dec_model = base_model_arr - dec_model
    
    '''
    # Weighted Avg
    for index2, each_arr in enumerate(agg_model):
        agg_model[index2].mul_(weight)
    
    # Sub
    if base_model is not None:
        agg_model = sub_model(base_model, agg_model)

    # decryption
    dec_model = []
    for index, each_arr in enumerate(agg_model):
        dec_model.append(decrypt(each_arr))
        
    dec_model = np.array(list(itertools.chain.from_iterable(dec_model)))
    '''
        
    model = sim.get_arr_net(dummy_model, dec_model, slist)
    
    return model
    
def federated_avg(models, base_model, rule = agg.Rule.FedAvg, **kwargs):
    if len(models) > 1:
        if rule is agg.Rule.FedAvg:
            model = FedAvg(base_model, models, **kwargs)
        if rule is agg.Rule.FedVal:
            model = agg.FedVal(base_model, models, **kwargs)
        if rule is agg.Rule.FoolsGold:
            model = agg.FoolsGold(base_model, models, **kwargs)            
        if rule is agg.Rule.FLTrust:
            model = agg.FLTrust(base_model, models, **kwargs)
        if rule is agg.Rule.FLTC:
            model = agg.FLTC(base_model, models, **kwargs)
        if rule is agg.Rule.Krum:
            model = agg.Krum(base_model, models, **kwargs)
        if rule is agg.Rule.M_Krum:
            model = agg.M_Krum(base_model, models, **kwargs)
        if rule is agg.Rule.Median:
            model = agg.Median(base_model, models, **kwargs)
        if rule is agg.Rule.T_Mean:
            model = agg.T_Mean(base_model, models, **kwargs)
        if rule is agg.Rule.DnC:
            model = agg.DnC(base_model, models, **kwargs)            
    else:
        model = copy.deepcopy(list(models.values())[0])
    return model