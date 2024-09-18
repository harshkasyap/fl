import base64
import numpy as np

import asyncio, inspect, os, sys
from libs import agg
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
argsdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def writeCkks(ckks_vec, filename):
    ser_ckks_vec = base64.b64encode(ckks_vec)

    with open(filename, 'wb') as f:
        f.write(ser_ckks_vec)

def readCkks(filename):
    with open(filename, 'rb') as f:
        ser_ckks_vec = f.read()
    
    return base64.b64decode(ser_ckks_vec)

import tenseal as ts

poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale= 2**40

context = ts.context(
            ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree = poly_modulus_degree,
            coeff_mod_bit_sizes = coeff_mod_bit_sizes
            )
context.generate_galois_keys()
context.global_scale = global_scale

public_context = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)

def encrypt(arr, batch = True):
    enc_arr = ts.ckks_tensor(context, arr, None, batch)
    return enc_arr

def decrypt(enc_arr):
    return enc_arr.decrypt().tolist()

def serialise(enc_arr, filename):
    writeCkks(enc_querier.serialize(), "/../out/enc/"+ filename)
    
def enc_model_update(model_params):
    for key in model_params.keys():
        prepared_tensor = (torch.flatten(model_params[key]))
        np_params[key] = ts.plain_tensor(prepared_tensor)

    enc_model_params = OrderedDict()
    for key in np_params.keys():
        enc_model_params[key] = encrypt(np_params[key])
    return enc_model_params

def FedAvg(list_enc_model_parmas):
    # init a template model
    n_clients = len(list_enc_model_parmas)
    temp_sample_number, temp_model_params = list_enc_model_parmas[0]
    enc_global_params = copy.deepcopy(temp_model_params)

    for i in range(n_clients):
        list_enc_model_parmas[i] = list_enc_model_parmas[i][1]
        for key in enc_global_params.keys():
            list_enc_model_parmas[i][key] = fhe_core.ckks_vector_from(self.he_context,
                                                                      list_enc_model_parmas[i][key])

    for key in enc_global_params.keys():
        for i in range(n_clients):
            if i != 0:
                # temp = list_enc_model_parmas[i][key] * weight_factors[key]
                temp = list_enc_model_parmas[i][key]
                list_enc_model_parmas[0][key] = list_enc_model_parmas[0][key] + temp

    for key in enc_global_params.keys():
        list_enc_model_parmas[0][key] = list_enc_model_parmas[0][key].serialize()

    enc_global_params = list_enc_model_parmas[0]
    return enc_global_params

def federated_avg(models, base_model, rule):
    if len(models) > 1:
        if rule is agg.Rule.FedAvg:
            model = agg.FedAvg(base_model, models)
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