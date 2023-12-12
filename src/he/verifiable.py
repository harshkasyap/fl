import asyncio, nest_asyncio
nest_asyncio.apply()

import copy, os, socket, sys, time
from functools import partial
from multiprocessing import Pool, Process
from pathlib import Path
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, data, fl, log, nn, plot, poison, resnet, sim
from cfgs.fedargs import *


# In[3]:


import base64

def writeCkks(ckks_vec, filename):
    ser_ckks_vec = base64.b64encode(ckks_vec)

    with open(filename, 'wb') as f:
        f.write(ser_ckks_vec)

def readCkks(filename):
    with open(filename, 'rb') as f:
        ser_ckks_vec = f.read()
    
    return base64.b64decode(ser_ckks_vec)


# In[4]:


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


# In[5]:


# Initialize Global and Client models
global_model = copy.deepcopy(fedargs.model)
one_d_arr, _list = sim.get_net_arr(global_model)
one_d_arr


# In[ ]:


enc_one_d_arr = ts.ckks_tensor(context, [one_d_arr])
enc_one_d_arr_ser = enc_one_d_arr.serialize()
writeCkks(enc_one_d_arr_ser, "enc_one_d_arr_ser")
print("Serialised plain local model")


# In[6]:


w1 = [1,2,3]
w2 = [2,2,2]

_lambda = 4
#Encode into messages, assume pattern is to encode at index 0 and 2, given lambda is 4
def encode(wi):
    encoded_vec = []
    for identifier, i in enumerate(wi):
        identifier = identifier + 1
        enc_template = [0 for i in range(_lambda)]
        for j in range(_lambda):
            if j == 0 or j == 2:
                enc_template[j] = identifier + j + 1
            else:    
                enc_template[j] = i
        encoded_vec = encoded_vec + enc_template
    return encoded_vec

w1_encoded = encode(w1)
w2_encoded = encode(w2)

enc_w1 = ts.ckks_tensor(context, [w1_encoded])
enc_w2 = ts.ckks_tensor(context, [w2_encoded])

enc_w = (enc_w1 + enc_w2) * 0.5 # Average operation
enc_w.decrypt().tolist()

#To Verify, clients will check at all specified positions, assuming clients know the pattern, identifier and index.


# In[ ]:


encoded_one_d_arr = encode(one_d_arr)
enc_one_d_arr = ts.ckks_tensor(context, [encoded_one_d_arr])
enc_one_d_arr_ser = enc_one_d_arr.serialize()
writeCkks(enc_one_d_arr_ser, "enc_one_d_arr_ser")
print("Serialised encoded local model")

