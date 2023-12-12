#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.autograd import Variable
import numpy as nd
import time
'''
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import data as dt, neuronshap as ns, sim
from cfgs.fedargs import *
'''
from openfhe import *

# ML Model
'''
class MnistModel(torch.nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = torch.nn.Linear(64*7*7, 1024)
        self.fc2 = torch.nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    def forward_test(self, x):
        res = {}
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        res["layer_1"] = x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        res["layer_2"] = x
        x = x.view(-1, 64*7*7)   # reshape Variable
        res["layer_3"] = x        
        x = F.relu(self.fc1(x))
        res["layer_4"] = x        
        x = F.dropout(x, training=self.training)
        res["layer_5"] = x        
        x = self.fc2(x)
        res["layer_6"] = x        
        return res
    
model = MnistModel()
model
'''

# CKKS

arr = nd.random.rand(3274634)
arr = nd.random.rand(1000)
print(arr[0:10])

def get_enc_model(arr, he_batch_size = 8000):
    if len(arr)%he_batch_size != 0:
        arr = nd.append(arr, [0 for i in range(he_batch_size - len(arr)%he_batch_size)])

    print(len(arr))
    extd_arr = nd.split(arr, len(arr)/he_batch_size)
                        
    enc_model = []
    for index, element in enumerate(extd_arr):
        ptxt = cryptocontext.MakeCKKSPackedPlaintext(element,1,depth-1)
        ptxt.SetLength(he_batch_size)

        ciph = cryptocontext.Encrypt(key_pair.publicKey, ptxt)
        ciphertext_after = cryptocontext.EvalBootstrap(ciph)

        SerializeToFile("./enc_model"+str(index), ciphertext_after, BINARY)
        enc_model.append(ciphertext_after)
    return enc_model

def get_dec_model(enc_model, he_batch_size = 8000):
    dec_arr = nd.array([])
    for element in enc_model:
        dec = cryptocontext.Decrypt(element,key_pair.secretKey)
        dec.SetLength(he_batch_size)
        
        dec = str(dec)[1:][:-1].split(',')
        dec = nd.array([(d) for d in dec])
        
        dec_arr = nd.append(dec_arr, dec)

    return dec_arr

parameters = CCParamsCKKSRNS()

secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
parameters.SetSecretKeyDist(secret_key_dist)

parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
parameters.SetRingDim(1<<14)

if get_native_int()==128:
    rescale_tech = ScalingTechnique.FIXEDAUTO
    dcrt_bits = 78
    first_mod = 89
else:
    rescale_tech = ScalingTechnique.FLEXIBLEAUTO
    dcrt_bits = 59
    first_mod = 60

parameters.SetScalingModSize(dcrt_bits)
parameters.SetScalingTechnique(rescale_tech)
parameters.SetFirstModSize(first_mod)

level_budget = [4, 4]

levels_available_after_bootstrap = 10

depth = levels_available_after_bootstrap + FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)

parameters.SetMultiplicativeDepth(depth)

cryptocontext = GenCryptoContext(parameters)
cryptocontext.Enable(PKESchemeFeature.PKE)
cryptocontext.Enable(PKESchemeFeature.KEYSWITCH)
cryptocontext.Enable(PKESchemeFeature.LEVELEDSHE)
cryptocontext.Enable(PKESchemeFeature.ADVANCEDSHE)
cryptocontext.Enable(PKESchemeFeature.FHE)

ring_dim = cryptocontext.GetRingDimension()
# This is the mazimum number of slots that can be used full packing.

num_slots = int(ring_dim / 2)
print(f"CKKS is using ring dimension {ring_dim}")

cryptocontext.EvalBootstrapSetup(level_budget)

key_pair = cryptocontext.KeyGen()
cryptocontext.EvalMultKeyGen(key_pair.secretKey)
cryptocontext.EvalBootstrapKeyGen(key_pair.secretKey, num_slots)

start = time.time()
enc_model = get_enc_model(arr)
end = time.time()
print(end-start)

enc_model = DeserializeCiphertext("enc_model", SERBINARY)
print(get_dec_model(enc_model)[0:10])