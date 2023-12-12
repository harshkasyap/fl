import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import data as dt, neuronshap as ns, sim
from cfgs.fedargs import *
from openfhe import *

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

arr, slist = sim.get_net_arr(model)

# CKKS
parameters = CCParamsCKKSRNS()

secret_key_dist = SecretKeyDist.UNIFORM_TERNARY
parameters.SetSecretKeyDist(secret_key_dist)

parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
parameters.SetRingDim(1<<12)

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


x = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
#x = [i for i in range(1000)]
#encoded_length = len(x)
encoded_length = len(x)

ptxt = cryptocontext.MakeCKKSPackedPlaintext(x,1,depth-1)
ptxt.SetLength(encoded_length)


print(f"Input: {ptxt}")

ciph = cryptocontext.Encrypt(key_pair.publicKey, ptxt)

print(f"Initial number of levels remaining: {depth - ciph.GetLevel()}")

ciphertext_after = cryptocontext.EvalBootstrap(ciph)

print(f"Number of levels remaining after bootstrapping: {depth - ciphertext_after.GetLevel()}")

result = cryptocontext.Decrypt(ciphertext_after,key_pair.secretKey)
result.SetLength(encoded_length)
print(f"Output after bootstrapping: {result}")