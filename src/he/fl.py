#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

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

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from libs import agg, data, fl, log, nn, plot, poison, resnet, sim, wandb, he
from cfgs.fedargs import *


# In[2]:


project = 'fl'
name = 'fl-he'

#Define seed
torch.manual_seed(1)

#Define Custom CFGs
fedargs.enc = True
fedargs.num_clients = 50

# Save Logs To File (info | debug | warning | error | critical) [optional]
log.init("info")
wb = wandb.init(name, project)


# In[3]:


# Device settings
use_cuda = fedargs.cuda and torch.cuda.is_available()
torch.manual_seed(fedargs.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}


# In[4]:


# Prepare clients
host = socket.gethostname()
clients = [host + "(" + str(client + 1) + ")" for client in range(fedargs.num_clients)]


# In[5]:


# Initialize Global and Client models
global_model = copy.deepcopy(fedargs.model)
# Load Data to clients
train_data, test_data = data.load_dataset(fedargs.dataset)


# In[6]:


clients_data = data.split_data(train_data, clients)

# for fast test
new_data = {}
for index, (client, details) in enumerate(clients_data.items()):
    new_data[client] = details
    
    if index == 9:
        break

clients = clients[:10]
clients_data = new_data
print(len(clients), len(clients_data))


# In[7]:


client_train_loaders, _ = data.load_client_data(clients_data, fedargs.client_batch_size, None, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=fedargs.test_batch_size, shuffle=True, **kwargs)

client_details = {
        client: {"train_loader": client_train_loaders[client],
                 "model": copy.deepcopy(global_model),
                 "model_update": None}
        for client in clients
    }


# In[8]:


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

@background
def process(client, epoch, model, train_loader, fedargs, device):
    # Train
    model_update, model, loss = fedargs.train_func(model, train_loader, 
                                                   fedargs.learning_rate,
                                                   fedargs.weight_decay,
                                                   fedargs.local_rounds, device)

    log.jsondebug(loss, "Epoch {} of {} : Federated Training loss, Client {}".format(epoch, fedargs.epochs, client))
    log.modeldebug(model_update, "Epoch {} of {} : Client {} Update".format(epoch, fedargs.epochs, client))
    
    return model_update

@background
def enc(model_update):
    # Train
    return he.enc_model_update(model_update)


# In[ ]:


import time
start_time = time.time()
    
# Federated Training
for epoch in tqdm(range(fedargs.epochs)):
    log.info("Federated Training Epoch {} of {}".format(epoch, fedargs.epochs))

    # Global Model Update
    if epoch > 0:
        # Average
        if fedargs.enc:
            _, slist = sim.get_net_arr(global_model)            
            avgargs = {"dummy_model": fedargs.model,
                       "slist": slist}
            
            global_model = he.enc_model_update(global_model)
            global_model = he.federated_avg(client_model_updates, global_model, fedargs.agg_rule, **avgargs)
        else:
            global_model = fl.federated_avg(client_model_updates, global_model)
        log.modeldebug(global_model, "Epoch {} of {} : Server Update".format(epoch, fedargs.epochs))
        
        # Test, Plot and Log
        global_test_output = fedargs.eval_func(global_model, test_loader, device)
        wb.log({"epoch": epoch, "time": time.time(), "acc": global_test_output["accuracy"], "loss": global_test_output["test_loss"]})
        log.jsoninfo(global_test_output, "Global Test Outut after Epoch {} of {}".format(epoch, fedargs.epochs))
        
        # Update client models
        for client in clients:
            client_details[client]['model'] = copy.deepcopy(global_model)

    # Clients
    tasks = [process(client, epoch, client_details[client]['model'],
                     client_details[client]['train_loader'],
                     fedargs, device) for client in clients]
    try:
        updates = fedargs.loop.run_until_complete(asyncio.gather(*tasks))
    except KeyboardInterrupt as e:
        log.error("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        fedargs.loop.run_forever()
        tasks.exception()
        
    for client, update in zip(clients, updates):            
        client_details[client]['model_update'] = update

        if fedargs.enc:
            enc_update = he.enc_model_update(update)
            client_details[client]['model_update'] = enc_update
    
    '''
    if fedargs.enc:            
        # Parallel Enc
        tasks = [enc(client_details[client]['model_update']) for client in client_details]
        try:
            updates = fedargs.loop.run_until_complete(asyncio.gather(*tasks))
        except KeyboardInterrupt as e:
            log.error("Caught keyboard interrupt. Canceling tasks...")
            tasks.cancel()
            fedargs.loop.run_forever()
            tasks.exception()
            
        for client, update in zip(clients, updates):            
            client_details[client]['model_update'] = update
    '''

    client_model_updates = {client: details["model_update"] for client, details in client_details.items()}

print(time.time() - start_time)