import torch
import torch.nn.functional as F

from libs import log

def client_update(current_local_model, train_loader, optimizer, epoch):
    current_local_model.train()
    for e in range(epoch):
      running_loss = 0
      for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = current_local_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    log.logger.info("Epoch: %s - Training loss: %s", e, running_loss/len(train_loader))
    return loss.item()

def server_aggregate(global_model, client_models):  
    global_dict = global_model.state_dict()   
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())