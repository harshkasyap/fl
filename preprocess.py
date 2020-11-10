import torch
from torchvision import datasets,transforms

def load_mnist_dataset():
    transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
          ])

    train_data = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
    test_data = datasets.MNIST(root='./data',train=False,transform=transform,download=True)
    return train_data, test_data

def split_data(train_data, clients):
    splitted_data = torch.utils.data.random_split(train_data, [int(train_data.data.shape[0] / clients) for _ in range(clients)])
    return splitted_data

def split_label_wise(train_data):
    label_wise_data = []
    for i in range(10):
        templabeldata = []
        j = 0
        for instance, label in train_data:
            if label == i:
                templabeldata.append(train_data[j])
            j += 1
        label_wise_data.append(templabeldata)
    return label_wise_data

def distribute_data_in_clients(label_wise_data):
    clients_data = []
    for i in range(10):
        clients_data.append([])

    dist = [[300,323,400,500,500,2000,500,500,400,500],
            [1000,500,500,500,1000,500,1000,742,500,500],
            [500,458,1000,500,500,500,500,500,500,1000],
            [500,1000,500,500,1000,500,500,1000,500,131],
            [2000,500,500,500,100,100,100,20,22,2000],
            [200,200,100,1800,100,100,200,221,2000,500],
            [500,500,1000,300,300,1000,400,500,418,1000],
            [1000,1000,1000,500,500,500,500,500,265,500],
            [900,450,450,900,225,226,900,900,450,450],
            [500,500,200,200,3000,500,500,249,100,200]]
   
    for lable in range(len(dist)):
        loc = 0
        i = 0
        for ele in dist[lable]:
            for j in range(loc,loc+ele):
                clients_data[i].append(label_wise_data[lable][j])
            i += 1
            loc  += ele
                        
    return clients_data

def load_client_data(train_data, test_data, batch_size):
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in train_data]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle=True) 

    return train_loader, test_loader

def poison_label(client_id, sourcelabel, targetlabel, count_poison, client_data):
    label_poisoned = 0
    client_data[client_id] = list(client_data[client_id])
    i = 0 
    for instance,label in client_data[client_id]:
      client_data[client_id][i] = list(client_data[client_id][i])
      if client_data[client_id][i][1] == sourcelabel:
        client_data[client_id][i][1] = targetlabel
        label_poisoned += 1
      client_data[client_id][i] = tuple(client_data[client_id][i])
      if label_poisoned >= count_poison and count_poison != -1:
        break
      i += 1
    client_data[client_id] = tuple(client_data[client_id])
    return label_poisoned