import inspect
import os.path

import torch
from torchvision import datasets, transforms


def load_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datadir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../data'
    train_data = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)
    return train_data, test_data


def split_data(train_data, clients):
    splitted_data = torch.utils.data.random_split(train_data,
                                                  [int(train_data.data.shape[0] / len(clients)) for _ in range(len(clients))])
    
    clients_data = {client: splitted_data[index] for index, client in enumerate(clients)}
    return clients_data


def load_client_data(clients_data, batch_size, test_data=None, test_batch_size=None, **kwargs):
    train_loaders = {client: torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
                    for client, data in clients_data.items()}

    test_loader = None
    if test_data and test_batch_size:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loaders, test_loader
