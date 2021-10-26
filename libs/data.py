import csv, inspect, json
import os.path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import torch
import torch.autograd as autograd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0 = 1014):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)
        
            
    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase = True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                self.label.append(int(row[0]))
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

        self.y = torch.LongTensor(self.label)


    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class

def load_dataset(dataset):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datadir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/../data'
    
    train_data, test_data = None, None
    
    if dataset.upper() == "AGNEWS":
        train_path = os.path.abspath(os.path.join(os.getcwd(), datadir+"/ag_news_csv/train.csv"))
        train_data = AGNEWs(label_data_path=train_path, alphabet_path=datadir+"/ag_news_csv/alphabet.json", l0=1014)
        
        test_path = os.path.abspath(os.path.join(os.getcwd(), datadir+"/ag_news_csv/test.csv"))
        test_data = AGNEWs(label_data_path=test_path, alphabet_path=datadir+"/ag_news_csv/alphabet.json", l0=1014)
    if dataset.upper() == "MNIST":
        train_data = datasets.MNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.MNIST(root=datadir, train=False, transform=transform, download=True)
    if dataset.upper() == "FMNIST":
        train_data = datasets.FashionMNIST(root=datadir, train=True, transform=transform, download=True)
        test_data = datasets.FashionMNIST(root=datadir, train=False, transform=transform, download=True)
        
    if dataset.upper() == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR10(root=datadir, train=True, transform=transform_train)
        test_data = datasets.CIFAR10(root=datadir, train=False, transform=transform_test)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    if dataset.upper() == "WISCONSIN":
        data_path = os.path.abspath(os.path.join(os.getcwd(), datadir+"/Wisconsin/data.csv"))
        df = pd.read_csv(data_path)
        df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
        lab ={'B':0,'M':1}
        df = df.replace({'diagnosis':lab})

        predictors = df.iloc[:, 2:31]
        labels = df.iloc[:, 0:1]

        predictors_train, predictors_test, labels_train, labels_test = train_test_split(predictors,labels,test_size = 0.20)

        predictors_train = torch.tensor(np.array(predictors_train), dtype=torch.float)
        labels_train = torch.tensor(np.array(labels_train), dtype = torch.float)
        train_data = torch.utils.data.TensorDataset(predictors_train, labels_train)
        
        predictors_test = torch.tensor(np.array(predictors_test), dtype=torch.float)
        labels_test = torch.tensor(np.array(labels_test), dtype = torch.float)
        test_data = torch.utils.data.TensorDataset(predictors_test, labels_test)        

    return train_data, test_data


def random_split(data, ratio):
    split_arr = [int(len(data) * (1-ratio)), int(len(data) * ratio)]
    rem_data = len(data) - sum(split_arr)
    if rem_data > 0:
        split_arr[-1] = split_arr[-1] + rem_data
        
    return torch.utils.data.random_split(data, split_arr)

def split_data(train_data, clients):
    split_arr = [int(len(train_data) / len(clients)) for _ in range(len(clients))]
    rem_data = len(train_data) - sum(split_arr)
    if rem_data > 0:
        split_arr[-1] = split_arr[-1] + rem_data
    
    splitted_data = torch.utils.data.random_split(train_data, split_arr)
    clients_data = {client: splitted_data[index] for index, client in enumerate(clients)}

    return clients_data


def load_client_data(clients_data, batch_size, test_ratio=None, **kwargs):
    train_loaders = {}
    test_loaders = {}

    if test_ratio is None:
        for client, data in clients_data.items():
            train_loaders[client] = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
            
    else:
        for client, data in clients_data.items():
            train_test = torch.utils.data.random_split(data, [int(len(data) * (1-test_ratio)), 
                                                              int(len(data) * test_ratio)])

            train_loaders[client] = torch.utils.data.DataLoader(train_test[0], batch_size=batch_size, shuffle=True, **kwargs)
            test_loaders[client] = torch.utils.data.DataLoader(train_test[1], batch_size=batch_size, shuffle=True, **kwargs)

    return train_loaders, test_loaders