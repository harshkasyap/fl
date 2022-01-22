import torch

class HDC(torch.nn.Module):
    def __init__(self, img_len, hvd, num_classes, device):
        super(HDC, self).__init__()
        #self.data = []
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.num_classes = num_classes
        self.proj = torch.rand((img_len, hvd), device=device)
        self.proj_inv = self.get_proj_inv()
        self.train_vectors = torch.zeros((num_classes, hvd), device=device)
        
    def get_proj_inv(self):
        proj_trans = torch.transpose(self.proj, 0, 1)
        proj_mul_trans = self.proj @ proj_trans
        proj_mul_trans_inv = torch.inverse(proj_mul_trans)
        proj_inv = proj_trans @ proj_mul_trans_inv
        return proj_inv
        
    def avg(self, client_model_updates):
        for model_update in client_model_updates:
            self.train_vectors += model_update.train_vectors
        
    def train(self, train_loader, device):
        x_train, y_train = next(iter(train_loader))
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        hdc_train = x_train.reshape(x_train.shape[0], -1) @ self.proj
        for i in range(x_train.shape[0]):
            #self.data.append((hdc_train[i], y_train[i]))
            self.train_vectors[y_train[i]] += hdc_train[i]

        return self.test(train_loader, device)
    
    def test(self, test_loader, device):
        x_test, y_test = next(iter(test_loader))
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        x_test = x_test.reshape(x_test.shape[0], -1) @ self.proj
        pred = torch.stack([self.cos(x_test, self.train_vectors[i:i+1]) for i in range(self.num_classes)], axis=-1)
        acc = 100 * torch.mean((pred.argmax(axis=1) == y_test).float())

        return acc.item()
    
'''
from torchvision import datasets, transforms
import copy, os, socket, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import agg, data, fl, log, nn, plot, poison, resnet, sim, wandb

train_data = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=True)


train_data = torch.utils.data.random_split(train_data, [1000, 59000])[0]

train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True,
                               num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False,
                              num_workers=0, pin_memory=True)

import time
start_time = time.time()
hdc = HDC(784, 10000, 10, 'cpu')
print("Train", hdc.train(train_loader, 'cpu'))
print("Time to Train", time.time() - start_time)
print("Test", hdc.test(test_loader, 'cpu'))
'''