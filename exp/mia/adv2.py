import sys 
import numpy as np 
#import matplotlib.pyplot as plt
#%matplotlib inline  
from skimage import io
import pickle 

import torch
import torchvision 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset

sys.path.insert(0, '../../ml-leaks')

import models
from train import *
from metrics import *  
from data_downloaders import *

print("Python: %s" % sys.version)
print("Pytorch: %s" % torch.__version__)

# determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_epochs = 20
batch_size = 48
lr = 0.01
k = 3

target_net_type = models.mlleaks_cnn
shadow_net_type = models.mlleaks_cnn

# define series of transforms to pre process images 
train_transform = torchvision.transforms.Compose([  
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
    


# load training set 
cifar100_trainset = torchvision.datasets.CIFAR100('./data/', train=True, transform=train_transform, download=True)
cifar100_trainloader = torch.utils.data.DataLoader(cifar100_trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# load test set 
cifar100_testset = torchvision.datasets.CIFAR100('./data/', train=False, transform=test_transform, download=True)
cifar100_testloader = torch.utils.data.DataLoader(cifar100_testset, batch_size=batch_size, shuffle=False, num_workers=2)

# helper function to unnormalize and plot image 
def imshow(img):
    img = np.array(img)
    img = img / 2 + 0.5
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)
    
# display sample from dataset 
imgs,labels = iter(cifar100_trainloader).next()
imshow(torchvision.utils.make_grid(imgs))

get_lfw('./data/')


data_dir = "./data/lfw/lfw_20/"

img_paths = []
for p in os.listdir(data_dir): 
    for i in os.listdir(os.path.join(data_dir,p)): 
        img_paths.append(os.path.join(data_dir,p,i))
        
people = []
people_to_idx = {}
people_idx = 0 
for i in img_paths: 
    name = i.split('/')[-2]
    if name not in people_to_idx: 
        people.append(name)
        people_to_idx[name] = people_idx
        people_idx += 1


n_lfw_classes = len(people)

img_paths = np.random.permutation(img_paths)

lfw_size = len(img_paths)

lfw_train_size = int(0.8 * lfw_size)

lfw_train_list = img_paths[:lfw_train_size]
lfw_test_list = img_paths[lfw_train_size:]

class LFWDataset(Dataset): 
    def __init__(self, file_list, class_to_label, transform=None): 
        self.file_list = file_list
        self.transform = transform
        
        self.people_to_idx = class_to_label
        
                
    def __len__(self): 
        return len(self.file_list)
    def __getitem__(self, idx): 
        img_path = self.file_list[idx]
        image = io.imread(img_path)
        label = self.people_to_idx[img_path.split('/')[-2]]
        
        if self.transform is not None: 
            image = self.transform(image)
        
        return image, label
        

# Data augmentation 
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
    

lfw_trainset = LFWDataset(lfw_train_list, people_to_idx, transform=train_transform)
lfw_testset = LFWDataset(lfw_test_list, people_to_idx, transform=test_transform)

lfw_trainloader = torch.utils.data.DataLoader(lfw_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
lfw_testloader = torch.utils.data.DataLoader(lfw_testset, batch_size=batch_size, shuffle=False, num_workers=2)


# display sample from dataset 
imgs,labels = iter(lfw_trainloader).next()
imshow(torchvision.utils.make_grid(imgs))

# the model being attacked (architecture can be different than shadow)
target_net = target_net_type(n_out=100).to(device)
target_net.apply(models.weights_init)

target_loss = nn.CrossEntropyLoss()
target_optim = optim.Adam(target_net.parameters(), lr=lr)


# shadow net mimics the target network (architecture can be different than target)
shadow_net = shadow_net_type(n_out=n_lfw_classes, size=64).to(device)
shadow_net.apply(models.weights_init)

shadow_loss = nn.CrossEntropyLoss()
shadow_optim = optim.Adam(shadow_net.parameters(), lr=lr)


# attack net is a binary classifier to determine membership 

attack_net = models.mlleaks_mlp(n_in=k).to(device)
attack_net.apply(models.weights_init)

attack_loss = nn.BCEWithLogitsLoss()
#attack_loss = nn.BCELoss()
#attack_optim = optim.Adam(attack_net.parameters(), lr=lr)
attack_optim = optim.SGD(attack_net.parameters(), momentum=0.7, nesterov=True,lr=lr)

train(shadow_net, lfw_trainloader, lfw_testloader, shadow_optim, shadow_loss, n_epochs)