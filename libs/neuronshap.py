import random, torch, os, sys
from itertools import permutations
from functools import partial
from multiprocessing import Pool, Process
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import sim

def calculate_shapley_values_fa(model, data_loader, repeats=100):
    model_arr, model_slist = sim.get_net_arr(model)
    num_neurons = len(model_arr)
    shapley_values = torch.zeros(num_neurons).numpy()  # Initialize Shapley values for each neuron
    
    for x, y in data_loader:

        for i in range(repeats):
            perm = random.sample(range(num_neurons), int(num_neurons*0.25))  # Randomly sample a permutation
            
            # Set all neurons to zero except the ones in the current permutation
            zeroed_neurons = torch.ones(num_neurons)
            zeroed_neurons[list(perm)] = 0
            zeroed_model = np.multiply(model_arr, zeroed_neurons.numpy())
                
            zeroed_model = sim.get_arr_net(model, zeroed_model, model_slist)
            
            # Compute the output with the zeroed neurons
            zeroed_output = zeroed_model(x)
            loss = F.cross_entropy(zeroed_output, y)
            loss.backward()
            
            prev_index = 0
            index = 0
            for param in zeroed_model.parameters():
                prev_index = index
                index = index + len(param.flatten())
                if param.grad != None:
                    shapley_values[prev_index:index] = shapley_values[prev_index:index] + np.abs(param.grad.detach().numpy().flatten() * model_arr[prev_index:index])
    
    return shapley_values

def calculate_shapley_values(model, dataset):
    model_arr, model_slist = sim.get_net_arr(model) # convert into one-d array, s_list preserving the structure of the model
    num_neurons = len(model_arr) # total number of neurons
    shapley_values = torch.zeros(num_neurons)  # Initialize Shapley values for each neuron
    
    for i in range(num_neurons):
        contributions = []
        
        # Iterate over all possible permutations of neuron combinations
        for perm in permutations(range(num_neurons), i+1):
            total_contribution = 0.0
            
            for data in dataset:
                inputs, _ = data
                inputs = inputs.unsqueeze(0)
                
                # Set all neurons to zero except the ones in the current permutation
                zeroed_neurons = torch.ones(num_neurons)
                zeroed_neurons[list(perm)] = 0
                zeroed_model = np.multiply(model_arr, zeroed_neurons.numpy())
                
                zeroed_model = sim.get_arr_net(model, zeroed_model, model_slist)
                
                # Compute the output with the zeroed neurons
                zeroed_output = zeroed_model(inputs)
                
                # Compute the difference in output when adding the current permutation
                diff = model(inputs) - zeroed_output
                
                total_contribution += diff.abs().sum().item()
            
            # Calculate the average marginal contribution for the current permutation
            avg_contribution = total_contribution / len(dataset)
            
            contributions.append(avg_contribution)
        
        # Calculate the Shapley value for the current neuron
        shapley_values[i] = sum(contributions) / len(contributions)
    
    return shapley_values


def calculate_shapley_values_mc(model, dataset, num_samples=10):
    model_arr, model_slist = sim.get_net_arr(model)
    num_neurons = len(model_arr)
    shapley_values = torch.zeros(num_neurons)  # Initialize Shapley values for each neuron
    
    for i in range(num_neurons):
        print(i)
        total_contribution = 0.0
        
        for _ in range(num_samples):
            perm = random.sample(range(num_neurons), i+1)  # Randomly sample a permutation
            
            data = random.choice(dataset)  # Randomly select a data point from the dataset
            inputs, _ = data
            inputs = inputs.unsqueeze(0)
            
            # Set all neurons to zero except the ones in the current permutation
            zeroed_neurons = torch.ones(num_neurons)
            zeroed_neurons[list(perm)] = 0
            zeroed_model = np.multiply(model_arr, zeroed_neurons.numpy())
                
            zeroed_model = sim.get_arr_net(model, zeroed_model, model_slist)
            
            # Compute the output with the zeroed neurons
            zeroed_output = zeroed_model(inputs)
            
            # Compute the difference in output when adding the current permutation
            diff = model(inputs) - zeroed_output
            
            total_contribution += diff.abs().sum().item()
        
        # Calculate the average contribution for the current neuron
        avg_contribution = total_contribution / num_samples
        
        # Set the Shapley value for the current neuron
        shapley_values[i] = avg_contribution
    
    return shapley_values

def shapley_value_mc(model, model_arr, num_neurons, dataset, i, _):
    perm = random.sample(range(num_neurons), i+1)  # Randomly sample a permutation
            
    data = random.choice(dataset)  # Randomly select a data point from the dataset
    inputs, _ = data
    inputs = inputs.unsqueeze(0)

    # Set all neurons to zero except the ones in the current permutation
    zeroed_neurons = torch.ones(num_neurons)
    zeroed_neurons[list(perm)] = 0
    zeroed_model = np.multiply(model_arr, zeroed_neurons.numpy())

    zeroed_model = sim.get_arr_net(model, zeroed_model, model_slist)

    # Compute the output with the zeroed neurons
    zeroed_output = zeroed_model(inputs)

    # Compute the difference in output when adding the current permutation
    diff = model(inputs) - zeroed_output
    
    return diff

def calculate_shapley_values_mcp(model, dataset, num_samples=10):
    model_arr, model_slist = sim.get_net_arr(model)
    num_neurons = len(model_arr)
    shapley_values = torch.zeros(num_neurons)  # Initialize Shapley values for each neuron
    
    for i in range(num_neurons):
        print(i)
        total_contribution = 0.0
        
        torch.multiprocessing.set_sharing_strategy('file_system')
        with Pool(num_samples) as p:
            func = partial(shapley_value_mc, model, model_arr, num_neurons, dataset, i)
            sv_mc = p.map(func, [_ for _ in range(num_samples)])
            p.close()
            p.join()

        for diff in sv_mc:
            total_contribution += diff.abs().sum().item()

        # Calculate the average contribution for the current neuron
        avg_contribution = total_contribution / num_samples
        
        # Set the Shapley value for the current neuron
        shapley_values[i] = avg_contribution
    
    return shapley_values