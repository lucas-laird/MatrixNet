import json
import numpy as np
import pandas as pd
from pathlib import Path

import sympy
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup

from torch.utils.data import Dataset, DataLoader, random_split
import torch

np.set_printoptions(threshold=np.inf)

def json_formatter(data_dict):
    keys = list(data_dict.keys())
    data_list = []
    for key in keys:
        braids = data_dict[key]
        for braid in braids:
            word = braid[0]
            coords = braid[1]
            data = (int(key), braid[0], braid[1])
            data_list.append(data)
    return data_list

def onehot(word, word_length, num_generators = 2):
    #Creates onehot encoding of braid word: Set of N generators are mapped to a binary vector. Generator g_i has a 1 in index i and 0 elsewhere
    #Note that generator g_i and its inverse are treated as 2 independent generators. If there are N generators with N inverses, the binary vector will be 2N dim.
    if word_length < 1:
        word_length = 1
    val = torch.zeros((word_length, num_generators*2))
    for (i,a) in enumerate(word):
        if a < 0:
            temp = torch.abs(a)
            j = int(temp + num_generators - 1)
        else:
            j = int(a - 1)
        val[i, j] = 1
    return val
            
def signed_onehot(word, word_length, num_generators = 2):
    #Signed version of onehot encoding. Same as normal onehot, g_i will have a 1 in index i and 0 elsewhere. 
    #Instead of treating inverses as indepedent, inv(g_i) will have a -1 at index i and 0 elsewhere. N generators and N inverses will have N dim vectors.
    if word_length < 1:
        word_length = 1
    val = torch.zeros((word_length, num_generators))
    for (i,a) in enumerate(word):
        j = int(torch.abs(a) - 1)
        if a < 0:
            val[i,j] =  -1
        else:
            val[i,j] = 1
    return val

class onehot_transform:
    #Transform class for data loading in pytorch. Applies the onehot transformation to the input data.
    def __init__(self, num_generators = 2, signed = False):
        self.num_generators = num_generators
        self.signed = signed
    def __call__(self, word, word_length):
        if self.signed:
            val = signed_onehot(word, word_length, self.num_generators)
        else:
            val = onehot(word, word_length, num_generators = self.num_generators)
        return val
        
class BraidDataset(Dataset):
    #Pytorch dataset class for loading in data from JSON file. Has options for variable word length but are not used as of yet. 
    def __init__(self, data_file, transform = None, max_length = False,
                 target_transform = None):
        with open(data_file, "r") as f:
            data_dict = json.load(f)
        self.data_list = json_formatter(data_dict)
        if max_length:
            self.max_length = max([d[0] for d in self.data_list])
        else:
            self.max_length = -1
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        #word_length = torch.Tensor([data[0]])
        word = torch.Tensor(data[1])
        coords = torch.Tensor(data[2])
        if self.max_length == -1:
            if data[0] < 1:
                word_length = 1
            else:
                word_length = data[0]
        else:
            word_length = self.max_length
        if self.transform:
            word = self.transform(word, word_length)
        if self.target_transform:
            coords = self.target_transform(coords)
        word_length = torch.Tensor([word_length])
        return word_length, word, coords



####################################### SYMMETRIC GROUP DATASET GENERATION CODE #####################################################


def gen_onehot_tensor(gens, max_word_len, samples): #Returns a gens x max_word_len onehot matrix on each slice of last dim
    onehot_tensor = np.eye(gens)[np.random.choice(gens, max_word_len)]

    for i in range(samples-1):
        onehot_tensor = np.dstack((onehot_tensor,rand_onehot(gens,max_word_len)))
        
    return onehot_tensor

def save_to_json(onehot_tensor, perm_orders, num_gens, max_word_len, filename, save_dir = None):
    data_dict = {"num_gens": gens, "max_word_len": max_word_len, "data": []}
    samples = len(perm_orders)
    if save_dir:
        Path(save_dir).mkdir(parents = True, exist_ok = True)
        filename = save_dir+filename
    for i in range(samples):
        inds = onehot_tensor[:,:,i].nonzero()[1].tolist()
        order = int(perm_orders[i])
        data_dict["data"].append((inds, order))
    with open(filename, "w") as f:
        json.dump(data_dict, f)

def perm_tensor_orders(onehot_tensor): #Returns a list of permutation orders corresponding to the samples of onehot_tensor
    permutation_orders = []
    for i in range((onehot_tensor.shape)[-1]):
        P = Permutation()
        E = onehot_tensor[:,:,i]
        
        for j in range(len(E)):
            for k in range(len(E[j])):
                if E[j][k] == 1:
                    P = P*sigma[k]
                    
        permutation_orders.append(P.order())
    return permutation_orders
            
        