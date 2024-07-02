import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pandas as pd
import sys
import pickle
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
from datetime import date, timedelta,datetime
import random
import pickle as pkl
import string

valid_chars = 'EFHILOTUYZ'

alphabetic_labels = [char1 + char2 for char1 in valid_chars for char2 in valid_chars]
alphabetic_labels.sort()
label_mapping = {label: idx for idx, label in enumerate(alphabetic_labels)} # to number
reverse_label_mapping = {v: k for k, v in label_mapping.items()} # to alphabetic

single_alphabetic_labels=[char1 for char1 in valid_chars]
single_alphabetic_labels.sort()
single_label_mapping = {label: idx for idx, label in enumerate(single_alphabetic_labels)}
single_reverse_label_mapping = {v: k for k, v in single_label_mapping.items()}

def get_mnist_dataset(data_dir='data/multi_mnist.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y -= 10
                
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_building_dataset(data_dir='data/building_with_index.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = label_mapping[entry.y]  
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_mbuilding_dataset(data_dir='data/mp_building.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = label_mapping[entry.y]  
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_sbuilding_dataset(data_dir='data/single_building.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = single_label_mapping[entry.y]  
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_smnist_dataset(data_dir='data/single_mnist.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_dbp_dataset(data_dir='data/triple_building.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = 1 if entry.y>=1 else 0
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def affine_transform_to_range(ds, target_range=(-1, 1)):
    # Find the extent (min and max) of coordinates in both x and y directions
    for item in ds:
        min_x  = torch.min(item.pos[:,0])
        min_y  = torch.min(item.pos[:,1])
        
        max_x  = torch.max(item.pos[:,0])
        max_y  = torch.max(item.pos[:,1])
        
        scale_x = (target_range[1] - target_range[0]) / (max_x - min_x)
        scale_y = (target_range[1] - target_range[0]) / (max_y - min_y)
        translate_x = target_range[0] - min_x * scale_x
        translate_y = target_range[0] - min_y * scale_y

        # Apply the affine transformation to 
        item.pos[:,0] = item.pos[:,0] * scale_x + translate_x
        item.pos[:,1] = item.pos[:,1] * scale_y + translate_y
    return ds

class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
if __name__ == '__main__':
    a,b,c=get_mnist_dataset()
    print("")