import torch
import pandas as pd
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import scipy
import torch.nn.functional as F
import torchvision

from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve, auc,confusion_matrix
from sklearn.feature_selection import r_regression

from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from math import pi as PI

def scipy_spanning_tree(edge_index, edge_weight,num_nodes ):
    row, col = edge_index.cpu()
    edge_weight=edge_weight.cpu()
    cgraph = csr_matrix((edge_weight, (row, col)), shape=(num_nodes, num_nodes))
    Tcsr = minimum_spanning_tree(cgraph)
    tree_row, tree_col = Tcsr.nonzero()
    spanning_edges = np.stack([tree_row,tree_col],0)    
    return spanning_edges
    
def build_spanning_tree_edge(edge_index,edge_weight, num_nodes):
    spanning_edges = scipy_spanning_tree(edge_index, edge_weight,num_nodes,)
        
    spanning_edges = torch.tensor(spanning_edges, dtype=torch.long, device=edge_index.device)
    spanning_edges_undirected = torch.cat([spanning_edges,torch.stack([spanning_edges[1],spanning_edges[0]])],1)
    return spanning_edges_undirected




def record(values,epoch,writer,phase="Train"):
    """ tfboard write """
    for key,value in values.items():
        writer.add_scalar(key+"/"+phase,value,epoch)           
def calculate(y_hat,y_true,y_hat_logit):
    """ calculate five metrics using y_hat, y_true, y_hat_logit """
    train_acc=(np.array(y_hat) == np.array(y_true)).sum()/len(y_true) 
    # recall=recall_score(y_true, y_hat,zero_division=0,average='micro')
    # precision=precision_score(y_true, y_hat,zero_division=0,average='micro')
    # Fscore=f1_score(y_true, y_hat,zero_division=0,average='micro')
    # roc=roc_auc_score(y_true, scipy.special.softmax(np.array(y_hat_logit),axis=1)[:,1],average='micro',multi_class='ovr')
    # one_hot_encoded_labels = np.zeros((len(y_true), 100))
    # one_hot_encoded_labels[np.arange(len(y_true)), y_true] = 1
    # roc=roc_auc_score(one_hot_encoded_labels, scipy.special.softmax(np.array(y_hat_logit),axis=1),average='micro',multi_class='ovr')
    return train_acc


def print_1(epoch,phase,values,color=None):
    """ print epoch info"""
    if color is not None:
        print(color( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))
    else:
        print(( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))

def get_angle(v1, v2):
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    return torch.atan2( torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
def get_theta(v1, v2):
    # v1 is starting line, right-hand rule to v2, if thumb is up, +, else -
    angle=get_angle(v1, v2)
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    v = torch.cross(v1, v2, dim=1)[...,2]
    flag = torch.sign((v))
    flag[flag==0]=-1 
    return angle*flag   

def triplets(edge_index, num_nodes):
    row, col = edge_index

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_col = adj_t[:,row]
    num_triplets = adj_t_col.set_value(None).sum(dim=0).to(torch.long)

    idx_j = row.repeat_interleave(num_triplets) 
    idx_i = col.repeat_interleave(num_triplets) 
    edx_2nd = value.repeat_interleave(num_triplets) 
    idx_k = adj_t_col.t().storage.col() 
    edx_1st = adj_t_col.t().storage.value()
    mask1 = (idx_i == idx_k) & (idx_j != idx_i)  # Remove go back triplets. 
    mask2 = (idx_i == idx_j) & (idx_j != idx_k)  # Remove repeat self loop triplets
    mask3 = (idx_j == idx_k) & (idx_i != idx_k)  # Remove self-loop neighbors 
    mask = ~(mask1 | mask2 | mask3) 
    idx_i, idx_j, idx_k, edx_1st, edx_2nd = idx_i[mask], idx_j[mask], idx_k[mask], edx_1st[mask], edx_2nd[mask]
    
    num_triplets_real = torch.cumsum(num_triplets, dim=0) - torch.cumsum(~mask, dim=0)[torch.cumsum(num_triplets, dim=0)-1]

    return torch.stack([idx_i, idx_j, idx_k]), num_triplets_real.to(torch.long), edx_1st, edx_2nd
