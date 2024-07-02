import argparse
import copy
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import cm
from sklearn.metrics import (auc, explained_variance_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from torch.nn.functional import softmax
from torch_geometric.utils import subgraph

torch.autograd.set_detect_anomaly(True)
import math
import pickle
import time
from datetime import date, datetime, timedelta

import torch.nn as nn
import torch_geometric
import torchvision.datasets
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GIN, MLP, GATConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

import dataset
import model_new
import util
from dataset import label_mapping, reverse_label_mapping
from model_new import Smodel

blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="our", type=str)
    parser.add_argument('--train_batch', default=64, type=int)
    parser.add_argument('--test_batch', default=128, type=int)
    parser.add_argument('--share', type=str, default="0")
    parser.add_argument('--edge_rep', type=str, default="True")
    parser.add_argument('--batchnorm', type=str, default="True")
    parser.add_argument('--extent_norm', type=str, default="T")
    parser.add_argument('--spanning_tree', type=str, default="T")
    
    parser.add_argument('--loss_coef', default=0.1, type=float)
    parser.add_argument('--h_ch', default=512, type=int)
    parser.add_argument('--localdepth', type=int, default=1)
    parser.add_argument('--num_interactions', type=int, default=4)
    parser.add_argument('--finaldepth', type=int, default=4)
    parser.add_argument('--classifier_depth', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--log', type=str, default="True")   
    parser.add_argument('--test_per_round', type=int, default=10)
    parser.add_argument('--patience', type=int, default=30)  #scheduler
    parser.add_argument('--nepoch', type=int, default=301)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--manualSeed', type=str, default="False")
    parser.add_argument('--man_seed', type=int, default=12345)
    args = parser.parse_args()
    args.log=True if args.log=="True" else False 
    args.edge_rep=True if args.edge_rep=="True" else False
    args.batchnorm=True if args.batchnorm=="True" else False
    args.save_dir=os.path.join('./save/',args.dataset,args.model)
    args.manualSeed=True if args.manualSeed=="True" else False
    return args

args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion=nn.CrossEntropyLoss()
if args.dataset in ["mnist"]:
    x_out=90
    args.data_dir='data/multi_mnist_with_index.pkl'
elif args.dataset in ["mnist_sparse"]:
    x_out=90
    args.data_dir='data/multi_mnist_sparse.pkl'
elif args.dataset in ["building"]:
    x_out=100
    args.data_dir='data/building_with_index.pkl'
elif args.dataset in ["mbuilding"]:
    x_out=100
    args.data_dir='data/mp_building.pkl'
elif args.dataset in ["sbuilding"]:
    x_out=10
    args.data_dir='data/single_building.pkl'
elif args.dataset in ["smnist"]:
    x_out=10
    args.data_dir='data/single_mnist.pkl'
elif args.dataset in ["dbp"]:
    x_out=2
    args.data_dir='data/triple_building_600.pkl'
    
    
if args.model=="our":
    model=Smodel(h_channel=args.h_ch,input_featuresize=args.h_ch,\
                    localdepth=args.localdepth,num_interactions=args.num_interactions,finaldepth=args.finaldepth,share=args.share,batchnorm=args.batchnorm)
    mlpmodel=MLP(in_channels=args.h_ch*args.num_interactions, hidden_channels=args.h_ch,out_channels=x_out, num_layers=args.classifier_depth,dropout=args.dropout)

elif args.model=="HGT":
    model=model_new.HGT(hidden_channels=args.h_ch, out_channels=args.h_ch, num_heads=2, num_layers=args.num_interactions)
    mlpmodel=MLP(in_channels=args.h_ch, hidden_channels=args.h_ch,out_channels=x_out, num_layers=args.classifier_depth,dropout=args.dropout)
elif args.model=="HAN":
    model=model_new.HAN(hidden_channels=args.h_ch, out_channels=args.h_ch, num_heads=2, num_layers=args.num_interactions)
    mlpmodel=MLP(in_channels=args.h_ch, hidden_channels=args.h_ch,out_channels=x_out, num_layers=args.classifier_depth,dropout=args.dropout)

model.to(device), mlpmodel.to(device)
opt_list=list(model.parameters())+list(mlpmodel.parameters())

optimizer = torch.optim.Adam( opt_list, lr=args.lr)    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   

def contrastive_loss(embeddings,labels,margin):
    
    positive_mask = labels.view(-1, 1) == labels.view(1, -1)
    negative_mask = ~positive_mask

    # Calculate the number of positive and negative pairs
    num_positive_pairs = positive_mask.sum() - labels.shape[0] 
    num_negative_pairs = negative_mask.sum()

    # If there are no negative pairs, return a placeholder loss
    if num_negative_pairs==0 or num_positive_pairs== 0:
        print("all pos or neg")
        return torch.tensor(0, dtype=torch.float)
    # Calculate the pairwise Euclidean distances between embeddings
    distances = torch.cdist(embeddings, embeddings)/np.sqrt(embeddings.shape[1])
    
    if num_positive_pairs>num_negative_pairs:
        # Sample an equal number of + pairs 
        positive_indices = torch.nonzero(positive_mask)
        random_positive_indices = torch.randperm(len(positive_indices))[:num_negative_pairs]
        selected_positive_indices = positive_indices[random_positive_indices]

        # Select corresponding negative pairs
        negative_mask.fill_diagonal_(False)
        negative_distances = distances[negative_mask].view(-1, 1)
        positive_distances = distances[selected_positive_indices[:,0],selected_positive_indices[:,1]].view(-1, 1)
    else: # case for most datasets
        # Sample an equal number of - pairs 
        negative_indices = torch.nonzero(negative_mask)
        random_negative_indices = torch.randperm(len(negative_indices))[:num_positive_pairs]
        selected_negative_indices = negative_indices[random_negative_indices]

        # Select corresponding positive pairs
        positive_mask.fill_diagonal_(False)
        positive_distances = distances[positive_mask].view(-1, 1)
        negative_distances = distances[selected_negative_indices[:,0],selected_negative_indices[:,1]].view(-1, 1)

    # Calculate the loss for positive and negative pairs
    loss = (positive_distances - negative_distances + margin).clamp(min=0).mean()
    return loss

def forward_HGT(data,model,mlpmodel):
    data = data.to(device)
    x,batch=data.pos, data['vertices'].batch
    data["vertices"]['x']=data.pos
    label=data.y.long().view(-1)

    optimizer.zero_grad()
    
    output=model(data.x_dict, data.edge_index_dict)  
    if args.dataset in ["dbp"]:
        graph_embeddings=global_add_pool(output,batch)
    else:
        graph_embeddings=global_add_pool(output,batch)
    graph_embeddings.clamp_(max=1e6)
    c_loss=contrastive_loss(graph_embeddings,label,margin=1)
    output=mlpmodel(graph_embeddings)
    # log_probs = F.log_softmax(output, dim=1)

    loss = criterion(output, label) 
    loss+=c_loss*args.loss_coef
    return loss,c_loss*args.loss_coef,output,label    
    
def forward(data,model,mlpmodel):
    data = data.to(device)
    edge_index1=data['vertices', 'inside', 'vertices']['edge_index']
    edge_index2=data['vertices', 'apart', 'vertices']['edge_index']
    combined_edge_index=torch.cat([data['vertices', 'inside', 'vertices']['edge_index'],data['vertices', 'apart', 'vertices']['edge_index']],1)
    num_edge_inside=edge_index1.shape[1]
    
    if args.spanning_tree == 'T':
        edge_weight=torch.rand(combined_edge_index.shape[1]) + 1
        undirected_spanning_edge = util.build_spanning_tree_edge(combined_edge_index, edge_weight,num_nodes=data.pos.shape[0])
        
        edge_set_1 = set(map(tuple, edge_index2.t().tolist()))
        edge_set_2 = set(map(tuple, undirected_spanning_edge.t().tolist()))

        common_edges = edge_set_1.intersection(edge_set_2)
        common_edges_tensor = torch.tensor(list(common_edges), dtype=torch.long).t().to(device)
        spanning_edge=torch.cat([edge_index1,common_edges_tensor],1)
        combined_edge_index=spanning_edge
    x,batch=data.pos, data['vertices'].batch
    label=data.y.long().view(-1)

    num_nodes=x.shape[0]
    edge_index_2rd, num_triplets_real, edx_jk, edx_ij = util.triplets(combined_edge_index, num_nodes)
    optimizer.zero_grad()
    input_feature=torch.zeros([x.shape[0],args.h_ch],device=device) 
    output=model(input_feature,x,[edge_index1,edge_index2], edge_index_2rd,edx_jk, edx_ij,batch,num_edge_inside,args.edge_rep)  
    output=torch.cat(output,dim=1)
    if args.dataset in ["dbp"]:
        graph_embeddings=global_add_pool(output,batch)
    else:
        graph_embeddings=global_add_pool(output,batch)
    graph_embeddings.clamp_(max=1e6)
    c_loss=contrastive_loss(graph_embeddings,label,margin=1)
    output=mlpmodel(graph_embeddings)

    loss = criterion(output, label) 
    loss+=c_loss*args.loss_coef
    return loss,c_loss*args.loss_coef,output,label
def train(train_Loader,model,mlpmodel ):
    epochloss=0
    epochcloss=0
    y_hat, y_true,y_hat_logit = [], [], []        
    optimizer.zero_grad()
    model.train()
    mlpmodel.train()
    for i,data in enumerate(train_Loader):
        if args.model=="our":
            loss,c_loss,output,label  =forward(data,model,mlpmodel)
        elif args.model in ["HGT","HAN"]:
            loss,c_loss,output,label  =forward_HGT(data,model,mlpmodel)

        loss.backward()
        optimizer.step()
        epochloss+=loss.detach().cpu()
        epochcloss+=c_loss.detach().cpu()
        
        _, pred = output.topk(1, dim=1, largest=True, sorted=True)
        pred,label,output=pred.cpu(),label.cpu(),output.cpu()
        y_hat += list(pred.detach().numpy().reshape(-1))
        y_true += list(label.detach().numpy().reshape(-1))
        y_hat_logit+=list(output.detach().numpy())
    return epochloss.item()/len(train_Loader),epochcloss.item()/len(train_Loader),y_hat, y_true,y_hat_logit

def test(loader,model,mlpmodel ):
    y_hat, y_true,y_hat_logit = [], [], []
    loss_total, pred_num = 0, 0
    model.eval()
    mlpmodel.eval()
    with torch.no_grad():
        for data in loader:
            if args.model=="our":
                loss,c_loss,output,label  =forward(data,model,mlpmodel)
            elif args.model in ["HGT","HAN"]:
                loss,c_loss,output,label  =forward_HGT(data,model,mlpmodel)
            
            _, pred = output.topk(1, dim=1, largest=True, sorted=True)
            pred,label,output=pred.cpu(),label.cpu(),output.cpu()
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(label.detach().numpy().reshape(-1))
            y_hat_logit+=list(output.detach().numpy())
            
            pred_num += len(label.reshape(-1, 1))
            loss_total += loss.detach() * len(label.reshape(-1, 1))
    return loss_total/pred_num,y_hat, y_true, y_hat_logit
def main(args,train_Loader,val_Loader,test_Loader):
    best_val_trigger = -1
    old_lr=1e3
    suffix="{}{}-{}:{}:{}".format(datetime.now().strftime("%h"),
                                    datetime.now().strftime("%d"),
                                    datetime.now().strftime("%H"),
                                    datetime.now().strftime("%M"),
                                    datetime.now().strftime("%S"))        
    if args.log: writer = SummaryWriter(os.path.join(tensorboard_dir,suffix))

    for epoch in range(args.nepoch):
        train_loss,train_closs,y_hat, y_true,y_hat_logit=train(train_Loader,model,mlpmodel )

        train_acc=util.calculate(y_hat,y_true,y_hat_logit)
        try:util.record({"loss":train_loss,"closs":train_closs,"acc":train_acc},epoch,writer,"Train") 
        except: pass
        util.print_1(epoch,'Train',{"loss":train_loss,"closs":train_closs,"acc":train_acc})
        if epoch % args.test_per_round == 0:
            val_loss, yhat_val, ytrue_val, yhatlogit_val = test(val_Loader,model,mlpmodel )
            test_loss, yhat_test, ytrue_test, yhatlogit_test = test(test_Loader,model,mlpmodel )
            val_acc=util.calculate(yhat_val,ytrue_val,yhatlogit_val)
            try:util.record({"loss":val_loss,"acc":val_acc},epoch,writer,"Val")
            except: pass
            util.print_1(epoch,'Val',{"loss":val_loss,"acc":val_acc},color=blue) 
            test_acc=util.calculate(yhat_test,ytrue_test,yhatlogit_test)
            try:util.record({"loss":test_loss,"acc":test_acc},epoch,writer,"Test")            
            except: pass
            util.print_1(epoch,'Test',{"loss":test_loss,"acc":test_acc},color=blue)
            val_trigger=val_acc
            if val_trigger > best_val_trigger:
                best_val_trigger = val_trigger
                best_model = copy.deepcopy(model)
                best_mlpmodel=copy.deepcopy(mlpmodel)
                best_info=[epoch,val_trigger]
        """ 
        update lr when epoch≥30
        """
        if epoch >= 30:
            lr = scheduler.optimizer.param_groups[0]['lr']
            if old_lr!=lr:
                print(red('lr'), epoch, (lr), sep=', ')
                old_lr=lr
            scheduler.step(val_trigger)        
    """
    use best model to get best model result 
    """
    val_loss, yhat_val, ytrue_val, yhat_logit_val  = test(val_Loader,best_model,best_mlpmodel)
    test_loss, yhat_test, ytrue_test, yhat_logit_test= test(test_Loader,best_model,best_mlpmodel)

    val_acc=util.calculate(yhat_val,ytrue_val,yhat_logit_val)
    util.print_1(best_info[0],'BestVal',{"loss":val_loss,"acc":val_acc},color=blue)
    test_acc=util.calculate(yhat_test,ytrue_test,yhat_logit_test)
    util.print_1(best_info[0],'BestTest',{"loss":test_loss,"acc":test_acc},color=blue)
    if args.model=="our":print(best_model.att)
                                                            
    """
    save training info and best result 
    """
    result_file=os.path.join(info_dir, suffix)
    with open(result_file, 'w') as f:
        print("Random Seed: ", Seed,file=f)
        print(f"acc  val : {val_acc:.3f}, Test : {test_acc:.3f}", file=f)
        print(f"Best info: {best_info}", file=f)
        for i in [[a,getattr(args, a)] for a in args.__dict__]:
            print(i,sep='\n',file=f)
    to_save_dict={'model':best_model.state_dict(),'mlpmodel':best_mlpmodel.state_dict(),'args':args,'labels':ytrue_test,'yhat':yhat_test,'yhat_logit':yhat_logit_test}
    torch.save(to_save_dict, os.path.join(model_dir,suffix+'.pth') )
    print("done")

if __name__ == '__main__':
    """
    build dir 
    """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)
    tensorboard_dir=os.path.join(args.save_dir,'log')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir,exist_ok=True)
    model_dir=os.path.join(args.save_dir,'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir,exist_ok=True)    
    info_dir=os.path.join(args.save_dir,'info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir,exist_ok=True)      

    Seed = 0
    test_ratio=0.2
    print("data splitting Random Seed: ", Seed)
    if args.dataset in ['mnist',"mnist_sparse"]:
        train_ds,val_ds,test_ds=dataset.get_mnist_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ['building']:
        train_ds,val_ds,test_ds=dataset.get_building_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ['mbuilding']:
        train_ds,val_ds,test_ds=dataset.get_mbuilding_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ['sbuilding']:
        train_ds,val_ds,test_ds=dataset.get_sbuilding_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ['smnist']:
        train_ds,val_ds,test_ds=dataset.get_smnist_dataset(args.data_dir,Seed,test_ratio=test_ratio)    
    elif args.dataset in ['dbp']:
        train_ds,val_ds,test_ds=dataset.get_dbp_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    if args.extent_norm=="T":
        train_ds= dataset.affine_transform_to_range(train_ds,target_range=(-1, 1))
        val_ds= dataset.affine_transform_to_range(val_ds,target_range=(-1, 1))
        test_ds= dataset.affine_transform_to_range(test_ds,target_range=(-1, 1))
                
    train_loader = torch_geometric.loader.DataLoader(train_ds,batch_size=args.train_batch, shuffle=False,pin_memory=True,drop_last=True) 
    val_loader = torch_geometric.loader.DataLoader(val_ds, batch_size=args.test_batch, shuffle=False, pin_memory=True)
    test_loader = torch_geometric.loader.DataLoader(test_ds,batch_size=args.test_batch, shuffle=False,pin_memory=True)
    """
    set model seed 
    """
    Seed = args.man_seed if args.manualSeed else random.randint(1, 10000)
    Seed=3407
    print("Random Seed: ", Seed)
    print(args)
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)    
    main(args,train_loader,val_loader,test_loader)
    