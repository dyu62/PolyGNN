import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
import time
import torch.optim as optim
import scipy

from matplotlib import cm
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from torch.nn.functional import softmax

torch.autograd.set_detect_anomaly(True)
import pickle
from torch.utils.tensorboard import SummaryWriter
import dataset,util
from model_new import Smodel
import model_new


import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models
import math
import shutil
import time
from datetime import date, timedelta,datetime
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GIN,GATConv,MLP
from torch_geometric.nn.pool import global_mean_pool,global_add_pool
import csv

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
    parser.add_argument('--spanning_tree', type=str, default="F")
    
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
    parser.add_argument('--nepoch', type=int, default=201)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--manualSeed', type=str, default="False")
    parser.add_argument('--man_seed', type=int, default=12345)
    
    parser.add_argument("--targetfiles", nargs='+', type=str, default=["Dec11-14:44:32.pth","Nov13-14:30:48.pth"])
    args = parser.parse_args()
    args.log=True if args.log=="True" else False
    args.edge_rep=True if args.edge_rep=="True" else False
    args.batchnorm=True if args.batchnorm=="True" else False
    args.save_dir=os.path.join('./save/',args.dataset)
    args.manualSeed=True if args.manualSeed=="True" else False
    return args

args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion=nn.CrossEntropyLoss()

def forward_HGT(args,data,model,mlpmodel):
    data = data.to(device)
    x,batch=data.pos, data['vertices'].batch
    data["vertices"]['x']=data.pos
    label=data.y.long().view(-1)
    
    output=model(data.x_dict, data.edge_index_dict)  
    if args.dataset in ["dbp"]:
        graph_embeddings=global_add_pool(output,batch)
    else:
        graph_embeddings=global_add_pool(output,batch)
    graph_embeddings.clamp_(max=1e6)

    output=mlpmodel(graph_embeddings)
    # log_probs = F.log_softmax(output, dim=1)

    loss = criterion(output, label) 
    return loss,output,label, graph_embeddings

def forward(args,data,model,mlpmodel):
    data = data.to(device)
    edge_index1=data['vertices', 'inside', 'vertices']['edge_index']
    edge_index2=data['vertices', 'apart', 'vertices']['edge_index']
    combined_edge_index=torch.cat([data['vertices', 'inside', 'vertices']['edge_index'],data['vertices', 'apart', 'vertices']['edge_index']],1)
    
    if args.spanning_tree == 'True':
        edge_weight=torch.rand(combined_edge_index.shape[1]) + 1
        combined_edge_index = util.build_spanning_tree_edge(combined_edge_index, edge_weight,num_nodes=num_nodes,)
    
    num_edge_inside=edge_index1.shape[1]
    x,batch=data.pos, data['vertices'].batch
    label=data.y.long().view(-1)
    """ 
    triplets are not the same for graphs when training 
    """
    num_nodes=x.shape[0]
    edge_index_2rd, num_triplets_real, edx_jk, edx_ij = util.triplets(combined_edge_index, num_nodes)
    
    input_feature=torch.zeros([x.shape[0],args.h_ch],device=device) 
    output=model(input_feature,x,[edge_index1,edge_index2], edge_index_2rd,edx_jk, edx_ij,batch,num_edge_inside,args.edge_rep)   
    output=torch.cat(output,dim=1)
    graph_embeddings=global_add_pool(output,batch)
    graph_embeddings.clamp_(max=1e6)
 
    output=mlpmodel(graph_embeddings)
    # log_probs = F.log_softmax(output, dim=1)

    loss = criterion(output, label) 
    return loss,output,label,graph_embeddings
def test(args,loader,model,mlpmodel,writer,reverse_mapping ):
    y_hat, y_true,y_hat_logit = [], [], [],
    embeddings=[]

    loss_total, pred_num = 0, 0
    model.eval()
    mlpmodel.eval()
    with torch.no_grad():
        for data in loader:
            if args.model=="our":
                loss,output,label,embedding  =forward(args,data,model,mlpmodel)
            elif args.model in ["HGT","HAN"]:
                loss,output,label,embedding =forward_HGT(args,data,model,mlpmodel)            
            _, pred = output.topk(1, dim=1, largest=True, sorted=True)
            pred,label,output=pred.cpu(),label.cpu(),output.cpu()
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(label.detach().numpy().reshape(-1))
            y_hat_logit+=list(output.detach().numpy())
            embeddings.append(embedding)
            
            pred_num += len(label.reshape(-1, 1))
            loss_total += loss.detach() * len(label.reshape(-1, 1))
            
    y_true_str=[reverse_mapping(item) for item in y_true] 
    writer.add_embedding(torch.cat(embeddings,dim=0).detach().cpu(),metadata=y_true_str,tag="numbers")
    writer.close()      
    return loss_total/pred_num,y_hat, y_true, y_hat_logit  
 
def main(args,train_Loader,val_Loader,test_Loader):
    donefiles=os.listdir(os.path.join(args.save_dir,args.model,'model'))
    tensorboard_dir=os.path.join(args.save_dir,args.model,'log') 
    if args.dataset in ["mnist","mnist_sparse"]:
        reverse_mapping=lambda x: x + 10
        # list(map(lambda x: x - 10, []))
    elif args.dataset in ["building","mbuilding"]:
        reverse_mapping=lambda x: dataset.reverse_label_mapping[x]
    elif args.dataset in ["sbuilding"]:
        reverse_mapping=lambda x: dataset.single_reverse_label_mapping[x]
    elif args.dataset in ["dbp","smnist"]:
        reverse_mapping=lambda x: x
    for file in donefiles:
        if file not in args.targetfiles:
            continue
        else:
            print(file)
            saved_dict=torch.load(os.path.join(args.save_dir,args.model,'model',file))   
            if saved_dict['args'].dataset in ["mnist","mnist_sparse"]:
                x_out=90
            elif saved_dict['args'].dataset in ["building","mbuilding"]:
                x_out=100
            elif saved_dict['args'].dataset in ["sbuilding","smnist"]:
                x_out=10
            elif saved_dict['args'].dataset in ["dbp"]:
                x_out=2
            if saved_dict['args'].model=="our":
                model=Smodel(h_channel=saved_dict['args'].h_ch,input_featuresize=saved_dict['args'].h_ch,\
                                localdepth=saved_dict['args'].localdepth,num_interactions=saved_dict['args'].num_interactions,finaldepth=saved_dict['args'].finaldepth,share=saved_dict['args'].share,batchnorm=saved_dict['args'].batchnorm)
                mlpmodel=MLP(in_channels=saved_dict['args'].h_ch*saved_dict['args'].num_interactions, hidden_channels=saved_dict['args'].h_ch,out_channels=x_out, num_layers=saved_dict['args'].classifier_depth)
            elif saved_dict['args'].model=="HGT":
                model=model_new.HGT(hidden_channels=saved_dict['args'].h_ch, out_channels=saved_dict['args'].h_ch, num_heads=2, num_layers=saved_dict['args'].num_interactions)
                mlpmodel=MLP(in_channels=saved_dict['args'].h_ch, hidden_channels=saved_dict['args'].h_ch,out_channels=x_out, num_layers=saved_dict['args'].classifier_depth,dropout=saved_dict['args'].dropout)
            elif saved_dict['args'].model=="HAN":
                model=model_new.HAN(hidden_channels=saved_dict['args'].h_ch, out_channels=saved_dict['args'].h_ch, num_heads=2, num_layers=saved_dict['args'].num_interactions)
                mlpmodel=MLP(in_channels=saved_dict['args'].h_ch, hidden_channels=saved_dict['args'].h_ch,out_channels=x_out, num_layers=saved_dict['args'].classifier_depth,dropout=saved_dict['args'].dropout)
            model.to(device), mlpmodel.to(device)         
            try:
                model.load_state_dict(saved_dict['model'],strict=True)
                mlpmodel.load_state_dict(saved_dict['mlpmodel'],strict=True)
            except OSError:
                print('loadfail: ',file)
                pass
            print(saved_dict['args'])

            writer = SummaryWriter(os.path.join(tensorboard_dir,file+"_embedding"))
            test_loss, yhat_test, ytrue_test, yhatlogit_test = test(saved_dict['args'],test_Loader,model,mlpmodel,writer,reverse_mapping)
                
            pred_dir=os.path.join(tensorboard_dir,file+"_test_record")
            to_save_dict={'labels':ytrue_test,'yhat':yhat_test,'yhat_logit':yhatlogit_test}
            torch.save(to_save_dict, pred_dir)
                           
            test_acc=util.calculate(yhat_test,ytrue_test,yhatlogit_test)
            util.print_1(0,'Test', {"loss":test_loss,"acc":test_acc},color=blue) 

        
if __name__ == '__main__':
    Seed = 0
    test_ratio=0.2
    print("data splitting Random Seed: ", Seed)
    if args.dataset in ["mnist"]:
        args.data_dir='data/multi_mnist_with_index.pkl'
        train_ds,val_ds,test_ds=dataset.get_mnist_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ["mnist_sparse"]:
        args.data_dir='data/multi_mnist_sparse.pkl'  
        train_ds,val_ds,test_ds=dataset.get_mnist_dataset(args.data_dir,Seed,test_ratio=test_ratio)      
    elif args.dataset in ["building"]:
        args.data_dir='data/building_with_index.pkl'
        train_ds,val_ds,test_ds=dataset.get_mbuilding_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ["mbuilding"]:
        args.data_dir='data/mp_building.pkl'
        train_ds,val_ds,test_ds=dataset.get_building_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ["sbuilding"]:
        args.data_dir='data/single_building.pkl'
        train_ds,val_ds,test_ds=dataset.get_sbuilding_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ["smnist"]:
        args.data_dir='data/single_mnist.pkl'
        train_ds,val_ds,test_ds=dataset.get_smnist_dataset(args.data_dir,Seed,test_ratio=test_ratio)
    elif args.dataset in ['dbp']:
        args.data_dir='data/triple_building_600.pkl'
        train_ds,val_ds,test_ds=dataset.get_dbp_dataset(args.data_dir,Seed,test_ratio=test_ratio)

    if args.extent_norm=="T":
        train_ds= dataset.affine_transform_to_range(train_ds,target_range=(-1, 1))
        val_ds= dataset.affine_transform_to_range(val_ds,target_range=(-1, 1))
        test_ds= dataset.affine_transform_to_range(test_ds,target_range=(-1, 1))              
    train_loader = torch_geometric.loader.DataLoader(train_ds,batch_size=args.train_batch, shuffle=False,pin_memory=True) 
    val_loader = torch_geometric.loader.DataLoader(val_ds, batch_size=args.test_batch, shuffle=False, pin_memory=True)
    test_loader = torch_geometric.loader.DataLoader(test_ds,batch_size=args.test_batch, shuffle=False,pin_memory=True)

    Seed=random.randint(1, 10000)
    print("Random Seed: ", Seed)
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)     
    main(args,train_loader,val_loader,test_loader)