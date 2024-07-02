import os
import re
import json
import numpy as np
import pandas as pd

import torch
import random
import pickle as pkl
from tqdm import tqdm
from torch import Tensor
from scipy.spatial import distance_matrix
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
# from shapely.geometry import Point, Polygon


def cross_product(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

def colinear(p1, p2, p3):
  if (p1[1]-p2[1])*(p2[0]-p3[0]) == (p1[0]-p2[0])*(p2[1]-p3[1]) and p3[0]>min(p1[0],p2[0]) and p3[0]<max(p1[0],p2[0]): return True
  if (p1[1]-p2[1])*(p2[0]-p3[0]) == (p1[0]-p2[0])*(p2[1]-p3[1]) and p3[1]>min(p1[1],p2[1]) and p3[1]<max(p1[1],p2[1]): return True

def is_intersected(p1, p2, p3, p4):
  if colinear(p1, p2, p3) or colinear(p1, p2, p4): return True
  if cross_product(p1, p2, p3) * cross_product(p1, p2, p4) < 0 and cross_product(p3, p4, p1) * cross_product(p3, p4, p2) < 0:  return True
  else: return False

#Pos of single number
def read_singe(df, i):
  p_i = df[0][i]
  np_i = list(p_i)
  rflag = 0
  for x in range(len(p_i)-4):
    if p_i[x] == ')': rflag+=1
    if p_i[x:x+4] == '), (' and p_i[x-1]!=')': np_i.insert(x+rflag+1,' 0.0 0.0')
    elif p_i[x:x+4] == '), (' and p_i[x-1]==')': np_i.insert(x+rflag+1,' 1.0 1.0')
  p_i = ''.join(np_i)


  pos = np.empty((1,2))
  pi_nums = re.findall(r"\d+\.?\d*",p_i)
  j=0
  while j < len(pi_nums)-1:
    if j == 0:
      pos[0][0] = float(pi_nums[j])
      pos[0][1] = float(pi_nums[j+1])
      j+=2
      continue
    pos = np.append(pos,[[float(pi_nums[j]),0]],0)
    pos[j//2][1] = float(pi_nums[j+1])
    j+=2
  return pos

def Visi_Edge(pos_join, flag):
  inside_edge_index = [[],[]]
  apart_edge_index = [[],[]]

  vg_point = []
  for i in range(len(pos_join)): vg_point.append((pos_join[i][0], pos_join[i][1]))

  hole_p = np.where(flag==1)[0]
  if len(hole_p) != 0:
    last_id = 0
    for m in range(len(flag)):
      if flag[m] == 2 or flag[m] == 3:
        if sum(flag[last_id:m]) == 0:
          last_id = m+1
          continue
        poly_i = vg_point[last_id:m+1]
        pos_i = np.arange(last_id, m+1)
        last_id = m+1
        for i in range(len(poly_i)):
          if flag[pos_i[i]] == 1:
            for j in range(i, len(flag)):
              if flag[j]==1 or flag[j] == 2 or flag[j] == 3:
                hole_i = poly_i[i:j+1]
                pos_hole = np.arange(i, j+1)
                for p1 in hole_i:
                  for p2 in poly_i:
                    if p2 not in hole_i:
                      inter_count = 0
                      for d in range(len(poly_i)-1):
                        p3, p4 = poly_i[d], poly_i[d+1]
                        if is_intersected(p1, p2, p3, p4): inter_count+=1
                      if inter_count==0:
                        head, tail = pos_i[poly_i.index(p1)], pos_i[poly_i.index(p2)]
                        inside_edge_index[0].append(head), inside_edge_index[1].append(tail)

  for i in range(len(vg_point)):
    p1 = vg_point[i]
    p1_id = np.count_nonzero(flag[0:i] == 2) + np.count_nonzero(flag[0:i] == 3)
    for j in range(len(vg_point)):
      p2 = vg_point[j]
      if p1 == p2: continue
      p2_id = np.count_nonzero(flag[0:j] == 2) + np.count_nonzero(flag[0:j] == 3)
      inter_count = 0
      for m in range(len(flag-1)):
        if flag[m]!=1 and flag[m]!=2 and flag[m]!=3: p3, p4 = vg_point[m], vg_point[m+1]
        if is_intersected(p1, p2, p3, p4): inter_count+=1
      if inter_count==0:
        head, tail = vg_point.index(p1), vg_point.index(p2)
        cc = np.count_nonzero(flag[min(head, tail):max(head, tail)] == 2) + np.count_nonzero(flag[min(head, tail): max(head, tail)] == 3)
        if p1_id!=p2_id and cc!=0: apart_edge_index[0].append(head), apart_edge_index[1].append(tail)
    #print(i)

  ninside_edge_index = [[],[]]
  napart_edge_index = [[],[]]
  exteriors = [[],[]]

  if len(hole_p)!=0:
    for i in range(len(flag)):
      link_i = [pos_join[inside_edge_index[1][j]] for j in range(len(inside_edge_index[1])) if inside_edge_index[0][j]==i]
      if len(link_i)==0: continue
      ninside_edge_index[0].append(i)
      dis_matrix = distance_matrix([pos_join[i]], link_i)
      node_i = (link_i[np.argmin(dis_matrix[0])][0], link_i[np.argmin(dis_matrix[0])][1])
      ninside_edge_index[1].append(vg_point.index(node_i))

  for i in range(len(vg_point)-1):
    if flag[i]!=1 and flag[i]!=2 and flag[i]!=3 : exteriors[0].append(i), exteriors[1].append(i+1)

  for i in range(len(flag)):
    link_i = [pos_join[apart_edge_index[1][j]] for j in range(len(apart_edge_index[1])) if apart_edge_index[0][j]==i]
    if len(link_i)==0: continue
    napart_edge_index[0].append(i)
    dis_matrix = distance_matrix([pos_join[i]], link_i)
    node_i = (link_i[np.argmin(dis_matrix[0])][0], link_i[np.argmin(dis_matrix[0])][1])
    napart_edge_index[1].append(vg_point.index(node_i))

  inside_edge_index, apart_edge_index = ninside_edge_index, napart_edge_index

  return inside_edge_index, apart_edge_index, exteriors


def HeteroEdge(pos,k):
  pos_join = np.delete(pos, np.where(np.sum(pos, 1)==0)[0], axis=0)
  pos_join = np.delete(pos_join, np.where(np.sum(pos_join, 1)==2)[0], axis=0)
  pos_join = np.delete(pos_join, np.where(np.sum(pos_join, 1)==4)[0], axis=0)
  flag = np.zeros(len(pos_join))
  pos = np.delete(pos, 0, axis=0)
  count, id = 0, 0
  while count<k:
    for i in range(len(pos)):
      if pos[i][0]==0:
        flag[i-1]=1
        pos = np.delete(pos, i, axis=0)
        break
      elif pos[i][0]==1:
        flag[i-1]=2
        pos = np.delete(pos, i, axis=0)
        break
      elif pos[i][0]==2:
        flag[i-1]=3
        pos = np.delete(pos, i, axis=0)
        pos_join[id:i, 0]+=count
        count+=1
        id = i
        break
  pos_join = pos_join
  inside_edge_index, apart_edge_index, exteriors = Visi_Edge(pos_join, flag)

  return pos_join, inside_edge_index, apart_edge_index, exteriors

#build heterovg of k-digit from MNIST
def NNIST_HeteroVG(df, label_df, k):
  pos = [[0,0]]
  label = ''
  for i in np.random.randint(0, len(df), k):
    while True:
      if len(pos) == 1 and label_df[0][i] == 0: i = random.randint(0, len(df))
      else: break
    pos = np.append(pos, read_singe(df, i), 0)
    pos = np.append(pos, [[2,2]], 0)
    label = label+'%d'%(label_df[0][i])

  label = int(label)
  pos_join, inside, apart, exteriors = HeteroEdge(pos,k)

  data = HeteroData()

  data['vertices'].x = torch.zeros((len(pos_join), 1), dtype=torch.float)
  data.y = torch.tensor(label, dtype=torch.int)
  data.pos = torch.tensor(pos_join, dtype=torch.float)

  data['vertices', 'inside', 'vertices'].edge_index = torch.tensor([inside[0]+inside[1]+exteriors[0],inside[1]+inside[0]+exteriors[1]], dtype=torch.long)
  data['vertices', 'apart', 'vertices'].edge_index = torch.tensor([apart[0]+apart[1],apart[1]+apart[0]], dtype=torch.long)
  data['vertices', 'inside', 'vertices'].edge_attr = torch.zeros((len(data['vertices', 'inside', 'vertices'].edge_index[0]),1), dtype=torch.float)
  data['vertices', 'apart', 'vertices'].edge_attr = torch.zeros((len(data['vertices', 'apart', 'vertices'].edge_index[0]),1), dtype=torch.float)

  return data

mnist_filename = '/content/drive/MyDrive/MINST_Polygons/polyMNIST/mnist_polygon_test.json'
label_filename = '/content/drive/MyDrive/MINST_Polygons/polyMNIST/mnist_label_test.json'
df = pd.read_json(mnist_filename)
label_df = pd.read_json(label_filename)

K = 2 # number of digits
N = 10 # number of generated graphs
multi_mnist_dataset = []
for k in range(2, K+1):
  for i in tqdm(range(N)):
    data = NNIST_HeteroVG(df, label_df, k=k)
    multi_mnist_dataset.append(data)

if not os.path.exists('/content/drive/MyDrive/MINST_Polygons/multi_mnist'):
    os.makedirs('/content/drive/MyDrive/MINST_Polygons/multi_mnist')
with open('/content/drive/MyDrive/MINST_Polygons/multi_mnist/multi_mnist.pkl','wb') as file:
    pkl.dump(multi_mnist_dataset, file)
