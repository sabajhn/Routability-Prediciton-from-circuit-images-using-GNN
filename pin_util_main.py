import random
import torch
import time
import glob, os
import numpy as np
import torch.nn.functional as F
import random
import time
from torch_geometric.data import Data as dataa
import torch
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from train.sklearn_train import do_sklearn_train

from preprocess_atom.features import get_gdata, get_features1, get_routability, edge_ex, node_features,finout,vinout_blks,finout_blks, add_metis, totuple, normalize_gf, fine_grained_normalizer, vpr_txt
from preprocess_atom.process_features import process_features, load_process_features
from model.gat import GATGraphModel
from train.train import test1,train1
from analysis.performance_results import plot_results

def perf_value(x):
    if(len(x) == 13):
        return x[1]    
    assert(len(x) == 29)
    xx = x[1] * x[2] * x[3] * x[4] * x[5] * x[21] * x[27] 
    return xx

def label_compare(x, y):
    if (len(x)> len(y)):
        return 0
    if (len(x)< len(y)):
        return 1
    if(len(x) == 13):
        if (perf_value(x) > perf_value(y)):
            return 1
        return 0
    assert(len(x) == 29)
    if (perf_value(x) > perf_value(y)):
       return 1
    return 0
       


my_seed = 123456
torch.manual_seed(my_seed)
random.seed(my_seed)
np.random.seed(my_seed)

# file_address = "/home/mani/Desktop/hyperGNN/drive"
# file_address = "/home/mani/Desktop/hyperGNN/new100dataset"
file_address = "/home/mani/Desktop/hyperGNN/DATASET"



# Open the file with the data
n = 1744
n = 1000
# print("saving features to files ...")
# process_features(file_address, n)
print("loading features from files ...")
loaded_graph_features, loadd_edges, loadd_edge_f, loadd_node_f, loaded_labels = load_process_features(n)
print("----------------------------------------------------------------------------")


blks_size=[]
circuit_names=[]
routability=[]  
nodes_blktype=[]
  
vprr=[]
  
print("GRAPH FEATURES AND LABELS")
for i in range(n):
  print(i)
  circuit = get_gdata(file_address+"/features (%d)/graph_features.txt"%i)
  a,b,c,d= get_features1(circuit)
  blks_size.append(d)
  circuit_names.append(b)
  routability.append(get_routability(file_address+"/features (%d)/routability.txt"%i))

# print(len(loaded_graph_features), len(loadd_edges), len(loadd_edge_f), len(loadd_node_f), len(loaded_labels))
if(len(routability) != len(loaded_graph_features) or len(circuit_names) != len(loaded_graph_features)):
  print("STH is WRONG")
  exit()


node_ff, edge_f = fine_grained_normalizer(loadd_node_f, loadd_edge_f)


dataset = []
best_dict = {}
for i in range(len(routability)):
  key = [circuit_names[i]] + loaded_graph_features[i][40:45]
  best_dict[tuple(key)] = i

for i in range(len(routability)):
  key = [circuit_names[i]] + loaded_graph_features[i][40:45]
  bid = best_dict[tuple(key)]
  z = label_compare(loaded_labels[bid],loaded_labels[i])
  if(z == 1):
     best_dict[tuple(key)] = i

for i in range(len(routability)):
  key = [circuit_names[i]] + loaded_graph_features[i][40:45]
  my_nodes = node_ff[i]
  my_edges = torch.tensor(loadd_edges[i], dtype=torch.int64).T
  my_edge_attrs = edge_f[i]
  my_lables = torch.tensor([loaded_labels[i]])
  # print(my_nodes.shape, my_edges.shape, my_edge_attrs.shape, my_lables.shape)
  my_data = dataa(x=my_nodes, edge_index=my_edges,edge_attr=my_edge_attrs,y=my_lables)
  bid = best_dict[tuple(key)]
  if i == bid:
    print(i,"name: ",circuit_names[i],loaded_graph_features[i][40:46], ":" , perf_value(loaded_labels[i]))
    print(my_data)
    dataset.append(my_data)
batch = dataset

GAT = GATGraphModel(16, 1,able_gnn= False)
