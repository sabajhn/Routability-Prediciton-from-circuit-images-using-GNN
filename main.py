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

from preprocess.features import get_gdata, get_features1, get_routability, edge_ex, node_features,finout,vinout_blks,finout_blks, add_metis, totuple, normalize_gf, fine_grained_normalizer, vpr_txt
from preprocess.process_features import process_features, load_process_features
from model.gat import GATGraphModel
from train.train import test1,train1
from analysis.performance_results import plot_results

my_seed = 123456
torch.manual_seed(my_seed)
random.seed(my_seed)
np.random.seed(my_seed)

# file_address = "/home/mani/Desktop/hyperGNN/drive"
# file_address = "/home/mani/Desktop/hyperGNN/new100dataset"
# file_address = "/home/mani/Desktop/hyperGNN/ManiResault"
file_address = "/home/saba/DL/SmartVPR-master/pack-res/dataset/blif-res/stratixiv_arch.timing.xml"



# Open the file with the data
n = 530
# print("saving features to files ...")
# process_features(file_address, n)
print("loading features from files ...")
circuits = ["alu4.blif", "clock_aliases.blif",  "elliptic.blif",  "pdc.blif", "tseng.blif","apex4.blif", "des.blif", "ex1010.blif", "s38417.blif","bigkey.blif",  "diffeq.blif", "frisc.blif", "s38584.1.blif","clma.blif", "dsip.blif", "misex3.blif","seq.blif"]

loaded_graph_features, loadd_edges, loadd_edge_f, loadd_node_f = process_features(file_address, circuits)
#  load_process_features(n)
print("----------------------------------------------------------------------------")


blks_size=[]
circuit_names=[]
routability=[]  
nodes_blktype=[]
  
vprr=[]
print("GRAPH FEATURES AND LABELS")
for i in circuits:
  print(i)
  circuit = get_gdata(file_address + i +"/features/graph_features.txt")
  a,b,c,d= get_features1(circuit)
  blks_size.append(d)
  circuit_names.append(b)
  routability.append(get_routability(file_address + i +"/features/routability.txt"))

if(len(routability) != len(loaded_graph_features) or len(circuit_names) != len(loaded_graph_features)):
  print("STH is WRONG")
  exit()


node_ff, edge_f = fine_grained_normalizer(loadd_node_f, loadd_edge_f)


dataset = []
for i in range(len(routability)):
  my_nodes = node_ff[i]
  my_edges = torch.tensor(loadd_edges[i], dtype=torch.int64).T
  my_edge_attrs = edge_f[i]
  my_lables = torch.tensor([routability[i]])
  # print(my_nodes.shape, my_edges.shape, my_edge_attrs.shape, my_lables.shape)
  my_data = dataa(x=my_nodes, edge_index=my_edges,edge_attr=my_edge_attrs,y=my_lables)
  print(my_data)
  dataset.append(my_data)
batch = dataset

GAT = GATGraphModel(16, 1,able_gnn= True)


# t1=time.now()
torch.manual_seed(my_seed)
random.seed(my_seed)
np.random.seed(my_seed)

st_indices = list(set(circuit_names))
sz = len(dataset)
st_sz = len(st_indices)
random.shuffle(st_indices)
test_indices = []
train_indices = []
for i in range(sz):
  # if(graph_features[i][0] == 100):
  #   continue
  if(circuit_names[i] in st_indices[:int(st_sz/3)]):
    test_indices.append(i)
  else:
    train_indices.append(i)


print("NORMALIZE")
loaded_graph_features = torch.tensor(loaded_graph_features)
gf = normalize_gf(loaded_graph_features).float()
# print(gf[:,0])
# print(loaded_graph_features[:,0])
# exit()
# for i in range(len(indices)):
#   print(circuit_names[i],graph_features[i][0],routability[i][0],blks_size[i])
#   print(gf[i])
#   print(graph_features[i])
# print(gf)

optimizer = torch.optim.Adam(GAT.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)
# LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
# test_acc = test1(test_indices, batch, gf)

# print(test1(test_indices, batch, gf))
# print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f},  Time: {running_secs}')
# exit()
mx_train = 0
mx_test = 0
trainlist = []
testlist = []

train_indices =  [362, 451, 516, 64, 253, 313, 133, 412, 102, 497, 379, 414, 427, 487, 38, 158, 444, 223, 436, 36, 262, 83, 183, 207, 374, 120, 90, 28, 103, 70, 257, 499, 178, 477, 53, 13, 388, 463, 492, 0, 269, 240, 132, 170, 466, 486, 329, 5, 57, 15, 123, 439, 509, 218, 225, 493, 195, 403, 88, 274, 434, 78, 264, 87, 228, 127, 420, 109, 85, 246, 502, 469, 201, 459, 454, 449, 156, 72, 359, 324, 494, 336, 37, 287, 110, 462, 10, 484, 190, 205, 284, 65, 481, 256, 148, 112, 66, 394, 508, 343, 334, 337, 517, 317, 289, 361, 163, 244, 105, 162, 418, 510, 238, 255, 347, 279, 199, 60, 79, 193, 202, 309, 364, 294, 252, 424, 402, 135, 261, 519, 25, 98, 192, 397, 157, 22, 312, 169, 472, 175, 214, 20, 200, 367, 512, 4, 198, 411, 267, 30, 19, 155, 450, 184, 229, 128, 108, 404, 447, 369, 495, 285, 3, 291, 406, 474, 316, 399, 276, 501, 372, 143, 203, 527, 124, 352, 421, 432, 172, 100, 366, 97, 51, 268, 297, 95, 314, 496, 306, 526, 344, 298, 248, 233, 327, 479, 375, 130, 429, 117, 524, 43, 241, 452, 213, 40, 331, 75, 405, 55, 7, 307, 448, 339, 381, 12, 185, 322, 321, 504, 34, 165, 391, 153, 113, 409, 525, 80, 81, 441, 342, 357, 272, 220, 216, 188, 360, 351, 259, 254, 392, 27, 160, 270, 94, 514, 282, 277, 186, 332, 489, 35, 145, 147, 58, 208, 358, 177, 286, 230, 8, 523, 125, 50, 68, 187, 345, 457, 319, 354, 67, 346, 456, 396, 249, 141, 417, 52, 232, 426, 387, 242, 465, 407, 433, 376, 210, 237, 464, 384, 467, 301, 126, 522, 231, 245, 18, 42, 271, 93, 482, 304, 299, 442, 292, 168, 507, 33, 437, 48, 419, 96, 150, 45, 73, 140, 471, 283, 215, 235, 217, 138, 154, 139, 222, 315, 511, 435, 115, 82, 422, 118, 373, 302, 180, 382, 23, 377, 349, 529, 142, 389, 478, 243, 63, 6, 328, 173, 49]
test_indices =  [14, 144, 134, 275, 171, 189, 460, 260, 401, 258, 320, 265, 338, 430, 386, 197, 266, 119, 71, 483, 528, 149, 219, 76, 26, 408, 146, 101, 56, 335, 470, 350, 453, 278, 485, 29, 515, 488, 333, 131, 355, 348, 293, 280, 330, 236, 356, 181, 47, 518, 393, 179, 385, 174, 166, 11, 326, 520, 69, 416, 61, 353, 305, 2, 106, 91, 475, 415, 443, 1, 461, 122, 370, 176, 378, 300, 380, 383, 41, 44, 500, 395, 273, 506, 263, 247, 323, 423, 288, 151, 428, 438, 234, 226, 325, 398, 311, 62, 521, 129, 16, 458, 239, 425, 77, 390, 505, 21, 84, 440, 480, 498, 86, 340, 212, 89, 152, 137, 111, 290, 196, 296, 17, 54, 9, 227, 371, 431, 491, 59, 473, 182, 159, 341, 167, 513, 251, 116, 455, 121, 368, 31, 46, 365, 136, 74, 92, 114, 303, 206, 32, 318, 468, 99, 445, 104, 107, 24, 413, 221, 310, 490, 281, 211, 503, 164, 250, 308, 363, 191, 446, 161, 224, 209, 476, 295, 194, 400, 410, 39, 204]

do_sklearn_train(gf, routability, train_indices, test_indices)
# exit()

for epoch in range(1, 100):
    print("EPOCH ",epoch)
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    start = time.time()
    train1(train_indices, batch, gf,GAT,criterion,optimizer,scheduler)
    train_acc = test1(train_indices, batch, gf,GAT,batch, loaded_graph_features)
    test_acc = test1(test_indices, batch, gf,GAT,batch, loaded_graph_features)
    mx_train = max(mx_train,train_acc[0])
    mx_test = max(mx_test,test_acc[0])
    running_secs = time.time() - start
    print("Train",train_acc)
    print("Test",test_acc)
    print("maxxx ", mx_train,mx_test)

    trainlist.append(train_acc)
    testlist.append(test_acc)
    print("trainlist = ",trainlist)
    print("testlist = ",testlist)
    print("train_indices = ",train_indices)
    print("test_indices = ",test_indices)
    # print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f},  Time: {running_secs}')
    # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f} , Test Acc: {test_acc:.4f},  Time: {running_secs}')
plot_results(trainlist, testlist)