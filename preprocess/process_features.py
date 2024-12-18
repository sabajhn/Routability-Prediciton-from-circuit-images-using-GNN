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

from preprocess.features import get_gdata, get_features1, get_routability, edge_ex, node_features,finout,vinout_blks,finout_blks, add_metis, totuple, normalize_gf, fine_grained_normalizer, vpr_txt, get_perfs, get_times

def process_features(file_address, n):
    graph_features=[]
    blks_size=[]
    edge_size=[]
    circuit_names=[]
    routability=[]  
    edges=[]  
    nodes_blktype=[]

    labels = []
    
    vprr=[]
    for i in range(n):
        print(i)
        vprr.append(vpr_txt(file_address+"/features (%d)/vpr.txt"%i))
        circuit = get_gdata(file_address+"/features (%d)/graph_features.txt"%i)
        # routability.append(get_routability("/content/gdrive/MyDrive/features (2)/featuress/features (%d)/features/routaibility.txt"%i))
        a,b,c,d= get_features1(circuit)
        # print(get_features1(dt))
        # print(len(a))
        graph_features.append(a)
        blks_size.append(d)
        edge_size.append(c)
        perfs = get_perfs(file_address+"/features (%d)/labels.txt"%i)
        times = get_times(file_address+"/features (%d)/time.txt"%i)
        circuit_names.append(b)
        routability.append(get_routability(file_address+"/features (%d)/routability.txt"%i))
        labels.append(perfs + times + [1-routability[i][0]])
    
    graph_features = np.asarray(graph_features)
    vprr = np.asarray(vprr)
    graph_features = np.concatenate((graph_features,vprr), axis=1)
    
    # print(graph_features)
    for i in range(graph_features.shape[0]):
        f = open("processed_features/graph_features/gf"+str(i)+".txt", "w")
        for j in range(graph_features.shape[1]):
            f.write(str(float(graph_features[i][j])))
            f.write("\n")
        f.close()
    

    print("EDGES")
    edge2=[]
    for i in range(0,n):
        if(i%100 == 0):
            print(i)
        edge2.append([])
        ee = edge_ex(file_address+"/features (%d)/edges.txt"%i)
        ee.sort()
        ee2 = []
        ee3 = []
        for j in range(len(ee)):
            a = ee[j][0]
            b = ee[j][1]
            if(j == 0 or ee2[-1] != [a,b]):
                ee2.append([a,b])
                edge2[i].append(1)
            else:
                edge2[i][-1] += 1
        edges.append(ee2)

    # print(edges)
    for i in range(len(edges)):
        f = open("processed_features/edges/edge"+str(i)+".txt", "w")
        for j in range(len(edges[i])):
            f.write(str(edges[i][j][0]) + ","+ str(edges[i][j][1])+"\n")
        f.close()
    # print(len(edges), len(edges[0]))

    print("NODE FEATUES")
    for i in range(0,n):
        nodes_blktype.append(node_features(file_address+"/features (%d)/node_f-blktype.txt"%i))

    fin=[]
    fout=[]
    for i in range(0,n):
        a,b=finout(file_address+"/features (%d)/nodef-finfout.txt"%i)
        fin.append(a)
        fout.append(b)
    # print(len(fin), len(fin[0]))
    vin_blks , vout_blks, vin_blks2, vout_blks2 = vinout_blks(edges, blks_size, fin, fout)
    # print(len(vin_blks),len(edges))

    in_blks , out_blks, in_blks2 , out_blks2 = finout_blks(edges, blks_size, fin, fout)
    # print(len(in_blks),len(edges))

    node_ff=[]
    for i in range(len(edges)):
        node_ff.append(torch.concat((torch.tensor(nodes_blktype[i]).view(-1,6), torch.tensor(fin[i]).view(-1,1),
                                torch.tensor(fout[i]).view(-1,1), torch.tensor(in_blks[i]).view(-1,1),torch.tensor(out_blks[i]).view(-1,1), 
                                torch.tensor(in_blks2[i]).view(-1,1) ,torch.tensor(out_blks2[i]).view(-1,1), torch.tensor(vin_blks[i]).view(-1,1) 
                                ,torch.tensor(vout_blks[i]).view(-1,1),torch.tensor(vin_blks2[i]).view(-1,1) ,torch.tensor(vout_blks2[i]).view(-1,1)),dim=1))

    for i in range(len(node_ff)):
        f = open("processed_features/node_ff/nodes_f"+str(i)+".txt", "w")
        for j in range(len(node_ff[i])):
            for k in range(len(node_ff[i][j])):
                f.write(str(float(node_ff[i][j][k])) + ",")
            f.write("\n")
        f.close()

    print("METIS")
    edge_f=[]
    metis_times=[]
    print("-------------------")
    for i in range(len(edges)):
        print(i, len(edges[i]))
        t1 = time.time()
        f1=add_metis(list(totuple(edges[i])), 2)
        # f1= [[0] for i in range(len(edges[i]))]
        t2 = time.time()
        # print(i,t2-t1)
        f2=add_metis(list(totuple(edges[i])), 5)
        t3 = time.time()
        # print(i,t3-t2)
        # f3=add_metis(list(totuple(edges[i])), 10)
        t4 = time.time()
        # print(i,t4-t3)
        # f4=add_metis(list(totuple(edges[i])), 50)
        # t5 = time.time()
        # print(i,t5-t4)
        # f5=add_metis(list(totuple(edges[i])), 100)
        # t6 = time.time()
        # print(i,t6-t5)
        # f6=add_metis(list(totuple(edges[i])), 1000)
        # t7 = time.time()
        # print(i,t7-t6)
        # f7=add_metis(list(totuple(edges[i])), int(blks_size[i]/5))
        # t8 = time.time()
        # print(i,t8-t7)
        # f8=add_metis(list(totuple(edges[i])), int(blks_size[i]/5))
        t9 = time.time()
        metis_times.append(t9-t1)
        # print(i,i,t9-t1)
        # f9=add_metis(list(totuple(edges[i])), int(len(nodes)/100))

        # edge_f.append(torch.stack((torch.tensor(f1),torch.tensor(f2),torch.tensor(f3)),dim=0))
        edge_f.append(torch.concat((torch.tensor(edge2[i]).view(-1,1),torch.tensor(f1),torch.tensor(f2)),dim=1))
        # print("edge_f",len(edges[i]), edge_f[i].shape, "  ",torch.tensor(edge2[i]).view(-1,1).shape,torch.tensor(f1).shape,torch.tensor(f2).shape)

    for i in range(len(edge_f)):
        f = open("processed_features/edge_f/edge_f"+str(i)+".txt", "w")
        for j in range(len(edge_f[i])):
            for k in range(len(edge_f[i][j])):
                f.write(str(float(edge_f[i][j][k])) + ",")
            f.write("\n")
        f.close()

    for i in range(len(labels)):
        f = open("processed_features/labels/label"+str(i)+".txt", "w")
        for j in range(len(labels[i])):
            f.write(str(float(labels[i][j])) + ",")
        f.close()

def load_process_features(n):
    loaded_graph_features = []
    for i in range(n):
        f = open("processed_features/graph_features/gf"+str(i)+".txt", "r")
        loaded_graph_features.append([])
        for line in f:
            x = float(line)
            loaded_graph_features[i].append(x)
        f.close
    # print(loaded_graph_features)

    loadd_edges = []
    for i in range(n):
        f = open("processed_features/edges/edge"+str(i)+".txt", "r")
        loadd_edges1 = []
        for line in f:
            values = line.split(",")
            ss = []
            for value in values:
                ss.append(int(value))
            loadd_edges1.append(ss)
        f.close()
        loadd_edges.append(loadd_edges1)
    # print(loadd_edges)

    loadd_edge_f = []
    for i in range(n):
        f = open("processed_features/edge_f/edge_f"+str(i)+".txt", "r")
        loadd_edges_f1 = []
        for line in f:
            values = line.split(",")
            ss = []
            for i in range(len(values) - 1):
                value = values[i]
                ss.append(float(value))
            loadd_edges_f1.append(ss)
        f.close()
        loadd_edge_f.append(loadd_edges_f1)
    # print(loadd_edge_f)

    loadd_node_f = []
    for i in range(n):
        f = open("processed_features/node_ff/nodes_f"+str(i)+".txt", "r")
        loadd_node_f1 = []
        for line in f:
            values = line.split(",")
            ss = []
            for i in range(len(values) - 1):
                value = values[i]
                ss.append(float(value))
            loadd_node_f1.append(ss)
        f.close()
        loadd_node_f.append(loadd_node_f1)
    # print(loadd_node_f)

    loaded_labels = []
    for i in range(n):
        f = open("processed_features/labels/label"+str(i)+".txt", "r")
        for line in f:
            values = line.split(",")
            ss = []
            for i in range(len(values) - 1):
                value = values[i]
                ss.append(float(value))
            loaded_labels.append(ss)
        f.close()
    print(loaded_labels)

    return loaded_graph_features, loadd_edges, loadd_edge_f, loadd_node_f, loaded_labels
