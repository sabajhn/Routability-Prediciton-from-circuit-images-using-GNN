import torch
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool

# Define hyperparameters
num_heads = 12 # Number of attention heads per layer
hidden_dim = 14 # Hidden dimension per head
dropout = 0.3 # Dropout ratio
merge_type = "concat" # How to merge the outputs of different heads
dnn_dim = 400 # Dimension of each dnn layer
pool_dim = 50
# Define the model class
class GATGraphModel(nn.Module):
  def __init__(self, in_channels, out_channels,able_gnn=True):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.able_gnn = able_gnn

    self.layers = nn.ModuleList()
    self.bnlayers = nn.ModuleList()
    self.bnlayers.append(nn.BatchNorm1d(in_channels))
    for i in range(7):
        if i == 0:
            self.layers.append(GATConv(in_channels,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True))
        else:
            self.layers.append(GATConv(hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True))
        self.bnlayers.append(nn.BatchNorm1d(hidden_dim * num_heads))

    self.layers.append(GATConv(hidden_dim * num_heads,
    hidden_dim,
    heads=num_heads,
    dropout=dropout,
    concat=False))
    self.bnpool = nn.BatchNorm1d(hidden_dim)

    self.ugp_layer1 = nn.Linear(hidden_dim, pool_dim)
    self.ugp_layer2 = nn.Linear(pool_dim, pool_dim)

    self.fc_bn_layers = nn.ModuleList()
    self.dnn_layers = nn.ModuleList()
    for i in range(7):
        if i == 0:
            if(self.able_gnn):
                self.dnn_layers.append(nn.Linear(45+pool_dim, dnn_dim))
            else:
                self.dnn_layers.append(nn.Linear(45, dnn_dim))

            # self.fc_bn_layers.append(nn.BatchNorm1d(48))
        else:
            self.dnn_layers.append(nn.Linear(dnn_dim, dnn_dim))
            self.dnn_layers.append(nn.Dropout(p=0.2))
            self.fc_bn_layers.append(nn.BatchNorm1d(dnn_dim))
    
    self.dnn_layers.append(nn.Linear(dnn_dim, 1))

    self.lastpart = torch.nn.Sequential(nn.Linear(dnn_dim + 45, dnn_dim), nn.ReLU()
                                        ,nn.Linear(dnn_dim, dnn_dim), nn.ReLU()
                                        ,nn.Linear(dnn_dim, dnn_dim), nn.ReLU()
                                        ,nn.Linear(dnn_dim, 1))
    self.sigmoid = torch.nn.Sigmoid()

#------------------------------------------------------

  def forward(self, x, edge_index, edge_attr, batch,gf1):
    # print("--GAT--")
    if(self.able_gnn):
        x=x.T
        for i, layer in enumerate(self.layers):
            x = self.bnlayers[i](x)
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = nn.functional.elu(x)
          
        x = self.bnpool(x)
        x = self.ugp_layer1(x)
        x = global_mean_pool(x, batch)
        x = self.ugp_layer2(x)
    
        all_features = torch.concat([x, gf1.view(1,-1)],1) 
        x=all_features
    else:
        x=gf1.view(1,-1)
        
    # print("this is shape ",x.shape)
    # print(x)
    for i, layer in enumerate(self.dnn_layers):
        x = layer(x)
        if i < len(self.dnn_layers) - 1:
            x = nn.functional.elu(x)
        # print(x)
    
    # x = torch.concat([x, gf1.view(1,-1)],1) 
    # x = self.lastpart(x)
    x = self.sigmoid(x)
    # print(x)
    return x
