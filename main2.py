import egnn as egnn
import egnn_clean 

import data.struct_dataloader as struct_dataloader
from torch.nn import Linear

import torch
from torch import nn, optim
import argparse
import json
import utils_sys



import os 

import torch
from torch import nn
from torch.nn import functional as F
import math
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data
import torch
from torch_geometric.data import Dataset, download_url

from torch_geometric.nn import global_mean_pool


import wandb


parser = argparse.ArgumentParser(description='EGNN structure')


parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')

args = parser.parse_args()
args.cuda = True # not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
# print(args)

# mlp = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.SiLU(),
#             nn.Linear(64, 5))

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin = Linear(hidden_channels, 7)

    def forward(self, h, batch):

        # 1. Obtain node embeddings 

        # 2. Readout layer
        h = global_mean_pool(h, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin(h)
        
        return h
    

wandb.init(project="egnn-geom", name='batch-1')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize EGNN
egnn = egnn_clean.EGNN(in_node_nf=256, out_node_nf = 256, hidden_nf=128, in_edge_nf=256, device='cuda:0')



optimizer = optim.Adam(egnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
#loss_l1 = nn.L1Loss() #nn.NLLLoss() #
criterion = torch.nn.CrossEntropyLoss()





def train(h, x, y, batch, edges, edge_attr):

    optimizer.zero_grad()
    pred_h, pred_x = egnn(h, x, edges, edge_attr)


    pred_h = pred_h.to(device)

    print('pred_h shape', pred_h.shape)
    print('y shape', y.shape)

    #loss = loss_l1(pred_h, y)

    # mlp.to(device)
    mlp = MLP(256).to(device)
    pred = mlp(pred_h, batch) 

    # pred = pred.argmax(dim=1)  # Use the class with highest probability.

    print('**** final')  
    print('pred shape', pred.shape)
    print('y shape', y.shape)

    loss = criterion(pred, y)


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss, pred_h



# PyG dataloader 
batch_size = 16
root = '/users/rvinod/data/rvinod/code-prot-geometric/structures'
dataset = struct_dataloader.StructureData(root)
paths = dataset.processed_paths[:1000]
data_list = [torch.load(pt) for pt in paths]
loader = DataLoader(data_list, batch_size=batch_size)


for epoch in range(1000):

    for step, data in enumerate(loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)

        h = data.node_attr.to(device) #torch.ones(batch_size * n_nodes, n_feat)
        x = data.x.to(device) #torch.ones(batch_size * n_nodes, x_dim)
        y_adjusted = torch.full((batch_size, ), 1)
        y = (data.y - y_adjusted).to(device)

        n_nodes = x.shape[0]
        batch = data.batch.to(device)

        print('batch', batch.shape)
        print(batch)



        edges, edge_attr = egnn_clean.get_edges_batch(n_nodes, batch_size)

        edges = torch.cat([edges[0], edges[1]]).reshape(2, -1).to(device)
        edge_attr = edge_attr.to(device)

        print('*** DATA ***')
        print('x', x.shape)
        print('h', h.shape)
        print('edges', edges.shape)
        print('edge_attr', edge_attr.shape)
        print('************')


        loss, h = train(h, x, y, batch, edges, edge_attr)

        if epoch % 10 == 0:
            print('Loss: ', loss.item())
            wandb.log({'loss': loss.item()})

