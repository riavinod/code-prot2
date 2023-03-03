import egnn as egnn
import egnn_clean 

import data.struct_dataloader as struct_dataloader

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
print(args)


wandb.init(project="dummy-egnn", name='dummy-vars')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


utils_sys.makedir(args.outf)
# utils_sys.makedir(args.outf + "/" + args.exp_name)


# model = egnn.EGNN(in_node_nf=256, in_edge_nf=256, hidden_nf=args.nf, device=device, n_layers=4, coords_weight=1.0)


# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
# loss_l1 = nn.L1Loss()



############ this code works

# Dummy parameters
batch_size = 16
n_nodes = 100
n_feat = 256
x_dim = 3

# Dummy variables h, x and fully connected edges
h = torch.ones(batch_size * n_nodes, n_feat).to(device)
x = torch.ones(batch_size * n_nodes, x_dim).to(device)
y = torch.randint(0, 5, (batch_size * n_nodes, 1)).to(device)
edges, edge_attr = egnn_clean.get_edges_batch(n_nodes, batch_size)

edges = torch.cat([edges[0], edges[1]]).reshape(2, -1).to(device)
edge_attr = edge_attr.to(device)


# print('Shape of x: ', x.shape)
# print('Shape of node attrs: ', h.shape)
# print('Shape of edges: ', edges.shape)
# print('Shape of edge attrs: ', edge_attr.shape)



# Initialize EGNN
egnn = egnn_clean.EGNN(in_node_nf=n_feat, out_node_nf = 256, hidden_nf=32, in_edge_nf=256, device='cuda:0')



optimizer = optim.Adam(egnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss() #nn.NLLLoss() #


mlp = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1))


def train(h, x, edges, edge_attr):

    optimizer.zero_grad()
    pred_h, pred_x = egnn(h, x, edges, edge_attr)


    pred_h = pred_h.to(device)
    mlp.to(device)
    pred = mlp(pred_h)   

    loss = loss_l1(pred, y)


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss, pred_h



for epoch in range(10000):
        

    loss, h = train(h, x, edges, edge_attr)

    x = x.detach()
    h = h.detach()
    edges = edges.detach()
    edge_attr = edge_attr.detach()
            
    if epoch % 10 == 0:
        print('Loss: ', loss.item())
        wandb.log({'loss': loss.item()})







# ##############

# # PyG dataloader 
# root = '/users/rvinod/data/rvinod/code-prot-geometric/structures'
# dataset = struct_dataloader.StructureData(root)
# paths = dataset.processed_paths[:200]
# data_list = [torch.load(pt) for pt in paths]
# loader = DataLoader(data_list, batch_size=1)


# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)

#     h = data.node_attr #torch.ones(batch_size * n_nodes, n_feat)
#     x = data.x #torch.ones(batch_size * n_nodes, x_dim)
#     # edges = data.edge_index
#     # edge_attr = data.edge_attr

#     edges, edge_attr = egnn_clean.get_edges_batch(n_nodes, batch_size)


#     print('Shape of x', x.shape)
#     print('Shape of h', h.shape)
#     print('Shape of edges: ', edges.shape)
#     print('Shape of edge attrs: ', edge_attr.shape)

#     # print('edges')
#     # print(edges)
#     # print(edges[0])
#     # print(edges[1])

    

#     break
