import os 

import torch
from torch import nn
from torch.nn import functional as F
import math
from torch_geometric.loader import DataLoader

from torch_geometric.data import Data
import torch
from torch_geometric.data import Dataset, download_url

import data.embeddings as embeddings
import data.struct_dataloader as struct_dataloader

# import data.seq_dataloader as seq_dataloader
# import data.joint_dataloader as joint_dataloader

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from matplotlib import pyplot as plt

import wandb
wandb.init(project="simple-gcn")

# # Load using processed objects

# root = '/users/rvinod/data/rvinod/code-prot-geometric/sequences'
# dataset = seq_dataloader.SequenceData(root)

# root = '/users/rvinod/data/rvinod/code-prot-geometric/joint'
# dataset = joint_dataloader.JointData(root)

root = '/users/rvinod/data/rvinod/code-prot-geometric/structures'
dataset = struct_dataloader.StructureData(root)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(3, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 3)
        self.classifier = Linear(3, 5)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h
    
def train(data):

    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    #print('out shape', out.shape)
    #loss = criterion(out, torch.full(data.y)) # criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    i, j = out.size()[0], out.size()[1]
    if data.y[0] == 6:
        y_ = torch.full((i, ), 4)
    else:
        y_ = torch.full((i, ), (data.y[0] - 1))
    y_ = y_.to(device)

    loss = criterion(out, y_)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


model = GCN()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


paths = dataset.processed_paths[:200]
data_list = [torch.load(pt) for pt in paths]

loader = DataLoader(data_list, batch_size=128)


losses = []
steps = 0

for epoch in range(8):
    for step, data in enumerate(loader):
        data = data.to(device)
        # print(data)
        # print(f'Step {step + 1}:')
        # print('=======')
        # print(f'Number of graphs in the current batch: {data.num_graphs}')
        loss, h = train(data)
        # _, h = model(data.x, data.edge_index)
        # break
        losses.append(loss.item())
        wandb.log({'loss': loss.item()})
        steps+=1
        if epoch % 10 == 0:
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print('Loss: ', loss.item())
            
steps = range(steps)

plt.plot(steps, losses, linewidth=0.5)
plt.savefig('dummy.png')
