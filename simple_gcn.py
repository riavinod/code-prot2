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

wandb.init(project="simple-gcn", name='NLL loss')

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
criterion = torch.nn.NLLLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


paths = dataset.processed_paths[:2000]
data_list = [torch.load(pt) for pt in paths]

loader = DataLoader(data_list, batch_size=256)

for epoch in range(4000):
    for step, data in enumerate(loader):
        data = data.to(device)
        loss, h = train(data)
        
        wandb.log({'loss': loss.item()})
        
        if epoch % 10 == 0:
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print('Loss: ', loss.item())



# switch the model from classification to generation
# switch to NLL likelihood


if __name__ == "__main__":
    root = '/users/rvinod/data/rvinod/code-prot-geometric/structures'
    dataset = struct_dataloader.StructureData(root)
    paths = dataset.processed_paths[:2000]
    data_list = [torch.load(pt) for pt in paths]

    loader = DataLoader(data_list, batch_size=32)