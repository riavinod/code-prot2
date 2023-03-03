import torch
from torch_geometric.data import Data
from torch import nn
import torch


from data import embeddings as embeddings
from data import utils as utils

# x = torch.load('/users/rvinod/data/rvinod/cath/processed/data_222.pt')

# y_url = 'http://www.cathdb.info/version/v4_3_0/api/rest/domain_summary/2vszB02' 
# y = torch.tensor(utils.get_cath_code(y_url))

# print(y)

# x.y = torch.tensor(y)

# print(x)

# new = Data(x = x.x, edge_index = x.edge_index, edge_attr = x.edge_attr, node_attr = x.node_attr, y = x.y)
# print(new)
# torch.save('tmp/new.pt', new)


# D = 256

# def get_node_attr(node_ids):
#     node_attrs = []
#     for n in node_ids:
#         h = embeddings.SinusoidalPositionEmbeddings(D).forward(torch.tensor([n])).T + \
#                     torch.matmul(utils.get_R(D), embeddings.SinusoidalPositionEmbeddings(D).forward(torch.tensor([0])).T) #timestep is 0 for the first layer
#         node_attrs.append(h)
#     node_attrs = torch.cat(node_attrs).reshape(-1, 256)
#     return node_attrs

# # coordinates
# x = torch.load('/users/rvinod/data/rvinod/code-prot/coords/2fphX02.pt')

# # number of nodes
# num_nodes = x.shape[0]
# node_ids = torch.arange(num_nodes)

# # node attributes
# node_attrs = get_node_attr(node_ids)

# # edges
# rows, cols = [], []
# edges = []
# for i in range(num_nodes):
#     for j in range(i, num_nodes):
#         if i!=j:
#             edges.append([i, j])

# # edge attributes
# edge_attr = []
# for e in edges:
#     elem = torch.tensor([e[1] - e[0]])
#     edge_attr.append(embeddings.SinusoidalPositionEmbeddings(D).forward(elem).T)

# edges = torch.tensor(edges).T
# edge_attr = torch.cat(edge_attr).reshape(-1, D)

# print(x.shape)
# print(node_attrs.shape)
# print(edges.shape)
# print(edge_attr.shape)

# x = torch.rand(100, 3)
# h = torch.rand(100, 256)
# e = torch.rand(2, 4950)
# e_attr = torch.rand(2, 1267200)

# torch.save(x, '/users/rvinod/data/rvinod/code-prot/dummy/x.pt')
# torch.save(h, '/users/rvinod/data/rvinod/code-prot/dummy/h.pt')
# torch.save(e, '/users/rvinod/data/rvinod/code-prot/dummy/e.pt')
# torch.save(e_attr, '/users/rvinod/data/rvinod/code-prot/dummy/e_attr.pt')


from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin = Linear(hidden_channels, 6)

    def forward(self, h, batch):

        # 1. Obtain node embeddings 

        # 2. Readout layer
        h = global_mean_pool(h, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin(h)
        
        return h

mlp = MLP(256)
x = mlp(torch.rand(471, 256))

print(x.shape)