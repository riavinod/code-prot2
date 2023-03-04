import os 

import torch
from torch import nn
from torch.nn import functional as F
import math
from torch_geometric.loader import DataLoader

# from torch_geometric.data import Data
# import torch
# from torch_geometric.data import Dataset, download_url

# import data.embeddings as embeddings
# import data.struct_dataloader as struct_dataloader

# root = '/users/rvinod/data/rvinod/code-prot-geometric/structures'
# dataset = struct_dataloader.StructureData(root)
# print(dataset)
# paths = dataset.processed_paths[:100]
# data_list = [torch.load(pt) for pt in paths]

# loader = DataLoader(data_list, batch_size=32)

# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print(data.batch.shape)
#     print(data.ptr)
