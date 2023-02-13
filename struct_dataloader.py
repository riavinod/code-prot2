import os.path as osp
import os

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.data import Dataset, download_url
import Bio.PDB
from Bio import SeqIO
import os
import numpy as np
import torch
import argparse
import warnings
import utils
warnings.filterwarnings("ignore") # toggle

from sequence_models.pretrained import load_model_and_alphabet

import embeddings




class StructureData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
       

    @property
    def raw_file_names(self):
        return os.listdir(self.root)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir) #self.processed_file_names 
    
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     print('**RAW DIR**', self.raw_dir)
       
    #     # if zipped; note that there can be some better optimization here
    #     unzip = self.raw_dir + '/' + os.listdir(self.raw_dir)[0]
    #     print('**TO UNZIP**', unzip)
    #     file = url.split('/')[-1]
    #     print('**FILE**', file)
    #     import tarfile
    #     # open file
    #     file = tarfile.open(unzip)
    #     # extracting file
    #     file.extractall(self.raw_dir)
    #     file.close()
    #     utils.move_files('/users/rvinod/data/rvinod/cath/raw/dompdb', '/users/rvinod/data/rvinod/cath/raw')
    #     os.rmdir('/users/rvinod/data/rvinod/cath/raw/dompdb')
    #     os.remove(unzip)

                
    def process(self):
        idx = 0
        D = 256 # TODO change here

        for file in os.listdir(self.raw_dir):
            # Read data from `raw_path`.
            file_path = self.raw_dir +'/'+file

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            print('** raw file path **', file_path)
            print('** processed file number **', osp.join(self.processed_dir, f'data_{idx}.pt'))


            # get number of nodes and coordinates
            x = utils.get_coord_from_file(file_path)
            num_nodes = x.shape[0]

            # get node features
            node_ids = torch.arange(num_nodes)
            node_attrs = []
            for n in node_ids:
                h = embeddings.SinusoidalPositionEmbeddings(D).forward(torch.tensor([n])).T + \
                            torch.matmul(utils.get_R(D), embeddings.SinusoidalPositionEmbeddings(D).forward(torch.tensor([0])).T) #timestep is 0 for the first layer
                node_attrs.append(h)
            node_attrs = torch.cat(node_attrs).reshape(-1, 256)

        
            # get edge indices and features
            rows, cols = [], []
            edge_attr = []
            for batch_idx in range(1):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        rows.append(i + 1 * num_nodes)
                        cols.append(j + 1 * num_nodes)
                        elem = torch.tensor([i-j])
                        edge_attr.append(embeddings.SinusoidalPositionEmbeddings(D).forward(elem).T)
            
            edges = torch.tensor([rows, cols])
            edge_attr = torch.cat(edge_attr).reshape(-1, 256)


            # save a data object
            data = Data(x=x, node_attr=node_attrs, edge_index=edges, edge_attr=edge_attr)
            print(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
          
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join('', f'data_{idx}.pt'))
        return data


url = 'http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz'
root = '/users/rvinod/data/rvinod/cath'

dataset = StructureData(root)
print(dataset)
print('loading...')

print('Sample Data object: ')
print(torch.load(dataset.processed_paths[0]))
print(torch.load(dataset.processed_paths[1]))

# paths = dataset.processed_paths
# data_list = [torch.load(pt) for pt in paths]
# loader = DataLoader(data_list, batch_size=4)

# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

