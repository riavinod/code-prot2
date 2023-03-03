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
import data.embeddings

from sequence_models.pretrained import load_model_and_alphabet



class JointData(Dataset):
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

            # get label
            y_url = 'http://www.cathdb.info/version/v4_3_0/api/rest/domain_summary/' + file_path.split('/')[-1]
            print('** y_url **', y_url)
            y = utils.url_cath(y_url)


            # get coordinates and number of nodes
            x = utils.get_coord_from_file(file_path)/10
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
            # edges 2.0
            rows, cols = [], []
            edges = []
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    if i!=j:
                        edges.append([i, j])
            
            # edge_attr 2.0
            edge_attr = []
            for e in edges:
                elem = torch.tensor([e[1] - e[0]])
                edge_attr.append(embeddings.SinusoidalPositionEmbeddings(D).forward(elem).T)

            edges = torch.tensor(edges).T

            edge_attr = torch.cat(edge_attr).reshape(-1, D)


            # CARP representation of sequence
            seq = utils.get_sequence(file_path)
            seq_input = collater([seq])[0]
            s = model(seq_input)['representations'][16].reshape(128, -1)


            # save the data
            data = Data()
            print(x.shape, h.shape, s.shape)
            #node_features = torch.cat((x, h.T, s.T), dim=1)
            data = Data(x=x, s=s, node_attr=node_attrs, edge_index=edges, edge_attr=edge_attr, y=y)
            print(data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
          
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        print('IN THE GET')
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data



# model, collater = load_model_and_alphabet('carp_600k')

# url = 'http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz'
# root = '/users/rvinod/data/rvinod/code-prot-geometric/joint'

# dataset = JointData(root)
# print(dataset)
# print('loading...')

# paths = dataset.processed_paths
# data_list = [torch.load(pt) for pt in paths]
# loader = DataLoader(data_list, batch_size=16)

# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()