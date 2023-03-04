import Bio.PDB
from Bio import SeqIO
import os
import numpy as np
import torch
import argparse
import warnings
import sys
sys.path.append("..")

import numpy as np
from torch.linalg import qr

import requests
import json

import urllib.request
import urllib.parse

import embeddings


import shutil
import os
warnings.filterwarnings("ignore") # toggle

parser = argparse.ArgumentParser(description='Encode sequence and structure data')
parser.add_argument('--data', type=str, default='structure', help='sequence, structure')

path_to_pdbs = '/users/rvinod/data/rvinod/cath/cath/dompdb' # TODO change path to PDBs
path_to_coords = '/users/rvinod/data/rvinod/code-prot/coords'
path_to_fastas = '/users/rvinod/data/rvinod/code-prot/fastas'

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

cath_codes = { 1:'Mainly Alpha',
              2: 'Mainly Beta',
              3: 'Alpha Beta',
              4: 'Few Secondary Structures',
              6: 'Special'}


def get_coord(pdb):
    p = Bio.PDB.PDBParser()
    structure = p.get_structure(pdb, path_to_pdbs+'/'+pdb)
    ids = [a.get_id() for a in structure.get_atoms() if a.get_id()=='CA']
    coords = [a.get_coord() for a in structure.get_atoms() if a.get_id()=='CA']

    return torch.tensor(coords)

def get_coord_from_file(pdb_file):
    p = Bio.PDB.PDBParser()
    print(pdb_file)
    structure = p.get_structure(pdb_file.split('/')[-1], pdb_file)
    ids = [a.get_id() for a in structure.get_atoms() if a.get_id()=='CA']
    coords = [a.get_coord() for a in structure.get_atoms() if a.get_id()=='CA']

    return torch.tensor(coords)


def get_sequence(pdb_file):
    p = Bio.PDB.PDBParser()
    structure = p.get_structure(pdb_file.split('/')[-1], pdb_file)
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if residue.resname in d3to1.keys():
                    seq.append(d3to1[residue.resname])
                else:
                    seq.append('X')
            s = '>some_header\n',''.join(seq)
    return ''.join(seq)

def get_seq2(pdb):
    for record in SeqIO.parse(pdb, "pdb-atom"):
        print(record.seq)


def move_files(source_dir, target_dir):    
    file_names = os.listdir(source_dir)
        
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)


def get_R(n):
    H = torch.randn(n, n)
    Q, R = qr(H)
    return torch.mm(Q.t(), Q)


def get_cath_code(url):
    html = requests.get(url).text
    res = json.loads(html)
    code_class = res['data']['s35_id'].split('.')[0]
    return torch.tensor(int(code_class)) #cath_codes[code_class]

def url_cath(url):
    # trying to read the URL but with no internet connectivity
    try:
        x = urllib.request.urlopen(url)
        x = x.read()
    
    # Catching the exception generated     
    except Exception as e :
        print(str(e))
        return torch.tensor(5)
    

    label = int(json.loads(x.decode('utf-8'))['data']['s35_id'].split('.')[0])
    if label == 6:
        label = 5
    return torch.tensor(label - 1)

def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges

def get_edge_attrs(edges):
    edge_attr = []
    for i in range(len(edges[0])):
        elem = torch.tensor([edges[1][i] - edges[0][i]])
        edge_attr.append(embeddings.SinusoidalPositionEmbeddings(256).forward(elem).T)

    edge_attr = torch.cat(edge_attr).reshape(-1, 256)
    return edge_attr

def get_edges_batch(n_nodes, batch_size):
    print('getting edges...')
    edges = get_edges(n_nodes)
    #edge_attr = torch.ones(len(edges[0]) * batch_size, 256)
    edge_attr = get_edge_attrs(edges)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
        edge_attr = get_edge_attrs(edges)
    
    print('batched edges', edge_attr.shape)
    return edges, edge_attr



if __name__ == "__main__":
    pass