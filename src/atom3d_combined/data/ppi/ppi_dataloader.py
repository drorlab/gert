import sys
import math
import random

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

sys.path.append('../../atom3d')
import atom3d.shard.shard as sh
import atom3d.datasets.ppi.neighbors as nb

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')

class PPI_Dataset(data.IterableDataset):
    def __init__(self, sharded, max_radius=10.0, seed=131313):
        self.sharded = sh.Sharded(sharded, None)
        self.num_shards = self.sharded.get_num_shards()
        self.seed = seed
        self.max_radius = max_radius

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dataset_generator(self.sharded, range(self.num_shards), 
                      self.max_radius, shuffle=True)
        else:  # in a worker process, split workload
            per_worker = int(math.ceil(self.num_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)
            gen = dataset_generator(self.sharded, range(self.num_shards)[iter_start:iter_end],
                      self.max_radius, shuffle=True)
        return gen

class AtomEnvironment:
    def __init__(self, pos, label):
        self.pos = pos
        self.label = label


class DataLoader(data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=custom_collate, **kwargs)

def dataset_generator(sharded, shard_indices, max_radius, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)

        neighbors = pd.read_hdf(sharded._get_shard(shard_idx), 'neighbors')

        if shuffle:
            groups = [df for _, df in shard.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (ensemble_name, target_df) in enumerate(shard.groupby(['ensemble'])):
            subunits = target_df['subunit'].unique()
            sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
            positives = neighbors[neighbors.ensemble0 == ensemble_name]
            negatives = nb.get_negatives(positives, bound1, bound2)
            negatives['label'] = 0
            labels = create_labels(positives, negatives, neg_pos_ratio=1)
            
            for _, row in labels.iterrows():
                label = float(row['label'])
                chain_res1 = row[['chain0', 'residue0']].values
                chain_res2 = row[['chain1', 'residue1']].values
                graph1 = df_to_graph(bound1, chain_res1, label, max_radius)
                if graph1 is None:
                    continue
                graph2 = df_to_graph(bound2, chain_res2, label, max_radius)
                if graph2 is None:
                    continue
                yield graph1, graph2



def df_to_graph(struct_df, chain_res, label, max_radius):
    """
    struct_df: Dataframe
    """

    chain, resnum = chain_res
    res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]

    if 'CA' not in res_df.name.tolist():
        return None
    CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

    kd_tree = KDTree(struct_df[['x','y','z']].to_numpy())
    neighbors_pt_idx = kd_tree.query_ball_point(CA_pos, r=max_radius, p=2.0)
    neighbors_df = struct_df.iloc[neighbors_pt_idx].reset_index(drop=True)
    atom_pos = neighbors_df[['x', 'y', 'z', 'element']].to_numpy()
    atom_pos[..., :3] = atom_pos[..., :3] - CA_pos
    
    for i in range(len(atom_pos)):
        if atom_pos[i][3] in PROT_ATOMS:
            atom_pos[i][3] = PROT_ATOMS.index(atom_pos[i][3])
        else:
            atom_pos[i][3] = len(PROT_ATOMS)

    return AtomEnvironment(torch.from_numpy(atom_pos.astype(np.float64)), label)



def custom_collate(data_list):
    batch_1, batch_2 = zip(*data_list)

    pos_1 = pad_sequence([atom_env.pos for atom_env in batch_1], batch_first=True)
    labels_1 = torch.tensor([atom_env.label for atom_env in batch_1])

    pos_2 = pad_sequence([atom_env.pos for atom_env in batch_2], batch_first=True)
    labels_2 = torch.tensor([atom_env.label for atom_env in batch_2])

    return AtomEnvironment(pos_1, labels_1), AtomEnvironment(pos_2, labels_2)

def create_labels(positives, negatives, neg_pos_ratio):
    n = positives.shape[0] * neg_pos_ratio
    negatives = negatives.sample(n, random_state=131313, axis=0)
    labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
    return labels
