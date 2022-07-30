import sys
import math
import random

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

sys.path.append('../../atom3d')
import atom3d.shard.shard as sh

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P', 'H')
BASE_LABEL = ('A', 'U', 'G', 'C')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS', 'LIG')


class LBA_Dataset(data.IterableDataset):
    def __init__(self, sharded, labels_dir, max_radius=10.0, seed=131313):
        self.sharded = sh.Sharded(sharded, None)
        self.num_shards = self.sharded.get_num_shards()
        self.labels = pd.read_csv(labels_dir)
        self.seed = seed
        self.radius = max_radius

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = dataset_generator(self.sharded, range(self.num_shards), 
                                    self.labels, self.radius, shuffle=True)
        else:  # in a worker process, split workload
            per_worker = int(math.ceil(self.num_shards / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.num_shards)
            gen = dataset_generator(self.sharded, range(self.num_shards)[iter_start:iter_end], 
                                    self.labels, self.radius, shuffle=True)
        return gen


class AtomEnvironment:
    def __init__(self, pos, label, mask=None):
        self.pos = pos
        self.label = label
        self.mask = mask


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

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=custom_collate, **kwargs)


def dataset_generator(sharded, shard_indices, labels_df, max_radius, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)

        if shuffle:
            groups = [df for _, df in shard.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (ensemble_name, target_df) in enumerate(shard.groupby(['ensemble'])):
            label = labels_df[labels_df['pdb'] == ensemble_name]['label'].to_numpy()[0]
            protein = df_to_graph(target_df, label, max_radius)
            if protein is None:
                continue
            yield protein


def get_labels(ensemble_name, labels_df):
    labels_df = labels_df[labels_df['ensemble'] == ensemble_name]
    label = labels_df.iloc[0]['label']
    return label


def df_to_graph(target_df, label, max_radius):
    """
    struct_df: Dataframe
    """
    atom_pos = target_df[['x', 'y', 'z']].to_numpy()
    lig_df = target_df[target_df['subunit'] == 'LIG']
    lig_pos = lig_df[['x', 'y', 'z']].to_numpy()
    center_of_mass = np.mean(lig_pos, axis=0)
    kd_tree = KDTree(atom_pos)
    neighbors_pt_idx = kd_tree.query_ball_point(center_of_mass, r=max_radius, p=2.0)
    if len(neighbors_pt_idx) == 0:
        return None
    neighbors_df = target_df.iloc[neighbors_pt_idx].reset_index(drop=True)

    atom_pos = neighbors_df[['x', 'y', 'z', 'element']].to_numpy()
    atom_pos[..., :3] = atom_pos[..., :3] - center_of_mass

    for i in range(len(atom_pos)):
        if atom_pos[i][3] in PROT_ATOMS:
            atom_pos[i][3] = PROT_ATOMS.index(atom_pos[i][3])
        else:
            atom_pos[i][3] = len(PROT_ATOMS)

    return AtomEnvironment(torch.tensor(atom_pos.astype(np.float32)), label)

def custom_collate(data_list):
    pos = pad_sequence([atom_env.pos for atom_env in data_list]).permute(1, 0, 2)
    labels = torch.tensor([atom_env.label for atom_env in data_list])

    return AtomEnvironment(pos, labels)
