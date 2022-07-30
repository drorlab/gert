import sys
import math
import random

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

sys.path.append('../../../atom3d')
import atom3d.shard.shard as sh

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')


class PSR_Dataset(data.IterableDataset):
    def __init__(self, sharded, max_radius=7.0, seed=131313):
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
    def __init__(self, pos, elements, rmsd, gdt_ts, gdt_ha, tm, target, decoy):
        self.pos = pos
        self.elements = elements
        self.rmsd = rmsd
        self.gdt_ts = gdt_ts
        self.gdt_ha = gdt_ha
        self.tm = tm
        self.target = target
        self.decoy = decoy


class DataLoader(data.DataLoader):
    """Data loader which merges data objects from a
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
        labels = pd.read_hdf(sharded._get_shard(shard_idx), 'labels')

        if shuffle:
            groups = [df for _, df in shard.groupby('ensemble')]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for i, (ensemble_name, target_df) in enumerate(shard.groupby(['ensemble', 'subunit'])):
            rmsd, gdt_ts, gdt_ha, tm, target, decoy = get_labels(ensemble_name, labels)
            protein = df_to_graph(target_df, rmsd, gdt_ts, gdt_ha, tm, target, decoy, max_radius)
            if protein is None:
                continue
            yield protein


def get_labels(ensemble_name, labels):
    target, decoy = ensemble_name
    labels_df = labels[(labels['ensemble'] == target) & (labels['subunit'] == decoy)]
    rmsd = labels_df.iloc[0]['rmsd']
    gdt_ts = labels_df.iloc[0]['gdt_ts']
    gdt_ha = labels_df.iloc[0]['gdt_ha']
    tm = labels_df.iloc[0]['tm']
    return rmsd, gdt_ts, gdt_ha, tm, target, decoy


def df_to_graph(target_df, rmsd, gdt_ts, gdt_ha, tm, target, decoy, max_radius, filtered=True):
    """
    struct_df: Dataframe
    """
    if filtered:
        target_df = target_df[(target_df['fullname'].str.strip() == 'CA')]
        atom_pos = target_df[['x', 'y', 'z']].to_numpy()
        center_of_mass = np.mean(atom_pos, axis=0)
        kd_tree = KDTree(atom_pos)
        neighbors_pt_idx = kd_tree.query_ball_point(center_of_mass, r=max_radius, p=2.0)
        if len(neighbors_pt_idx) == 0:
           return None
        neighbors_df = target_df.iloc[neighbors_pt_idx].reset_index(drop=True)
        neighbors_df = neighbors_df[(neighbors_df['fullname'].str.strip() == 'CA')]
        if neighbors_df.shape[0] > 500:
            return None
        residues = neighbors_df['resname'].str.strip().to_numpy()
        atom_pos = neighbors_df[['x', 'y', 'z']].to_numpy() - center_of_mass
        one_hot = torch.zeros(len(atom_pos), len(RES_LABEL) + 1)
        for i in range(len(atom_pos)):
            if residues[i] in RES_LABEL:
                one_hot[i][RES_LABEL.index(residues[i])] = 1
        one_hot[:, -1][neighbors_df['fullname'].str.strip().to_numpy() == 'CA'] = 1
        return AtomEnvironment(torch.from_numpy(atom_pos), one_hot.float(), rmsd, gdt_ts, 
                                gdt_ha, tm, target, decoy)
    else:
        atom_pos = target_df[['x', 'y', 'z']].to_numpy()
        center_of_mass = np.mean(atom_pos, axis=0)
        kd_tree = KDTree(atom_pos)
        neighbors_pt_idx = kd_tree.query_ball_point(center_of_mass, r=max_radius, p=2.0)
        if len(neighbors_pt_idx) == 0:
            return None
        neighbors_df = target_df.iloc[neighbors_pt_idx].reset_index(drop=True)

        elements = neighbors_df['element'].to_numpy()
        atom_pos = neighbors_df[['x', 'y', 'z']].to_numpy() - center_of_mass
        one_hot = torch.zeros(len(atom_pos), len(PROT_ATOMS))
        for i in range(len(atom_pos)):
            if elements[i] in PROT_ATOMS:
                one_hot[i][PROT_ATOMS.index(elements[i])] = 1

        return AtomEnvironment(torch.from_numpy(atom_pos), one_hot.float(), rmsd, gdt_ts, 
                                gdt_ha, tm, target, decoy)


def custom_collate(data_list):
    pos = pad_sequence([atom_env.pos for atom_env in data_list], batch_first=True)
    elements = pad_sequence([atom_env.elements for atom_env in data_list], batch_first=True)
    rmsd = torch.tensor([atom_env.rmsd for atom_env in data_list])
    gdt_ts = torch.tensor([atom_env.gdt_ts for atom_env in data_list])
    gdt_ha = torch.tensor([atom_env.gdt_ha for atom_env in data_list])
    tm = torch.tensor([atom_env.tm for atom_env in data_list])
    target = np.array([atom_env.target for atom_env in data_list])
    decoy = np.array([atom_env.decoy for atom_env in data_list])

    return AtomEnvironment(pos, elements, rmsd, gdt_ts, gdt_ha, tm, target, decoy)
