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
RES_ONE_TO_THREE_LETTERS = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'E':'GLU', 'Y':'TYR', 
                            'Q':'GLN', 'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'V':'VAL',
                            'M':'MET', 'F':'PHE', 'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP'}
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')


class MSP_Dataset(data.IterableDataset):
    def __init__(self, sharded, max_radius=7.0, seed=131313):
        self.sharded = sh.Sharded(sharded, None)
        self.num_shards = self.sharded.get_num_shards()
        self.seed = seed
        self.counts = get_counts(self.sharded, range(self.num_shards))
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

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=custom_collate, **kwargs)


def get_counts(sharded, shard_indices):
    counts = {0:0, 1:0}
    for shard_idx in shard_indices:
        labels = pd.read_hdf(sharded._get_shard(shard_idx), 'labels')['label'].to_numpy()

        num_ones = np.sum(labels)
        num_zeros = labels.shape[0] - num_ones

        counts[0] += num_zeros
        counts[1] += num_ones

    return counts


def dataset_generator(sharded, shard_indices, max_radius, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    for shard_idx in shard_indices:
        struc_df = sharded.read_shard(shard_idx)

        labels_df = pd.read_hdf(sharded._get_shard(shard_idx), 'labels')

        if shuffle:
            groups = [df for _, df in struc_df.groupby('ensemble')]
            random.shuffle(groups)
            struc_df = pd.concat(groups).reset_index(drop=True)

        for i, (ensemble_name, target_df) in enumerate(struc_df.groupby(['ensemble'])):
            label, chain, residue, original_resname, mutated_resname = get_subunits(labels_df, ensemble_name)
            original_df = target_df[target_df['subunit'] == 'original']
            mutated_df = target_df[target_df['subunit'] == 'mutated']
            original = df_to_graph(original_df, chain, residue, original_resname, label, max_radius)
            if original is None:
                continue
            mutated = df_to_graph(mutated_df, chain, residue, mutated_resname, label, max_radius)
            if mutated is None:
                continue
            yield original, mutated


def get_subunits(labels_df, ensemble_name):
    ensemble_labels = labels_df[labels_df['ensemble'] == ensemble_name]
    ensemble_labels = ensemble_labels[['label', 'chain', 'residue', 'original_resname', 'mutated_resname']].to_numpy()[0]
    label, chain, residue, original_resname, mutated_resname = ensemble_labels
    if original_resname in RES_ONE_TO_THREE_LETTERS:
        original_resname = RES_ONE_TO_THREE_LETTERS[original_resname]
    if mutated_resname in RES_ONE_TO_THREE_LETTERS:
        mutated_resname = RES_ONE_TO_THREE_LETTERS[mutated_resname]
    return label, chain, residue, original_resname, mutated_resname


def df_to_graph(struct_df, chain, resnum, resname, label, max_radius):
    """
    struct_df: Dataframe
    """
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

    return AtomEnvironment(torch.tensor(atom_pos.astype(np.float32)), label)


def custom_collate(data_list):
    batch_1, batch_2 = zip(*data_list)

    pos_1 = pad_sequence([atom_env.pos for atom_env in batch_1]).permute(1, 0, 2)
    labels_1 = torch.tensor([atom_env.label for atom_env in batch_1])

    pos_2 = pad_sequence([atom_env.pos for atom_env in batch_2]).permute(1, 0, 2)
    labels_2 = torch.tensor([atom_env.label for atom_env in batch_2])

    return AtomEnvironment(pos_1, labels_1), AtomEnvironment(pos_2, labels_2)
