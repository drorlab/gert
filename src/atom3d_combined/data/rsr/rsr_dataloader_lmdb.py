import sys
import math
import random
import re

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data

sys.path.append('../../../atom3d')
import atom3d.shard.shard as sh
import atom3d.datasets as da

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P', 'H')
BASE_LABEL = ('A', 'U', 'G', 'C')
IMP_ATOMS = ('C', 'N', 'O')

IDX_TO_NAME = {
    '1': 'rna_puzzle_1',
    '2': 'rna_puzzle_2_hack',
    '3': 'rna_puzzle_3',
    '4': 'rna_puzzle_4_with_3IQP',
    '5': 'rna_puzzle_5_homology',
    '6': 'rna_puzzle_6_homology',
    '7': 'rna_puzzle_7',
    '8': 'other_rna_puzzle_8',
    '9': 'rna_puzzle_9_2xnw_tloop',
    '10': 'new_rna_puzzle_10',
    '11': 'rna_puzzle_11',
    '12': 'rna_puzzle_12',
    '13': 'other_rna_puzzle_13',
    '14b': 'rna_puzzle_14_bound',
    '14f': 'rna_puzzle_14_free',
    '15': 'rna_puzzle_15',
    '17': 'rna_puzzle_17',
    '18': 'rna_puzzle_18_with_4PQV',
    '19': 'rna_puzzle_19_t_loop',
    '20': 'rna_puzzle_20_t_loop',
    '21': 'rna_puzzle_21',
}

NAME_TO_IDX = {value: key for key, value in IDX_TO_NAME.items()}

pd.options.mode.chained_assignment = None


class RSR_Dataset(data.IterableDataset):
    def __init__(self, indices, max_radius=7.0, seed=131313):
        self.seed = seed
        self.max_radius = max_radius
        self.indices = indices

    def __iter__(self):
        gen = dataset_generator(self.indices, self.max_radius)
        return gen


class AtomEnvironment:
    def __init__(self, pos, elements, rmsd, target, decoy):
        self.pos = pos
        self.elements = elements
        self.rmsd = rmsd
        self.target = target
        self.decoy = decoy


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


def dataset_generator(indices, max_radius, shuffle=True):
    """
    Generator that convert sharded HDF dataset to graphs
    """
    PATH_TO_DATASET = '/oak/stanford/groups/rondror/projects/atom3d/lmdb/RSR/raw/candidates/data-1000-per-target/data'
    dataset = da.load_dataset(PATH_TO_DATASET, 'lmdb')
    dataset = data.Subset(dataset, indices)

    no_atoms, no_scores = 0, 0

    for elem in dataset:
        if elem['atoms'] is None or len(elem['atoms']) == 0:
            no_atoms += 1
            continue

        if elem['scores'] is None:
            no_scores += 1
            continue

        target_df = elem['atoms']
        rmsd = elem['scores']['rms']
        row = elem['atoms'].iloc[0]
        decoy = row['ensemble'][:-4]

        name, _ = re.findall(r"'(.*?)'", elem['id'])
        target = NAME_TO_IDX[name]

        protein = df_to_graph(target_df, rmsd, target, decoy, max_radius)
        if protein is None:
            continue
        yield protein

    print("Number of examples without atoms:", no_atoms)
    print("Number of examples without scores:", no_scores)


def get_labels(ensemble_name, labels):
    target, decoy = ensemble_name
    labels_df = labels[(labels['ensemble'] == target) & (labels['subunit'] == decoy)]
    rmsd = labels_df.iloc[0]['label']
    return rmsd, target, decoy


def df_to_graph(target_df, rmsd, target, decoy, max_radius, filtered=True):
    if filtered:
        # Filter for specific subset of atoms (C, N, O).
        target_df['fullname'] = target_df['fullname'].str.strip()
        target_df = target_df[(target_df.fullname.str.contains('C')) | 
                              (target_df.fullname.str.contains('N')) |
                              (target_df.fullname.str.match(pat = 'O[^P].*'))]

        # Give consistent labeling to all C, N, O atoms.
        target_df.loc[target_df.fullname.str.contains('C'), 'fullname'] = 'C'
        target_df.loc[target_df.fullname.str.contains('N'), 'fullname'] = 'N'
        target_df.loc[target_df.fullname.str.match(pat = 'O[^P].*'), 'fullname'] = 'O'

        atom_pos = target_df[['x', 'y', 'z']].to_numpy()
        center_of_mass = np.mean(atom_pos, axis=0)
        kd_tree = KDTree(atom_pos)
        neighbors_pt_idx = kd_tree.query_ball_point(center_of_mass, r=max_radius, p=2.0)
        target_df = target_df.iloc[neighbors_pt_idx].reset_index(drop=True)
        if len(target_df) <= 1:
            return None
        residues = target_df['resname'].str.strip().to_numpy()
        atom_pos = target_df[['x', 'y', 'z']].to_numpy() - center_of_mass

        # Two-hot vector: one of the first len(BASE_LABEL) positions will be
        # 1 (designating the base label), and one of the last len(IMP_ATOMS)
        # positions will be 1.
        two_hot = torch.zeros(len(atom_pos), len(BASE_LABEL) + len(IMP_ATOMS))
        for i in range(len(residues)):
            if residues[i] in BASE_LABEL:
                two_hot[i][BASE_LABEL.index(residues[i])] = 1
        for i in range(len(IMP_ATOMS)):
            two_hot[:, len(BASE_LABEL) + i][target_df['fullname'].str.strip().to_numpy() == IMP_ATOMS[i]] = 1

        return AtomEnvironment(torch.from_numpy(atom_pos), two_hot.float(), rmsd, target, decoy)
    else:
        print ("Not implemented!")
        assert False


def custom_collate(data_list):
    pos = pad_sequence([atom_env.pos for atom_env in data_list], batch_first=True)
    elements = pad_sequence([atom_env.elements for atom_env in data_list], batch_first=True)
    rmsd = torch.tensor([atom_env.rmsd for atom_env in data_list])
    target = np.array([atom_env.target for atom_env in data_list])
    decoy = np.array([atom_env.decoy for atom_env in data_list])

    return AtomEnvironment(pos, elements, rmsd, target, decoy)