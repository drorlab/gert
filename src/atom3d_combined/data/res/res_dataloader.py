import math
import sys
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.append('../../../atom3d')
import atom3d.shard.shard as sh

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')
WEIGHT_P = 1.4

class ResDel_Dataset(data.IterableDataset):
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
    def __init__(self, pos, label, class_weights):
        self.pos = pos
        self.label = label
        self.class_weights = class_weights


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
    Generate grids from sharded dataset
    """

    for shard_idx in shard_indices:
        shard = sharded.read_shard(shard_idx)
        if shuffle:
            groups = [df for _, df in shard.groupby(['ensemble', 'subunit'])]
            random.shuffle(groups)
            shard = pd.concat(groups).reset_index(drop=True)

        for e, target_df in shard.groupby(['ensemble', 'subunit']):
            _, subunit = e
            res_name = subunit.split('_')[-1]
            label = RES_LABEL.index(res_name)
            protein = df_to_graph(target_df, subunit, label, max_radius)
            if protein is None:
                continue

            yield protein


def df_to_graph(struct_df, chain_res, label, max_radius):
    """
    label: residue label (int)
    chain_res: chain ID_residue ID_residue name defining center residue
    struct_df: Dataframe with entire environment
    """
    chain, resnum, _ = chain_res.split('_')
    res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue.astype(str) == resnum)]
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

    count_keys = struct_df['resname'].value_counts().keys().to_numpy()
    count_values = struct_df['resname'].value_counts().to_numpy()
    class_weights = torch.zeros(len(RES_LABEL))
    for i, value in enumerate(count_values):
        if count_keys[i] in RES_LABEL:
            class_weights[RES_LABEL.index(count_keys[i])] = 1/(float(value))**WEIGHT_P
    class_weights = F.normalize(class_weights, p=1, dim=0)

    return AtomEnvironment(torch.tensor(atom_pos.astype(np.float32)), label, class_weights)
            

def custom_collate(data_list):
    pos = pad_sequence([atom_env.pos for atom_env in data_list]).permute(1, 0, 2)
    labels = torch.tensor([atom_env.label for atom_env in data_list])
    class_weights = torch.cat([atom_env.class_weights.unsqueeze(-1) for atom_env in data_list], dim=-1).permute(1, 0)

    return AtomEnvironment(pos, labels, class_weights)