import math
import sys
import random
import pandas as pd
from scipy.spatial import KDTree
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

sys.path.append('../../../atom3d')
import atom3d.shard.shard as sh

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')


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
    def __init__(self, pos, elements, label):
        self.pos = pos
        self.elements = elements
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
    CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].to_numpy()[0]
    kd_tree = KDTree(struct_df[['x','y','z']].to_numpy())
    neighbors_pt_idx = kd_tree.query_ball_point(CA_pos, r=max_radius, p=2.0)
    neighbors_df = struct_df.iloc[neighbors_pt_idx].reset_index(drop=True)

    atom_pos = neighbors_df[['x', 'y', 'z']].to_numpy()
    atom_pos = atom_pos - CA_pos
    
    elements = neighbors_df['element'].to_numpy()
    one_hot = torch.zeros(elements.shape[0], len(PROT_ATOMS))
    for i in range(len(atom_pos)):
        if elements[i] in PROT_ATOMS:
            one_hot[i][PROT_ATOMS.index(elements[i])] = 1

    return AtomEnvironment(torch.tensor(atom_pos), one_hot.float(), label)


def custom_collate(data_list):
    pos = pad_sequence([atom_env.pos for atom_env in data_list], batch_first=True)
    elements = pad_sequence([atom_env.elements for atom_env in data_list], batch_first=True)
    labels = torch.tensor([atom_env.label for atom_env in data_list])

    return AtomEnvironment(pos, elements, labels)