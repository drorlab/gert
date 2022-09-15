import os

import pandas as pd
import torch
from atom3d.protein_ligand.get_labels import get_label
from torch_geometric.data import Dataset, Data, DataLoader

from atom3d.util import formats as dt
from atom3d.util import file as fi
from atom3d.torch import graph
from atom3d.splits import splits as sp


# loader for pytorch-geometric

class GraphPDBBind(Dataset):
    """
    PDBBind dataset in pytorch-geometric format. 
    Ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphPDBBind, self).__init__(root, transform, pre_transform)

        self.pdb_idx_dict = self.get_idx_mapping()
        self.idx_pdb_dict = {v:k for k,v in self.pdb_idx_dict.items()}

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        num_samples = len(self.raw_file_names) // 3 # each example has protein/pocket/ligand files
        return [f'data_{i}.pt' for i in range(num_samples)]

    def get_idx_mapping(self):
        pdb_idx_dict = {}
        i = 0
        for file in self.raw_file_names:
            if '_pocket' in file:
                pdb_code = fi.get_pdb_code(file)
                pdb_idx_dict[pdb_code] = i
                i += 1
        return pdb_idx_dict


    def pdb_to_idx(self, pdb):
        return self.pdb_idx_dict.get(pdb)

    def process(self):
        label_file = os.path.join(self.root, 'pdbbind_refined_set_labels.csv')
        label_df = pd.read_csv(label_file)
        i = 0
        for raw_path in self.raw_paths:
            pdb_code = fi.get_pdb_code(raw_path)
            y = torch.FloatTensor([get_label(pdb_code, label_df)])
            if '_ligand' in raw_path:
                mol_graph = graph.mol_to_graph(dt.read_sdf_to_mol(raw_path, add_hs=True)[0])
            elif '_pocket' in raw_path:
                prot_graph = graph.prot_df_to_graph(dt.bp_to_df(dt.read_any(raw_path, name=pdb_code)))
                node_feats, edge_index, edge_feats, pos = graph.combine_graphs(prot_graph, mol_graph, edges_between=True)
                data = Data(node_feats, edge_index, edge_feats, y=y, pos=pos)
                data.pdb = pdb_code
                torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1
            else:
                continue

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def pdbbind_dataloader(batch_size, data_dir='../../data/pdbbind', split_file=None):
    """
    Creates dataloader for PDBBind dataset with specified split. 
    Assumes pre-computed split in 'split_file', which is used to index Dataset object
    TODO: implement on-the-fly splitting using split functions
    """
    dataset = GraphPDBBind(root=data_dir)
    if split_file is None:
        return DataLoader(dataset, batch_size, shuffle=True)
    indices = sp.read_split_file(split_file)
    print(indices)

    # if split specifies pdb ids, convert to indices
    if isinstance(indices[0], str):
        indices = [dataset.pdb_to_idx(x) for x in indices if dataset.pdb_to_idx(x)]
        pdb_codes = [x for x in indices if dataset.pdb_to_idx(x)]
    return DataLoader(dataset.index_select(indices), batch_size, shuffle=True)

if __name__=="__main__":
    dataset = GraphPDBBind(root='../../data/pdbbind')



