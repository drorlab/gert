from models.gert_noeqv import make_model
from models.heads import RSRFeedForward
from utils import get_mask, compute_global_correlations_mod
import atom3d.datasets as da
import data.rsr.rsr_dataloader as dl
import data.rsr.rsr_dataloader_lmdb as dl_lmdb
import random
import re
import os
import time
import numpy as np
import pandas as pd
import torch_geometric
import torch
import torch.nn as nn
import torch.utils.data as tdata
import atom3d.util.rosetta as sc

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P', 'H')
BASE_LABEL = ('A', 'U', 'G', 'C')

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


def train(epoch, transformer_model, ff_model, loader, criterion, optimizer, device, max_train_iter, print_frequency):
    transformer_model.train()
    ff_model.train()
    
    start = time.time()
    
    losses = []
    for it, subunit in enumerate(loader):
        subunit.pos = subunit.pos.to(device)
        subunit.elements = subunit.elements.to(device)
        subunit.rmsd = subunit.rmsd.to(device)
        optimizer.zero_grad()
        mask = get_mask(subunit.pos)
        # use the same dataloader as eqv and concat the elements here (formerly a one-hot)
        subunit.elements = torch.argmax(subunit.elements, dim=-1, keepdim=True)
        in_subunit = torch.cat([subunit.pos.float(), subunit.elements.float()], dim=-1)
        
        # TODO: Figure out what is going on here:
        if in_subunit.shape[1] == 1:
            print(in_subunit.shape)
            print(mask.shape)
            print(subunit.pos.shape)
            print(subunit.elements.shape)
            continue
        
        out = transformer_model(in_subunit, mask)
        output = ff_model(out, mask)
        loss = criterion(output, subunit.rmsd.float())
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()
        if it == max_train_iter:
            return np.mean(losses)

    return np.mean(losses)


@torch.no_grad()
def test(transformer_model, ff_model, loader, criterion, device, max_test_iter, print_frequency):
    transformer_model.eval()
    ff_model.eval()

    losses = []
    target = []
    decoy = []
    y_true = []
    y_pred = []
    for it, subunit in enumerate(loader):
        subunit.pos = subunit.pos.to(device)
        subunit.elements = subunit.elements.to(device)
        subunit.rmsd = subunit.rmsd.to(device)
        
        mask = get_mask(subunit.pos)
        subunit.elements = torch.argmax(subunit.elements, dim=-1, keepdim=True)
        in_subunit = torch.cat([subunit.pos.float(), subunit.elements.float()], dim=-1)
        out = transformer_model(in_subunit, mask)
        output = ff_model(out, mask)
        
        loss = criterion(output, subunit.rmsd)
        losses.append(loss.item())
        target.extend(subunit.target.tolist())
        decoy.extend(subunit.decoy.tolist())
        y_true.extend(subunit.rmsd.tolist())
        y_pred.extend(output.tolist())
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')
        if it == max_test_iter:
            break
    
    test_df = pd.DataFrame(
        np.array([target, decoy, y_true, y_pred]).T,
        columns=['target', 'decoy', 'true', 'pred'],
        )
    
    res = compute_global_correlations_mod(test_df)
    print(test_df)

    return np.mean(losses), res, test_df


def train_noneqv_rsr(ex, use_attention, data_dir, device, log_dir, checkpoint, num_epochs, batch_size,
              hidden_dim, learning_rate, workers, betas, eps, d_ff, d_atom,
              eta, max_radius, num_atoms, num_heads, max_train_iter, max_test_iter,
              print_frequency, test_mode=False):
    # Uncomment if you want the normal sharding dataloading back
    # train_set = dl.RSR_Dataset(os.path.join(data_dir, 'structures_train@10'), max_radius=max_radius)
    # train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    # val_set = dl.RSR_Dataset(os.path.join(data_dir, 'structures_val@10'), max_radius=max_radius)
    # val_loader = dl.DataLoader(val_set, batch_size=batch_size, num_workers=workers)

    PATH_TO_DATASET = '/oak/stanford/groups/rondror/projects/atom3d/lmdb/RSR/raw/candidates/data-1000-per-target/data'
    tmp_dataset = da.load_dataset(PATH_TO_DATASET, 'lmdb')

    # Uncomment if you want to do a split by target, but not ordered temporally
    train_indices, val_indices, test_indices = [], [], []
    # val_rnas = [IDX_TO_NAME[idx] for idx in ['2', '5', '17', '20']]
    # test_rnas = [IDX_TO_NAME[idx] for idx in ['3', '6', '18', '21']]

    # Normal splits:
    val_rnas = [IDX_TO_NAME[idx] for idx in ['14b', '14f', '15', '17']]
    test_rnas = [IDX_TO_NAME[idx] for idx in ['18', '19', '20', '21']]

    for i in range(len(tmp_dataset)):
        name, _ = re.findall(r"'(.*?)'", tmp_dataset[i]['id'])
        if name in val_rnas:
            val_indices.append(i)
        elif name in test_rnas:
            test_indices.append(i)
        else:
            train_indices.append(i)

    # Uncomment if you want to do a random split
    # indices = [x for x in range(len(tmp_dataset))]
    # random.shuffle(indices)
    # train_indices = indices[:-8000]
    # val_indices = indices[-8000:-4000]
    # test_indices = indices[-4000:]

    # Check the sizes of train/val/test sets
    print("Train set size:", len(train_indices))
    print("Validation set size:", len(val_indices))
    print("Test set size:", len(test_indices))

    train_set = dl_lmdb.RSR_Dataset(train_indices, max_radius=max_radius)
    train_loader = dl_lmdb.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl_lmdb.RSR_Dataset(val_indices, max_radius=max_radius)
    val_loader = dl_lmdb.DataLoader(val_set, batch_size=batch_size, num_workers=workers)

    transformer_model = make_model(
        num_heads=num_heads, 
        d_model=hidden_dim, 
        d_ff=d_ff, 
        d_atom=d_atom,
        eta=eta, 
        Rc=max_radius, 
        num_atoms=num_atoms, 
        N=2,
        num_dense=2,
        use_attention=use_attention).to(device)
    ff_model = RSRFeedForward(d_model=hidden_dim, n_out=1).to(device)

    model_parameters = filter(lambda p: p.requires_grad, transformer_model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters:', num_parameters)
    
    best_val_loss = 999
    best_val_corr = 0
    
    params = [x for x in transformer_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=2,
                                                           min_lr=1e-6)

    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        optimizer.load_state_dict(cpt['optimizer_state_dict'])
        print('loaded model from checkpoint')

    print(f'Training for {num_epochs} epochs')
    print('---------------------------------')
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(
            epoch, 
            transformer_model, 
            ff_model, 
            train_loader, 
            criterion, 
            optimizer,
            device,
            max_train_iter, 
            print_frequency)
        ex.log_scalar('Train Loss', train_loss)
        print('Validating...')
        val_loss, res, _ = test(
            transformer_model, 
            ff_model, 
            val_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            torch.save({
                'epoch': epoch,
                'transformer_state_dict': transformer_model.state_dict(),
                'ff_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights.pt'))
            best_val_loss = val_loss
            best_val_corr = res['all_spearman']
        elapsed = (time.time() - start)
        ex.log_scalar('Validation Loss', val_loss)
        ex.log_scalar('All Pearson', res['all_pearson'])
        ex.log_scalar('All Spearman', res['all_spearman'])
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(
            '\nVal Correlations (Pearson, Kendall, Spearman)\n'
            '    per-target averaged median: ({:.3f}, {:.3f}, {:.3f})\n'
            '    per-target averaged mean: ({:.3f}, {:.3f}, {:.3f})\n'
            '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
            float(res["per_target_median_pearson"]),
            float(res["per_target_median_kendall"]),
            float(res["per_target_median_spearman"]),
            float(res["per_target_mean_pearson"]),
            float(res["per_target_mean_kendall"]),
            float(res["per_target_mean_spearman"]),
            float(res["all_pearson"]),
            float(res["all_kendall"]),
            float(res["all_spearman"])))

    if test_mode:
        # test_set = dl.RSR_Dataset(os.path.join(data_dir, 'structures_test@10'), max_radius=max_radius)
        # test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
        test_set = dl_lmdb.RSR_Dataset(test_indices, max_radius=max_radius)
        test_loader = dl_lmdb.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        test_loss, res, _ = test(
            transformer_model, 
            ff_model, 
            test_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)
        ex.log_scalar('Test Loss', test_loss)
        print(
            '\nTest Correlations (Pearson, Kendall, Spearman)\n'
            '    per-target averaged median: ({:.3f}, {:.3f}, {:.3f})\n'
            '    per-target averaged mean: ({:.3f}, {:.3f}, {:.3f})\n'
            '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
            float(res["per_target_median_pearson"]),
            float(res["per_target_median_kendall"]),
            float(res["per_target_median_spearman"]),
            float(res["per_target_mean_pearson"]),
            float(res["per_target_mean_kendall"]),
            float(res["per_target_mean_spearman"]),
            float(res["all_pearson"]),
            float(res["all_kendall"]),
            float(res["all_spearman"])))

    return best_val_loss
