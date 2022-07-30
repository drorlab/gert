from models.gert import make_model
from models.heads import EqvLBAFeedForward
from utils import get_mask, compute_global_correlations
import data.lba.eqv_lba_dataloader as dl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time


PROT_ATOMS = ('C', 'O', 'N', 'S', 'P', 'H')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS',
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS', 'LIG')
BASE_LABEL = ('A', 'U', 'G', 'C')


def train(epoch, transformer_model, ff_model, loader, criterion, optimizer, device, max_train_iter, print_frequency):
    transformer_model.train()
    ff_model.train()

    start = time.time()

    losses = []
    for it, subunit in enumerate(loader):
        subunit.pos = subunit.pos.to(device)
        subunit.elements = subunit.elements.to(device)
        subunit.label = subunit.label.to(device)
        optimizer.zero_grad()
        mask = get_mask(subunit.pos)
        out = transformer_model(subunit.pos, subunit.elements, attn_mask=mask)
        output = ff_model(out, subunit.pos)
        loss = criterion(output, subunit.label.float())
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(
                f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()
        if it == max_train_iter:
            return np.mean(losses)

    return np.mean(losses)


def get_mask(x):
    xyz = x[..., :3]
    r = torch.norm(xyz, 2, dim=-1, keepdim=False)
    pad_mask = r != 0
    pad_mask = pad_mask.unsqueeze(-1) * pad_mask.unsqueeze(-2)

    return pad_mask


@torch.no_grad()
def test(transformer_model, ff_model, loader, criterion, device, max_test_iter, print_frequency):
    transformer_model.eval()
    ff_model.eval()

    losses = []
    y_true = []
    y_pred = []
    for it, subunit in enumerate(loader):
        subunit.pos = subunit.pos.to(device)
        subunit.elements = subunit.elements.to(device)
        subunit.label = subunit.label.to(device)
        mask = get_mask(subunit.pos)
        out = transformer_model(subunit.pos, subunit.elements, attn_mask=mask)
        output = ff_model(out, subunit.pos)
        loss = criterion(output, subunit.label)
        losses.append(loss.item())
        y_true.extend(subunit.label.tolist())
        y_pred.extend(output.tolist())
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')
        if it == max_test_iter:
            break

    test_df = pd.DataFrame(
        np.array([y_true, y_pred]).T,
        columns=['true', 'pred'],
    )

    res = compute_global_correlations(test_df)
    print(test_df)
    print('RMSD:', np.sqrt(np.mean(losses)))
    return np.mean(losses), res, test_df


def compute_global_correlations(results):
    # Save metrics.
    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')
    return res


def train_eqv_lba(ex, use_attention, labels_dir, data_dir, device, log_dir, checkpoint, num_epochs, batch_size, num_dense,
                  learning_rate, workers, betas, eps, encoder_layers, num_heads, lmax,
                  N_hidden, max_radius, number_of_basis, h, L, 
                  max_train_iter, max_test_iter, print_frequency, test_mode=False):
    train_set = dl.LBA_Dataset(os.path.join(data_dir, 'lba_train@10'), labels_dir, max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.LBA_Dataset(os.path.join(data_dir, 'lba_val@10'), labels_dir, max_radius=max_radius)
    val_loader = dl.DataLoader(val_set, batch_size=batch_size, num_workers=workers)

    Rs_in = [(len(PROT_ATOMS) + len(RES_LABEL), 0)]
    Rs_hidden = [(N_hidden, l) for l in range(lmax)]
    Rs_ff = [(N_hidden * lmax, 0)]
    Rs_out = [(1, 0)]

    transformer_model = make_model(Rs_in, Rs_hidden, num_heads, num_dense, encoder_layers, 
                                    max_radius, number_of_basis, h, L, use_attention=use_attention).to(device)
    ff_model = EqvLBAFeedForward(
        Rs_hidden, Rs_ff, Rs_out, max_radius=max_radius, number_of_basis=10).to(device)

    model_parameters = filter(
        lambda p: p.requires_grad, transformer_model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters:', num_parameters)

    best_val_loss = 999
    best_val_corr = 0

    params = [x for x in transformer_model.parameters()] + \
        [x for x in ff_model.parameters()]

    criterion = nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(
        params, lr=learning_rate, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
                                                           min_lr=1e-6)

    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        optimizer.load_state_dict(cpt['optimizer_state_dict'])
        print('Loaded model from checkpoint')

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
        ex.log_scalar('val loss', val_loss)
        ex.log_scalar('all pearson', res['all_pearson'])
        ex.log_scalar('all kendall', res['all_kendall'])
        ex.log_scalar('all spearman', res['all_spearman'])
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(
            '\nVal Correlations (Pearson, Kendall, Spearman)\n'
            '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
                float(res["all_pearson"]),
                float(res["all_kendall"]),
                float(res["all_spearman"])))

    if test_mode:
        test_set = dl.LBA_Dataset(os.path.join(
            data_dir, 'lba_test@10'), labels_dir, max_radius=max_radius)
        test_loader = dl.DataLoader(
            test_set, batch_size=batch_size, num_workers=workers)
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        # cpt = torch.load('best_weights.pt')  # uncomment if you want to test on a pretrained model
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        test_criterion = nn.MSELoss().to(device)
        test_loss, res, _ = test(
            transformer_model, 
            ff_model, 
            test_loader, 
            test_criterion, 
            device,
            max_test_iter,
            print_frequency)
        ex.log_scalar('test loss', test_loss)
        ex.log_scalar('all pearson test', res['all_pearson'])
        ex.log_scalar('all kendall test', res['all_kendall'])
        ex.log_scalar('all spearman test', res['all_spearman'])
        print(
            '\nTest Correlations (Pearson, Kendall, Spearman)\n'
            '    all averaged: ({:.3f}, {:.3f}, {:.3f})'.format(
                float(res["all_pearson"]),
                float(res["all_kendall"]),
                float(res["all_spearman"])))

    return best_val_loss