from models.gert_noeqv import make_model
from models.heads import PSRFeedForward
from utils import get_mask, compute_global_correlations_mod
import data.psr.psr_dataloader as dl
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')


def train(epoch, transformer_model, ff_model, loader, criterion, optimizer, device, max_train_iter, print_frequency):
    transformer_model.train()
    ff_model.train()

    start = time.time()

    losses = [] 
    for it, subunit in enumerate(loader):
        subunit.pos = subunit.pos.to(device)
        subunit.elements = subunit.elements.to(device)
        subunit.gdt_ts = subunit.gdt_ts.to(device)
        optimizer.zero_grad()
        mask = get_mask(subunit.pos)
        # use the same dataloader as eqv and concat the elements here (formerly a one-hot)
        subunit.elements = torch.argmax(subunit.elements, dim=-1, keepdim=True)
        in_subunit = torch.cat([subunit.pos.float(), subunit.elements.float()], dim=-1)
        out = transformer_model(in_subunit, mask)
        output = ff_model(out, mask)
        loss = criterion(output, subunit.gdt_ts.float())
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
        subunit.gdt_ts = subunit.gdt_ts.to(device)

        mask = get_mask(subunit.pos)
        subunit.elements = torch.argmax(subunit.elements, dim=-1, keepdim=True)
        in_subunit = torch.cat([subunit.pos.float(), subunit.elements.float()], dim=-1)
        out = transformer_model(in_subunit, mask)
        output = ff_model(out, mask)
        
        loss = criterion(output, subunit.gdt_ts)
        losses.append(loss.item())
        target.extend(subunit.target.tolist())
        decoy.extend(subunit.decoy.tolist())
        y_true.extend(subunit.gdt_ts.tolist())
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


def train_noneqv_psr(ex, use_attention, data_dir, device, log_dir, checkpoint, num_epochs, batch_size,
              hidden_dim, learning_rate, workers, betas, eps, d_ff, d_atom,
              eta, max_radius, num_atoms, num_heads, max_train_iter, max_test_iter,
              print_frequency, test_mode=False):
    train_set = dl.PSR_Dataset(os.path.join(data_dir, 'train_decoy_50@508'), max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.PSR_Dataset(os.path.join(data_dir, 'val_decoy_50@56'), max_radius=max_radius)
    val_loader = dl.DataLoader(val_set, batch_size=batch_size, num_workers=workers)

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
    ff_model = PSRFeedForward(d_model=hidden_dim, n_out=1).to(device)

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
        if res['all_spearman'] > best_val_corr:
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
        test_set = dl.PSR_Dataset(os.path.join(data_dir, 'test_decoy_all@85'), max_radius=max_radius)
        test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
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