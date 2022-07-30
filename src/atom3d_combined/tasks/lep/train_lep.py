from models.gert_noeqv import make_model
from models.heads import LEPFeedForward
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import get_mask
import data.lep.lep_dataloader as dl
import os
import time
import numpy as np
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
    for it, (active, inactive) in enumerate(loader):
        active.label, inactive.label = active.label.to(device), inactive.label.to(device)
        active.neighbors, inactive.neighbors = active.neighbors.to(device), inactive.neighbors.to(device)
        optimizer.zero_grad()
        mask1 = get_mask(active.neighbors)
        out_active = transformer_model(active.neighbors, mask1)
        mask2 = get_mask(inactive.neighbors)
        out_inactive = transformer_model(inactive.neighbors, mask2)
        output = ff_model(out_active, out_inactive)

        loss = criterion(output, active.label.float())
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
    y_true = []
    y_pred = []

    for it, (active, inactive) in enumerate(loader):
        active.label, inactive.label  = active.label.to(device), inactive.label.to(device)
        active.neighbors, inactive.neighbors = active.neighbors.to(device), inactive.neighbors.to(device)
        mask1 = get_mask(active.neighbors)
        mask2 = get_mask(inactive.neighbors)
        out_active = transformer_model(active.neighbors, mask1)
        out_inactive = transformer_model(inactive.neighbors, mask2)
        output = ff_model(out_active, out_inactive)
        loss = criterion(output, active.label.float())
        losses.append(loss.item())
        y_true.extend(active.label.tolist())
        y_pred.extend(output.tolist())
        if it % print_frequency == 0:
            print(f'iter {it}, loss {np.mean(losses)}')
        if it == max_test_iter:
            break

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)

    return np.mean(losses), auroc, auprc


def train_noneqv_lep(ex, use_attention, data_dir, device, log_dir, checkpoint, num_epochs, batch_size,
              hidden_dim, learning_rate, workers, betas, eps, d_ff, d_atom,
              eta, max_radius, num_atoms, num_heads, max_train_iter, max_test_iter,
              print_frequency, test_mode=False):
    
    train_set = dl.LEPDataset(os.path.join(data_dir, 'pairs_train@10'), max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.LEPDataset(os.path.join(data_dir, 'pairs_val@10'), max_radius=max_radius)
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
    ff_model = LEPFeedForward(hidden_dim).to(device)

    model_parameters = filter(lambda p: p.requires_grad, transformer_model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters:', num_parameters)

    best_val_loss = 999
    best_val_auroc = 0

    params = [x for x in transformer_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
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
        print('Validating...')
        val_loss, auroc, auprc = test(
            transformer_model, 
            ff_model, 
            val_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)
        scheduler.step(val_loss)
        if auroc > best_val_auroc:
            torch.save({
                'epoch': epoch,
                'transformer_state_dict': transformer_model.state_dict(),
                'ff_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights.pt'))
            best_val_auroc = auroc
        elapsed = (time.time() - start)
        ex.log_scalar('val loss', val_loss)
        ex.log_scalar('val auroc', auroc)
        ex.log_scalar('val auprc', auprc)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val AUPRC {auprc}')

    if test_mode:
        test_set = dl.LEPDataset(os.path.join(data_dir, 'pairs_test@10'), max_radius=max_radius)
        test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        test_loss, auroc, auprc = test(
            transformer_model, 
            ff_model, 
            test_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)
        ex.log_scalar('test loss', test_loss)
        ex.log_scalar('test auroc', auroc)
        ex.log_scalar('test auprc', auprc)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test AUPRC {auprc}')
        return test_loss, auroc, auprc

    return best_val_loss