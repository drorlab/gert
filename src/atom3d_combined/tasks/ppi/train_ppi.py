from models.gert_noeqv import make_model
from models.heads import PPIFeedForward
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import get_mask
import data.ppi.ppi_dataloader as dl
import data.ppi.noneqv_db5_dataloader as test_dl
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
    total = 0
    for it, (subunit1, subunit2) in enumerate(loader):
        tick = time.time()
        subunit1.label, subunit2.label = subunit1.label.to(device), subunit2.label.to(device)
        subunit1.pos, subunit2.pos = subunit1.pos.to(device), subunit2.pos.to(device)
        optimizer.zero_grad()

        mask1 = get_mask(subunit1.pos)
        out1 = transformer_model(subunit1.pos, mask1)
        mask2 = get_mask(subunit2.pos)
        out2 = transformer_model(subunit2.pos, mask2)
        output = ff_model(out1, out2)
        loss = criterion(output, subunit1.label.float())
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
    total = 0

    y_true = []
    y_pred = []
    for it, (subunit1, subunit2) in enumerate(loader):
        if subunit1.pos.shape[1] == 1:
            continue
        subunit1.label, subunit2.label = subunit1.label.to(device), subunit2.label.to(device)
        subunit1.pos, subunit2.pos = subunit1.pos.to(device), subunit2.pos.to(device)
        mask1 = get_mask(subunit1.pos)
        out1 = transformer_model(subunit1.pos, mask1)
        mask2 = get_mask(subunit2.pos)
        out2 = transformer_model(subunit2.pos, mask2)
        output = ff_model(out1, out2)
        loss = criterion(output, subunit1.label.float())
        losses.append(loss.item())
        y_true.extend(subunit1.label.tolist())
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


def train_noneqv_ppi(ex, use_attention, data_dir, device, log_dir, checkpoint, num_epochs, batch_size,
              hidden_dim, learning_rate, workers, betas, eps, d_ff, d_atom,
              eta, max_radius, num_atoms, num_heads, max_train_iter, max_test_iter,
              print_frequency, test_mode=False):
    train_set = dl.PPI_Dataset(os.path.join(data_dir, 'pairs_pruned_train@1000'), max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.PPI_Dataset(os.path.join(data_dir, 'pairs_pruned_val@1000'), max_radius=max_radius)
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
    ff_model = PPIFeedForward(hidden_dim).to(device)

    model_parameters = filter(lambda p: p.requires_grad, transformer_model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters:', num_parameters)
    
    best_val_loss = 999
    
    params = [x for x in transformer_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.BCELoss().to(device)
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
        val_loss, auroc, auprc = test(
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
        elapsed = (time.time() - start)
        ex.log_scalar('Validation Loss', val_loss)
        ex.log_scalar('Validation AUROC', auroc)
        ex.log_scalar('Validation AUPRC', auprc)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val AUPRC {auprc}')


    if test_mode:
        print("Testing on DIPS...")
        test_set = dl.PPI_Dataset(os.path.join(data_dir, 'pairs_pruned_test@1000'), max_radius=max_radius)
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
        ex.log_scalar('Test Loss', test_loss)
        ex.log_scalar('Test AUROC', auroc)
        ex.log_scalar('Test AUPRC', auprc)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test AUPRC {auprc}')

        print("Testing on DB5...")
        db5_dir = '/oak/stanford/groups/rondror/projects/atom3d/protein_protein_interfaces/DB5/'
        test_set = test_dl.PPI_Dataset(os.path.join(db5_dir, 'pairs@10'), max_radius=max_radius)
        test_loader = test_dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
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
        ex.log_scalar('Test Loss', test_loss)
        ex.log_scalar('Test AUROC', auroc)
        ex.log_scalar('Test AUPRC', auprc)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test AUPRC {auprc}')
        return test_loss, auroc, auprc
    else:
        return best_val_loss