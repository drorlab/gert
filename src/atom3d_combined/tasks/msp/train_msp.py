from models.gert_noeqv import make_model
from models.heads import MSPFeedForward
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import get_mask
import data.msp.msp_dataloader as dl
import os
import time
import numpy as np
import torch
import torch.nn as nn

PROT_ATOMS = ('C', 'O', 'N', 'S', 'P')
RES_LABEL = ('LEU', 'ILE', 'VAL', 'TYR', 'ARG', 'GLU', 'PHE', 'ASP', 'THR', 'LYS', 
             'ALA', 'GLY', 'TRP', 'SER', 'PRO', 'ASN', 'GLN', 'HIS', 'MET', 'CYS')

def train(counts, epoch, transformer_model, ff_model, loader, criterion, optimizer, device, max_train_iter, print_frequency):
    transformer_model.train()
    ff_model.train()

    start = time.time()

    losses = []
    total = 0

    # for the purposes of logging
    y_true = []
    y_pred = []

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

        y_true.extend(subunit1.label.tolist())
        y_pred.extend(output.tolist())

        # overwriting the prev. criterion
        weights = torch.ones(output.shape)
        labels = subunit1.label.float()

        weights[labels == 0] = counts[1] / counts[0]  # factor to multiply label = 0 weights by
        criterion = nn.BCELoss(weight=weights).to(device)

        loss = criterion(output, subunit1.label.float())
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        
        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()
        if it == max_train_iter:
           break

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    correct = (y_true == (y_pred > 0.5))
    accuracy = float(correct.sum()) / len(y_true)

    return np.mean(losses), auroc, auprc, accuracy


@torch.no_grad()
def test(counts, transformer_model, ff_model, loader, criterion, device, max_test_iter, print_frequency):
    transformer_model.eval()
    ff_model.eval()

    losses = []
    total = 0

    y_true = []
    y_pred = []

    for it, (subunit1, subunit2) in enumerate(loader):
        subunit1.label, subunit2.label = subunit1.label.to(device), subunit2.label.to(device)
        subunit1.pos, subunit2.pos = subunit1.pos.to(device), subunit2.pos.to(device)
        
        mask1 = get_mask(subunit1.pos) 
        out1 = transformer_model(subunit1.pos, mask1)
        mask2 = get_mask(subunit2.pos)
        out2 = transformer_model(subunit2.pos, mask2)
        output = ff_model(out1, out2)

        weights = torch.ones(output.shape)
        labels = subunit1.label.float()
        weights[labels == 0] = counts[1] / counts[0]  # factor to multiply label = 0 weights by
        criterion = nn.BCELoss(weight=weights).to(device)
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
    correct = (y_true == (y_pred > 0.5))
    accuracy = float(correct.sum()) / len(y_true)

    return np.mean(losses), auroc, auprc, accuracy


def get_mask(x):
    r = torch.norm(x[..., :3], 2, dim=-1, keepdim=False)
    mask = r != 0
    mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    return mask


def start_output_datadict(output_datadict, model_params, num_epochs, batch_size, learning_rate, betas, eps, test_mode=False):
    datadict_keys = ['mode', 'epoch', 'loss', 'auroc', 'auprc']
    
    for key in datadict_keys:
        output_datadict[key] = []

    if test_mode:
        n = num_epochs + 1
    else:
        n = num_epochs

    for key in model_params:
        output_datadict[key] = [model_params[key]] * n

    output_datadict['batch_size'] = [batch_size] * n
    output_datadict['learning_rate'] = [learning_rate] * n
    output_datadict['betas'] = [betas] * n
    output_datadict['eps'] = [eps] * n

def update_output_datadict(output_datadict, epoch, val_loss, auroc, auprc, test_mode=False):
    output_datadict['epoch'].extend([epoch])
    output_datadict['loss'].extend([val_loss])
    output_datadict['auroc'].extend([auroc])
    output_datadict['auprc'].extend([auprc])
    if test_mode:
        output_datadict['mode'].extend(['test'])
    else:
        output_datadict['mode'].extend(['val'])


def train_noneqv_msp(ex, use_attention, labels_dir, data_dir, device, log_dir, checkpoint, num_epochs, batch_size,
              hidden_dim, learning_rate, workers, betas, eps, d_ff, d_atom,
              eta, max_radius, num_atoms, num_heads, max_train_iter, max_test_iter,
              print_frequency, test_mode=False):
    # Create the data set
    train_set = dl.MSP_Dataset(os.path.join(data_dir, 'pairs_train@40'), max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.MSP_Dataset(os.path.join(data_dir, 'pairs_val@40'), max_radius=max_radius)
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
    ff_model = MSPFeedForward(hidden_dim).to(device)

    model_parameters = filter(lambda p: p.requires_grad, transformer_model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters:', num_parameters)

    best_val_loss = 999

    params = [x for x in transformer_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=3,
                                                           min_lr=0.00001)

    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        transformer_model.load_state_dict(cpt['transformer_state_dict'])

        ff_model.load_state_dict(cpt['ff_state_dict'])
        optimizer.load_state_dict(cpt['optimizer_state_dict'])

        print('Loaded model from checkpoint!')

    print(f'Training for {num_epochs} epochs')
    print('---------------------------------')
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss, auroc, auprc, acc = train(
            train_set.counts,
            epoch, 
            transformer_model, 
            ff_model, 
            train_loader, 
            criterion, 
            optimizer,
            device,
            max_train_iter, 
            print_frequency)
        ex.log_scalar('Train loss', train_loss)
        ex.log_scalar('Train AUROC', auroc)
        ex.log_scalar('Train AUPRC', auprc)
        ex.log_scalar('Train Accuracy', acc)
        print('Validating...')
        val_loss, auroc, auprc, acc = test(
            train_set.counts,
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

        ex.log_scalar('Validation loss', val_loss)
        ex.log_scalar('Validation AUROC', auroc)
        ex.log_scalar('Validation AUPRC', auprc)
        ex.log_scalar('Validation Accuracy', acc)
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val AUPRC {auprc}')

    if test_mode:
        test_set = dl.MSP_Dataset(os.path.join(data_dir, 'pairs_test@40'), max_radius=max_radius)
        test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
        test_file = os.path.join(log_dir, f'test_results.txt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        test_loss, auroc, auprc, acc = test(
            train_set.counts, 
            transformer_model, 
            ff_model, 
            test_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)
        ex.log_scalar('Test loss', test_loss)
        ex.log_scalar('Test AUROC', auroc)
        ex.log_scalar('Test AUPRC', auprc)
        ex.log_scalar('Test Accuracy', acc)
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test AUPRC {auprc}')
        return test_loss, auroc, auprc
    else:
        return best_val_loss