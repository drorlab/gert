from models.gert import make_model
from models.heads import EqvMSPFeedForward
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import get_mask, get_ff_mask
import data.msp.eqv_msp_dataloader as dl
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
    for it, (subunit1, subunit2) in enumerate(loader):
        subunit1.label, subunit2.label = subunit1.label.to(device), subunit2.label.to(device)
        subunit1.pos, subunit2.pos = subunit1.pos.to(device), subunit2.pos.to(device)
        if subunit1.pos.shape[1] + subunit2.pos.shape[1] > 850:
            print("Skipping because will be OOM")
            continue
        subunit1.elements, subunit2.elements = subunit1.elements.to(device), subunit2.elements.to(device)

        optimizer.zero_grad()

        mask1 = get_mask(subunit1.pos)
        out1 = transformer_model(subunit1.pos, subunit1.elements, attn_mask=mask1)
        mask2 = get_mask(subunit2.pos)
        out2 = transformer_model(subunit2.pos, subunit2.elements, attn_mask=mask2)
        ff_mask = get_ff_mask(subunit1.pos, subunit2.pos)
        output = ff_model(out1, out2, subunit1.pos, subunit2.pos, ff_mask)

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
            return np.mean(losses)

    return np.mean(losses)


@torch.no_grad()
def test(counts, transformer_model, ff_model, loader, criterion, device, max_test_iter, print_frequency):
    transformer_model.eval()
    ff_model.eval()

    losses = []
    y_true = []
    y_pred = []

    for it, (subunit1, subunit2) in enumerate(loader):
        subunit1.label, subunit2.label = subunit1.label.to(device), subunit2.label.to(device)
        subunit1.pos, subunit2.pos = subunit1.pos.to(device), subunit2.pos.to(device)
        subunit1.elements, subunit2.elements = subunit1.elements.to(device), subunit2.elements.to(device)

        mask1 = get_mask(subunit1.pos)
        out1 = transformer_model(subunit1.pos, subunit1.elements, attn_mask=mask1)
        mask2 = get_mask(subunit2.pos)
        out2 = transformer_model(subunit2.pos, subunit2.elements, attn_mask=mask2)
        ff_mask = get_ff_mask(subunit1.pos, subunit2.pos)
        output = ff_model(out1, out2, subunit1.pos, subunit2.pos, ff_mask)

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

    return np.mean(losses), auroc, auprc


def train_eqv_msp(ex, use_attention, labels_dir, data_dir, device, log_dir, checkpoint, num_epochs, batch_size, num_dense,
                  learning_rate, workers, betas, eps, encoder_layers, num_heads, lmax,
                  N_hidden, max_radius, number_of_basis, h, L, 
                  max_train_iter, max_test_iter, print_frequency, test_mode=False):
    
    train_set = dl.MSP_Dataset(os.path.join(data_dir, 'pairs_train@40'), max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.MSP_Dataset(os.path.join(data_dir, 'pairs_val@40'), max_radius=max_radius)
    val_loader = dl.DataLoader(val_set, batch_size=batch_size, num_workers=workers)

    Rs_in = [(len(PROT_ATOMS), 0)]
    Rs_hidden = [(N_hidden, l) for l in range(lmax)]
    Rs_ff = [(N_hidden * lmax, 0)]
    Rs_out = [(1, 0)]

    transformer_model = make_model(Rs_in, Rs_hidden, num_heads, num_dense, encoder_layers, 
                                    max_radius, number_of_basis, h, L, use_attention=use_attention).to(device)
    ff_model = EqvMSPFeedForward(Rs_hidden, Rs_ff, Rs_out, max_radius, number_of_basis, h, L).to(device)

    best_val_loss = 999

    params = [x for x in transformer_model.parameters()] + [x for x in ff_model.parameters()]

    criterion = nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)

    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        transformer_model.load_state_dict(cpt['transformer_model_state_dict'])

        print('Loaded model from checkpoint')

    print(f'Training for {num_epochs} epochs')
    print('---------------------------------')
    for epoch in range(1, num_epochs+1):
        start = time.time()
        train_loss = train(
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
        print('Validating...')
        val_loss, auroc, auprc = test(
            train_set.counts,
            transformer_model,
            ff_model,
            val_loader,
            criterion,
            device,
            max_test_iter,
            print_frequency)
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
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'\tTrain loss {train_loss}, Val loss {val_loss}, Val AUROC {auroc}, Val AUPRC {auprc}')

    if test_mode:
        test_set = dl.MSP_Dataset(os.path.join(data_dir, 'pairs_test@40'), max_radius=max_radius)
        test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
        test_file = os.path.join(log_dir, f'test_results.txt')
        cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
        transformer_model.load_state_dict(cpt['transformer_state_dict'])
        ff_model.load_state_dict(cpt['ff_state_dict'])
        test_loss, auroc, auprc = test(
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
        print(f'\tTest loss {test_loss}, Test AUROC {auroc}, Test AUPRC {auprc}')
        return test_loss, auroc, auprc

    return best_val_loss
