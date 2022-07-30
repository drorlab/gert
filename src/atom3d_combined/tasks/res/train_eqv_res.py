from models.gert import make_model
from models.heads import EqvRESFeedForward
from utils import get_mask, get_acc, get_top_k_acc
from sklearn.metrics import f1_score
import data.res.eqv_res_dataloader as dl
import os
import time
import numpy as np
import torch
import torch.nn as nn


"""
Adapted from: https://github.com/drorlab/atom3d/blob/master/examples/pytorch_geometric/resdel_dataloader.py
"""

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
        subunit.label = subunit.label.to(device)
        subunit.elements = subunit.elements.to(device)
        optimizer.zero_grad()
        mask = get_mask(subunit.pos)
        out = transformer_model(subunit.pos, subunit.elements, attn_mask=mask)
        out = ff_model(out, subunit.pos)
        train_loss = criterion(out, subunit.label)
        losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        if it % print_frequency == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch}, iter {it}, train loss {np.mean(losses)}, avg it/sec {print_frequency / elapsed}')
            start = time.time()
        
        if it == max_train_iter:
            break

    return np.mean(losses)

@torch.no_grad()
def test(transformer_model, ff_model, loader, criterion, device, max_test_iter, print_frequency):
    transformer_model.eval()
    ff_model.eval()

    losses = []
    avg_acc = []
    avg_top_k_acc = []
    f1_list = []

    for i, subunit in enumerate(loader):
        subunit.pos = subunit.pos.to(device)
        subunit.label = subunit.label.to(device)
        subunit.elements = subunit.elements.to(device)
        mask = get_mask(subunit.pos)
        out = transformer_model(subunit.pos, subunit.elements, attn_mask=mask)
        out = ff_model(out, subunit.pos)
        train_loss = criterion(out, subunit.label)
        losses.append(train_loss.item())

        acc = get_acc(out, subunit.label)
        f1 = f1_score(torch.argmax(out, dim=-1).cpu(), subunit.label.cpu(), average='micro')
        top_k_acc = get_top_k_acc(out, subunit.label, k=3)
        avg_acc.append(acc)
        avg_top_k_acc.append(top_k_acc)
        f1_list.append(f1)

        if i % print_frequency == 0:
            print(f'Iter {i}, train loss {np.mean(losses)}')
        
        if i == max_test_iter:
            break 

    return np.mean(losses), np.mean(avg_acc), np.mean(avg_top_k_acc), np.mean(f1_list)


def train_eqv_res(ex, use_attention, data_dir, device, log_dir, checkpoint, num_epochs, batch_size, num_dense,
              learning_rate, workers, betas, eps, encoder_layers, num_heads, lmax, 
              N_hidden, max_radius, number_of_basis, h, L, max_train_iter, max_test_iter, 
              print_frequency, test_mode=False):
    train_set = dl.ResDel_Dataset(os.path.join(data_dir, 'train_envs@1000'), max_radius=max_radius)
    train_loader = dl.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    val_set = dl.ResDel_Dataset(os.path.join(data_dir, 'val_envs@100'), max_radius=max_radius)
    val_loader = dl.DataLoader(val_set, batch_size=batch_size, num_workers=workers)

    Rs_in = [(len(PROT_ATOMS), 0)]
    Rs_hidden = [(N_hidden, l) for l in range(lmax)]
    Rs_out = [(len(RES_LABEL), 0)]
    
    transformer_model = make_model(Rs_in, Rs_hidden, num_heads, num_dense, encoder_layers, 
                                    max_radius, number_of_basis, h, L, use_attention=use_attention).to(device)
    ff_model = EqvRESFeedForward(Rs_hidden, Rs_out, max_radius, number_of_basis, h, L).to(device)

    model_parameters = filter(lambda p: p.requires_grad, transformer_model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters:', num_parameters)
    
    params = [x for x in transformer_model.parameters()] + [x for x in ff_model.parameters()]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)
    criterion = nn.CrossEntropyLoss().to(device)

    if checkpoint:
        cpt = torch.load(checkpoint, map_location=device)
        transformer_model.load_state_dict(cpt['transformer_model_state_dict'])
        optimizer.load_state_dict(cpt['optimizer_state_dict'])
        print('Loaded model from checkpoint')

    best_val_loss = 999

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
        curr_val_loss, val_acc, val_top_k_acc, val_f1 = test(
            transformer_model, 
            ff_model, 
            val_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            torch.save({
                'epoch': epoch,
                'transformer_model_state_dict': transformer_model.state_dict(),
                'generator_model_state_dict': ff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, os.path.join(log_dir, f'checkpoint_epoch{epoch}.pt'))
        elapsed = (time.time() - start)

        ex.log_scalar('Val Loss', curr_val_loss) 
        ex.log_scalar('Val Accuracy', val_acc) 
        ex.log_scalar('Val T3 Accuracy', val_top_k_acc) 
        ex.log_scalar('Val F1 Score', val_f1)
        
        print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
        print(f'Val Loss: {curr_val_loss} \nVal Accuracy: {val_acc} \nVal T3 Accuracy: {val_top_k_acc} \nVal F1 Score: {val_f1}')

    if test_mode:
        print('Testing...')
        test_set = dl.ResDel_Dataset(os.path.join(data_dir, 'test_envs@100'), max_radius=max_radius)
        test_loader = dl.DataLoader(test_set, batch_size=batch_size, num_workers=workers)
        test_loss, test_acc, test_top_k_acc, test_f1 = test(
            transformer_model, 
            ff_model, 
            test_loader, 
            criterion,
            device,
            max_test_iter,
            print_frequency)
        print(f'Test Loss: {test_loss} \nTest Accuracy: {test_acc} \nTest T3 Accuracy: {test_top_k_acc} \nTest F1 Score: {test_f1}')
	
        ex.log_scalar('Test Loss:', test_loss) 
        ex.log_scalar('Test Accuracy:', test_acc)
        ex.log_scalar('Test T3 Accuracy:', test_top_k_acc)
        ex.log_scalar('Test F1 Score:', test_f1)

        return test_loss, test_acc, test_top_k_acc, test_f1

    return best_val_loss