import torch
import pandas as pd

def get_mask(x):
    xyz = x[..., :3]
    r = torch.norm(xyz, 2, dim=-1, keepdim=False)
    pad_mask = r != 0
    pad_mask = pad_mask.unsqueeze(-1) * pad_mask.unsqueeze(-2)

    return pad_mask


def get_ff_mask(x1, x2):
    xyz = torch.cat((x1[..., :3], x2[..., :3]), dim=1)
    r = torch.norm(xyz, 2, dim=-1, keepdim=False)
    mask = r != 0
    return mask


def compute_global_correlations(results):
    # Save metrics.
    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')
    return res


def compute_global_correlations_mod(results):
    per_target = []
    for key, val in results.groupby(['target']):
        # Ignore target with 2 decoys only since the correlations are
        # not really meaningful.
        if val.shape[0] < 3:
            continue
        true = val['true'].astype(float)
        pred = val['pred'].astype(float)
        pearson = true.corr(pred, method='pearson')
        kendall = true.corr(pred, method='kendall')
        spearman = true.corr(pred, method='spearman')
        per_target.append((key, pearson, kendall, spearman))
    per_target = pd.DataFrame(
        data=per_target,
        columns=['target', 'pearson', 'kendall', 'spearman'])

    # Save metrics.
    res = {}
    all_true = results['true'].astype(float)
    all_pred = results['pred'].astype(float)
    res['all_pearson'] = all_true.corr(all_pred, method='pearson')
    res['all_kendall'] = all_true.corr(all_pred, method='kendall')
    res['all_spearman'] = all_true.corr(all_pred, method='spearman')

    res['per_target_mean_pearson'] = per_target['pearson'].mean()
    res['per_target_mean_kendall'] = per_target['kendall'].mean()
    res['per_target_mean_spearman'] = per_target['spearman'].mean()

    res['per_target_median_pearson'] = per_target['pearson'].median()
    res['per_target_median_kendall'] = per_target['kendall'].median()
    res['per_target_median_spearman'] = per_target['spearman'].median()
    return res


def get_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == label).sum(-1)) / label.size()[0]
    return acc


def get_top_k_acc(output, target, k=3):
    """Computes the accuracy over the k top predictions for the specified value of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = torch.topk(output, k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(1.0 / batch_size).item()