"""DiT/Flux pruning: only keep prune_magnitude and its helper find_layers, as used by pruning_utils."""
import torch
import torch.nn as nn
from collections import OrderedDict


def find_layers(module, layers=[nn.Linear, nn.Conv2d], name=''):
    """Recursively find layers of the given types in a DiT/Transformer."""
    if type(module) in layers:
        return {name: module}
    res = OrderedDict()
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + '.' + name1 if name != '' else name1,
            )
        )
    return res


@torch.no_grad()
def prune_magnitude(args, model, target_modules, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """Prune DiT/Transformer by weight magnitude. Used by pruning_utils for SD3/Flux shared_ratio."""
    all_layers = find_layers(model)
    for name, module in all_layers.items():
        if any(name.endswith(target_name) for target_name in target_modules):
            W = module.weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        indices = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + indices, True)
            else:
                flat_metrics = W_metric.flatten()
                thresh = torch.kthvalue(flat_metrics, int(W.numel() * args.sparsity_ratio))[0]
                W_mask = (W_metric <= thresh)
            W[W_mask] = 0
