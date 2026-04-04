"""UNet pruning."""
import torch
import torch.nn as nn
from collections import OrderedDict
import logging


def find_layers(module, layers=[nn.Linear, nn.Conv2d], name=''):
    if type(module) in layers:
        return {name: module}
    res = OrderedDict()
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def check_sparsity(model, pruning_config, logger=None):
    if not logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
    logger.info("\n" + "="*50 + "\nComputing sparsity of target modules (by group)...\n" + "="*50)
    all_found_layers = find_layers(model)
    total_target_zeros, total_target_params = 0, 0
    for group_name in pruning_config:
        group_zeros, group_params = 0, 0
        prefix_to_check = group_name
        if group_name == 'down':
            prefix_to_check = 'down_blocks'
        elif group_name == 'up':
            prefix_to_check = 'up_blocks'
        elif group_name == 'mid':
            prefix_to_check = 'mid_block'
        for layer_name, layer_module in all_found_layers.items():
            if hasattr(layer_module, 'weight') and layer_module.weight is not None and layer_name.startswith(prefix_to_check):
                weight = layer_module.weight
                zeros = torch.sum(weight == 0).item()
                total = weight.numel()
                group_zeros += zeros
                group_params += total
        if group_params > 0:
            sparsity = 100 * group_zeros / group_params
            logger.info(
                f"  - group '{group_name}' (prefix: '{prefix_to_check}'): "
                f"sparsity = {sparsity:.4f}% ({group_zeros}/{group_params})"
            )
            total_target_zeros += group_zeros
            total_target_params += group_params
    if total_target_params > 0:
        total_sparsity = 100 * total_target_zeros / total_target_params
        logger.info(f"\n  [summary] total sparsity over target modules: {total_sparsity:.4f}%")
        return total_sparsity
    logger.info("No matching target modules found or parameter count is zero.")
    return 0.0


def prune_magnitude(model, pruning_config, target_modules):
    all_layers = find_layers(model)
    analysis_results = {}
    for group_name, group_threshold in pruning_config.items():
        if group_threshold <= 0:
            print(f"\n--- Skip group: '{group_name}' (threshold <= 0) ---")
            continue
        prefix_to_check = group_name
        if group_name == 'down':
            prefix_to_check = 'down_blocks'
        elif group_name == 'up':
            prefix_to_check = 'up_blocks'
        elif group_name == 'mid':
            prefix_to_check = 'mid_block'
        print(f"\n--- Processing group: '{group_name}' (manual threshold: {group_threshold}) ---")
        layers_in_group_count = 0
        for layer_name, layer_module in all_layers.items():
            if 'attn2' not in layer_name and hasattr(layer_module, 'weight') and layer_name.startswith(
                    prefix_to_check) and any(tm in layer_name for tm in target_modules):
                layers_in_group_count += 1
                original_weight = layer_module.weight.data.clone()
                if original_weight.numel() == 0:
                    continue
                mask = torch.abs(original_weight) > group_threshold
                layer_module.weight.data.mul_(mask)
                W_original, W_kept, W_removed = original_weight.float(), original_weight[mask], original_weight[~mask]
                if W_original.numel() > 0:
                    analysis_results[layer_name] = {
                        'sparsity': W_removed.numel() / W_original.numel(),
                        'threshold': group_threshold,
                        'mean_original': W_original.mean().item(), 'std_original': W_original.std().item(),
                        'mean_kept': W_kept.mean().item() if W_kept.numel() > 0 else 0,
                        'std_kept': W_kept.std().item() if W_kept.numel() > 0 else 0,
                    }
        if layers_in_group_count == 0:
            print(f"Warning: no target layers found in group '{group_name}' for '{target_modules}'.")
        else:
            print(f"Processed {layers_in_group_count} target layers in group '{group_name}'.")
    return model, analysis_results


def prune_by_ratio(model, ratio_config, target_modules):
    """Prune UNet by down/up/mid ratio."""
    all_layers = find_layers(model)
    analysis_results = {}
    print(f"\n[INFO] Using per-block ratio pruning strategy.")
    print(f"[INFO] Skipping all layers whose names contain 'attn2'.")
    for group_name, group_ratio in ratio_config.items():
        if not (0 <= group_ratio < 1):
            print(
                f"Warning: ratio {group_ratio:.4f} for group '{group_name}' is invalid "
                f"(should be in [0, 1)). Skipping this group."
            )
            continue
        if group_ratio == 0:
            print(f"\n--- Skip group: '{group_name}' (ratio is 0) ---")
            continue
        prefix_to_check = group_name
        if group_name == 'down':
            prefix_to_check = 'down_blocks'
        elif group_name == 'up':
            prefix_to_check = 'up_blocks'
        elif group_name == 'mid':
            prefix_to_check = 'mid_block'
        group_weights_abs = []
        layers_in_group = []
        for layer_name, layer_module in all_layers.items():
            if 'attn2' not in layer_name and hasattr(layer_module, 'weight') and \
                    layer_module.weight is not None and layer_name.startswith(prefix_to_check) and \
                    any(tm in layer_name for tm in target_modules):
                if layer_module.weight.data.numel() > 0:
                    group_weights_abs.append(layer_module.weight.data.abs().view(-1))
                    layers_in_group.append((layer_name, layer_module))
        if not group_weights_abs:
            print(f"Warning: no target layers found in group '{group_name}' for '{target_modules}'.")
            continue
        all_group_weights_flat = torch.cat(group_weights_abs)
        num_weights_in_group = all_group_weights_flat.numel()
        num_to_prune = int(num_weights_in_group * group_ratio)
        if num_to_prune == 0:
            print(f"\n--- Skip group: '{group_name}' (ratio {group_ratio:.4f} too small, prune count is 0) ---")
            continue
        threshold_k = max(1, num_to_prune)
        threshold_val_tensor = torch.kthvalue(all_group_weights_flat.clone(), threshold_k).values
        threshold = threshold_val_tensor.item()
        print(
            f"\n--- Processing group: '{group_name}' (target ratio: {group_ratio:.4f}, "
            f"prune count: {num_to_prune}/{num_weights_in_group}) ---"
        )
        print(f"  - Computed threshold: {threshold:.6f}")
        actual_pruned_count_group = 0
        for layer_name, layer_module in layers_in_group:
            original_weight = layer_module.weight.data.clone()
            mask = original_weight.abs() > threshold
            layer_module.weight.data.mul_(mask)
            W_original = original_weight.float()
            W_kept = original_weight[mask].float()
            W_removed = original_weight[~mask].float()
            num_removed = W_removed.numel()
            actual_pruned_count_group += num_removed
            if W_original.numel() > 0:
                sparsity = num_removed / W_original.numel()
                analysis_results[layer_name] = {
                    'sparsity': sparsity,
                    'threshold': threshold,
                    'mean_original': W_original.mean().item(),
                    'std_original': W_original.std().item(),
                    'mean_kept': W_kept.mean().item() if W_kept.numel() > 0 else 0,
                    'std_kept': W_kept.std().item() if W_kept.numel() > 0 else 0,
                }
        actual_ratio_group = actual_pruned_count_group / num_weights_in_group if num_weights_in_group > 0 else 0
        print(
            f"  - Actual pruning ratio: {actual_ratio_group:.4f} "
            f"({actual_pruned_count_group}/{num_weights_in_group})"
        )
    return model, analysis_results
