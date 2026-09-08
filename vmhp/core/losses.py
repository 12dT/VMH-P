from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def pearson_corr_loss(tree_dist: torch.Tensor, geom_dist: torch.Tensor) -> torch.Tensor:
    tree_flat = tree_dist.flatten()
    geom_flat = geom_dist.flatten()
    if tree_flat.numel() < 2:
        return geom_flat.sum() * 0.0
    if tree_flat.std(unbiased=False) < 1e-6 or geom_flat.std(unbiased=False) < 1e-6:
        return geom_flat.sum() * 0.0
    tree_centered = tree_flat - tree_flat.mean()
    geom_centered = geom_flat - geom_flat.mean()
    corr = (tree_centered * geom_centered).mean() / (
        tree_centered.std(unbiased=False).clamp_min(1e-6)
        * geom_centered.std(unbiased=False).clamp_min(1e-6)
    )
    return 1.0 - corr


def vertical_hierarchy_consistency_loss(model) -> torch.Tensor:
    labels = torch.arange(model.num_classes, device=model.vhsh.class_tangent_params.device)
    geom_dist = model.vhsh.pairwise_tangent_distance(model.vhsh.class_tangent_params)
    tree_dist = model.vhsh.label_tree_distances(labels).to(device=geom_dist.device, dtype=geom_dist.dtype)
    if labels.numel() < 2:
        return _zero_like(geom_dist)
    mask = torch.triu(torch.ones_like(geom_dist, dtype=torch.bool), diagonal=1)
    return pearson_corr_loss(tree_dist[mask], geom_dist[mask])


def vmhp_loss(
    final_logits: torch.Tensor,
    aux: Dict,
    labels: torch.Tensor,
    model=None,
    lambda_str: float = 0.10,
    lambda_vh: float = 0.10,
    lambda_hr: float = 0.01,
    class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    task = F.cross_entropy(final_logits, labels, weight=class_weight, label_smoothing=label_smoothing)
    structured = F.cross_entropy(
        aux["structured_logits"],
        labels,
        weight=class_weight,
        label_smoothing=label_smoothing,
    )
    vh = (
        vertical_hierarchy_consistency_loss(model)
        if model is not None and lambda_vh != 0
        else _zero_like(final_logits)
    )
    hr = model.harf.relation_regularization() if model is not None and lambda_hr != 0 else _zero_like(final_logits)
    total = task + float(lambda_str) * structured + float(lambda_vh) * vh + float(lambda_hr) * hr
    return {
        "loss": total,
        "loss_task": task.detach(),
        "loss_structured": structured.detach(),
        "loss_vh": vh.detach(),
        "loss_hr": hr.detach(),
    }
