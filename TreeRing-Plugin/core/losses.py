from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def pearson_corr_loss(tree_dist: torch.Tensor, hyp_dist: torch.Tensor) -> torch.Tensor:
    tree_flat = tree_dist.flatten()
    hyp_flat = hyp_dist.flatten()
    if tree_flat.numel() < 2 or tree_flat.std(unbiased=False) < 1e-6 or hyp_flat.std(unbiased=False) < 1e-6:
        return _zero_like(hyp_flat)
    tree_centered = tree_flat - tree_flat.mean()
    hyp_centered = hyp_flat - hyp_flat.mean()
    corr = (tree_centered * hyp_centered).mean() / (
        tree_centered.std(unbiased=False).clamp_min(1e-6) * hyp_centered.std(unbiased=False).clamp_min(1e-6)
    )
    return 1.0 - corr


def thc_loss(plugin, aux: Dict, labels: torch.Tensor) -> torch.Tensor:
    z_dict = aux.get("z", {})
    weights = aux.get("source_weights", {})
    tree_logits = aux["tree_logits"]
    if not z_dict or labels is None:
        return _zero_like(tree_logits)
    z_bar = None
    for name, z in z_dict.items():
        weight = weights[name].detach()
        if weight.dim() == 2 and weight.shape[-1] != 1:
            if labels is not None and int(labels.max().item()) < weight.shape[-1]:
                weight = weight.gather(1, labels.to(weight.device).view(-1, 1))
            else:
                weight = weight.mean(dim=-1, keepdim=True)
        term = weight * z
        z_bar = term if z_bar is None else z_bar + term
    evidence_dist = plugin.tree.pairwise_tangent_distance(z_bar)
    tree_dist = plugin.tree.label_tree_distances(labels).to(device=evidence_dist.device, dtype=evidence_dist.dtype)
    n = labels.numel()
    if n < 2:
        return _zero_like(tree_logits)
    mask = torch.triu(torch.ones(n, n, device=evidence_dist.device, dtype=torch.bool), diagonal=1)
    return pearson_corr_loss(tree_dist[mask], evidence_dist[mask])


def treering_loss(
    final_logits: torch.Tensor,
    aux: Dict,
    labels: torch.Tensor,
    plugin=None,
    lambda_tree: float = 0.5,
    lambda_plugin: float = 0.5,
    lambda_thc: float = 0.1,
    lambda_rel: float = 0.01,
    class_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    final = F.cross_entropy(final_logits, labels, weight=class_weight, label_smoothing=label_smoothing)
    tree = F.cross_entropy(aux["tree_logits"], labels, weight=class_weight, label_smoothing=label_smoothing)
    plugin_ce = F.cross_entropy(aux["plugin_logits"], labels, weight=class_weight, label_smoothing=label_smoothing)
    thc = thc_loss(plugin, aux, labels) if plugin is not None and lambda_thc != 0 else _zero_like(final_logits)
    rel = plugin.ring.relation_regularization() if plugin is not None and lambda_rel != 0 else _zero_like(final_logits)
    total = final + lambda_tree * tree + lambda_plugin * plugin_ce + lambda_thc * thc + lambda_rel * rel
    return {
        "loss": total,
        "loss_final": final.detach(),
        "loss_tree": tree.detach(),
        "loss_plugin": plugin_ce.detach(),
        "loss_thc": thc.detach(),
        "loss_relation": rel.detach(),
    }
