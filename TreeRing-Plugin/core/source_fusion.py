import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(value: float) -> torch.Tensor:
    value = torch.tensor(float(value)).clamp_min(1e-6)
    return torch.log(torch.expm1(value))


class SourceReliability(nn.Module):
    def __init__(
        self,
        tree_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        classwise: bool = True,
        prob_bias_init: float = 0.25,
    ):
        super().__init__()
        tree_dim = int(tree_dim)
        hidden_dim = int(hidden_dim or max(4, tree_dim // 2))
        self.num_classes = int(num_classes)
        self.classwise = bool(classwise)
        self.encoder = nn.Sequential(
            nn.Linear(tree_dim + 2, hidden_dim),
            nn.GELU(),
        )
        self.global_scorer = nn.Linear(hidden_dim, 1)
        self.class_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.raw_prob_bias = nn.Parameter(_inv_softplus(prob_bias_init))

    @property
    def prob_bias(self) -> torch.Tensor:
        return F.softplus(self.raw_prob_bias)

    def _stats(self, z: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(float(self.num_classes))
        top2 = probs.topk(k=min(2, self.num_classes), dim=-1).values
        if self.num_classes == 1:
            margin = torch.ones_like(entropy)
        else:
            margin = top2[:, :1] - top2[:, 1:2]
        return torch.cat([z, 1.0 - entropy, margin], dim=-1)

    def forward(self, z_dict: Dict[str, torch.Tensor], prob_dict: Dict[str, torch.Tensor]):
        names = list(z_dict.keys())
        if len(names) == 1:
            shape = (z_dict[names[0]].shape[0], self.num_classes if self.classwise else 1)
            return {
                names[0]: torch.ones(
                    shape,
                    device=z_dict[names[0]].device,
                    dtype=z_dict[names[0]].dtype,
                )
            }

        global_logits = []
        class_logits = []
        for name in names:
            encoded = self.encoder(self._stats(z_dict[name], prob_dict[name]))
            global_logit = self.global_scorer(encoded)
            global_logits.append(global_logit)
            if self.classwise:
                class_logit = self.class_scorer(encoded)
                class_logit = class_logit + self.prob_bias * prob_dict[name].clamp_min(1e-8).log()
                class_logits.append(class_logit + global_logit)

        if self.classwise:
            stacked = torch.stack(class_logits, dim=1)
            weights = torch.softmax(stacked, dim=1)
            return {name: weights[:, idx, :] for idx, name in enumerate(names)}

        stacked = torch.stack(global_logits, dim=1)
        weights = torch.softmax(stacked, dim=1)
        return {name: weights[:, idx] for idx, name in enumerate(names)}
