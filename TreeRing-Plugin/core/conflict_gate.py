import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p: float) -> torch.Tensor:
    p = torch.tensor(float(p)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p / (1.0 - p))


def _inv_softplus(x: float) -> torch.Tensor:
    x = torch.tensor(float(x)).clamp_min(1e-6)
    return torch.log(torch.expm1(x))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1, keepdim=True)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1, keepdim=True)
    return 0.5 * (kl_pm + kl_qm)


class MonotonicConflictGate(nn.Module):
    def __init__(self, eta_init: float = 0.50, kappa_init: float = 8.0, mu_init: float = 0.35):
        super().__init__()
        self.raw_eta = nn.Parameter(_logit(eta_init))
        self.raw_kappa = nn.Parameter(_inv_softplus(kappa_init))
        self.raw_mu = nn.Parameter(_logit(mu_init))

    def forward(self, probs):
        names = list(probs.keys())
        first = probs[names[0]]
        batch, num_classes = first.shape
        if len(names) < 2:
            zero_global = first.new_zeros(batch, 1)
            zero_class = first.new_zeros(batch, num_classes)
            gate = torch.sigmoid(F.softplus(self.raw_kappa) * (zero_class - torch.sigmoid(self.raw_mu)))
            return {
                "global_conflict": zero_global,
                "class_disagree": zero_class,
                "conflict_score": zero_class,
                "gate": gate,
                "eta": torch.sigmoid(self.raw_eta).detach(),
                "kappa": F.softplus(self.raw_kappa).detach(),
                "mu": torch.sigmoid(self.raw_mu).detach(),
            }

        pair_js = []
        pair_delta = []
        for left, right in itertools.combinations(names, 2):
            pair_js.append(js_divergence(probs[left], probs[right]) / math.log(2.0))
            pair_delta.append((probs[left] - probs[right]).abs())
        global_conf = torch.stack(pair_js, dim=0).mean(dim=0).clamp(0.0, 1.0)
        class_delta = torch.stack(pair_delta, dim=0).mean(dim=0).clamp(0.0, 1.0)

        eta = torch.sigmoid(self.raw_eta)
        conflict_score = eta * global_conf + (1.0 - eta) * class_delta
        kappa = F.softplus(self.raw_kappa)
        mu = torch.sigmoid(self.raw_mu)
        gate = torch.sigmoid(kappa * (conflict_score - mu))
        return {
            "global_conflict": global_conf,
            "class_disagree": class_delta,
            "conflict_score": conflict_score,
            "gate": gate,
            "eta": eta.detach(),
            "kappa": kappa.detach(),
            "mu": mu.detach(),
        }
