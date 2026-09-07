import torch
import torch.nn.functional as F


def _safe_row_l1_norm(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    row_sum = a.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > eps, a / row_sum.clamp_min(eps), torch.zeros_like(a))


def build_relation_inits(class_centers: torch.Tensor):
    centers = class_centers.detach()
    offdiag = torch.ones(centers.shape[0], centers.shape[0], device=centers.device, dtype=centers.dtype)
    offdiag = offdiag - torch.eye(centers.shape[0], device=centers.device, dtype=centers.dtype)
    direction = F.normalize(centers - centers.mean(dim=0, keepdim=True), dim=-1, eps=1e-12)
    rho = direction @ direction.t()
    pos = torch.relu(rho) * offdiag
    neg = torch.relu(-rho) * offdiag
    return _safe_row_l1_norm(pos), _safe_row_l1_norm(neg)
