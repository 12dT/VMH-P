import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_row_l1_norm(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    row_sum = a.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > eps, a / row_sum.clamp_min(eps), torch.zeros_like(a))


class AffectiveRelationRing(nn.Module):
    def __init__(
        self,
        num_classes: int,
        delta_max: float = 0.05,
        refinement_mode: str = "multiplicative",
        refinement_clip: float = 5.0,
        gain_clip: float = 2.0,
        empty_row_mass: float = 0.05,
        empty_row_floor: float = 1e-3,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.delta_max = float(delta_max)
        self.refinement_mode = str(refinement_mode)
        if self.refinement_mode not in {"multiplicative", "multiplicative_gain", "additive"}:
            raise ValueError("refinement_mode must be 'multiplicative', 'multiplicative_gain', or 'additive'")
        self.refinement_clip = float(refinement_clip)
        self.gain_clip = float(gain_clip)
        self.empty_row_mass = float(empty_row_mass)
        self.empty_row_floor = float(empty_row_floor)
        self.register_buffer("A_pos_init", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("A_neg_init", torch.zeros(self.num_classes, self.num_classes))
        self.register_buffer("offdiag_mask", torch.ones(self.num_classes, self.num_classes) - torch.eye(self.num_classes))
        self.theta_pos = nn.Parameter(torch.zeros(self.num_classes, self.num_classes))
        self.theta_neg = nn.Parameter(torch.zeros(self.num_classes, self.num_classes))
        if self.refinement_mode == "multiplicative_gain":
            self.row_log_mass_pos = nn.Parameter(torch.zeros(self.num_classes))
            self.row_log_mass_neg = nn.Parameter(torch.zeros(self.num_classes))

    def build_from_centers(self, class_centers: torch.Tensor):
        centers = class_centers.to(device=self.A_pos_init.device, dtype=self.A_pos_init.dtype)
        centered = centers - centers.mean(dim=0, keepdim=True)
        direction = F.normalize(centered, p=2, dim=-1, eps=1e-12)
        rho = direction @ direction.t()
        pos = F.relu(rho) * self.offdiag_mask
        neg = F.relu(-rho) * self.offdiag_mask
        with torch.no_grad():
            self.A_pos_init.copy_(safe_row_l1_norm(pos))
            self.A_neg_init.copy_(safe_row_l1_norm(neg))
            self.theta_pos.zero_()
            self.theta_neg.zero_()
            self._reset_gain_params()

    def load_initial_relations(self, pos: torch.Tensor, neg: torch.Tensor):
        if pos.shape != self.A_pos_init.shape or neg.shape != self.A_neg_init.shape:
            raise ValueError("relation init shape mismatch")
        with torch.no_grad():
            self.A_pos_init.copy_(pos.to(device=self.A_pos_init.device, dtype=self.A_pos_init.dtype))
            self.A_neg_init.copy_(neg.to(device=self.A_neg_init.device, dtype=self.A_neg_init.dtype))
            self.theta_pos.zero_()
            self.theta_neg.zero_()
            self._reset_gain_params()

    def _reset_gain_params(self):
        if hasattr(self, "row_log_mass_pos"):
            self.row_log_mass_pos.zero_()
            self.row_log_mass_neg.zero_()

    def _multiplicative_refine(self, init: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        phi = torch.clamp(theta, min=-self.refinement_clip, max=self.refinement_clip)
        refined = init * torch.exp(phi) * self.offdiag_mask
        return safe_row_l1_norm(refined)

    def _empty_relaxed_base(self, init: torch.Tensor) -> torch.Tensor:
        row_sum = init.sum(dim=-1, keepdim=True)
        empty = (row_sum <= 1e-12).to(dtype=init.dtype)
        return init + empty * self.empty_row_floor * self.offdiag_mask

    def _row_mass(self, init: torch.Tensor, row_log_mass: torch.Tensor) -> torch.Tensor:
        row_sum = init.sum(dim=-1)
        base = torch.where(
            row_sum > 1e-12,
            torch.ones_like(row_sum),
            torch.full_like(row_sum, self.empty_row_mass),
        )
        gain = torch.exp(torch.clamp(row_log_mass, min=-self.gain_clip, max=self.gain_clip))
        return (base * gain).view(-1, 1)

    def _multiplicative_gain_refine(
        self,
        init: torch.Tensor,
        theta: torch.Tensor,
        row_log_mass: torch.Tensor,
    ) -> torch.Tensor:
        phi = torch.clamp(theta, min=-self.refinement_clip, max=self.refinement_clip)
        base = self._empty_relaxed_base(init)
        distribution = safe_row_l1_norm(base * torch.exp(phi) * self.offdiag_mask)
        return distribution * self._row_mass(init, row_log_mass) * self.offdiag_mask

    def _multiplicative_gain_prior(self, init: torch.Tensor) -> torch.Tensor:
        base = self._empty_relaxed_base(init)
        distribution = safe_row_l1_norm(base * self.offdiag_mask)
        row_sum = init.sum(dim=-1)
        mass = torch.where(
            row_sum > 1e-12,
            torch.ones_like(row_sum),
            torch.full_like(row_sum, self.empty_row_mass),
        ).view(-1, 1)
        return distribution * mass * self.offdiag_mask

    def _additive_refine(self, init: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        delta = self.delta_max * torch.tanh(theta)
        refined = F.relu(init + delta) * self.offdiag_mask
        return safe_row_l1_norm(refined)

    def get_relations(self):
        if self.refinement_mode == "multiplicative":
            return (
                self._multiplicative_refine(self.A_pos_init, self.theta_pos),
                self._multiplicative_refine(self.A_neg_init, self.theta_neg),
            )
        if self.refinement_mode == "multiplicative_gain":
            return (
                self._multiplicative_gain_refine(self.A_pos_init, self.theta_pos, self.row_log_mass_pos),
                self._multiplicative_gain_refine(self.A_neg_init, self.theta_neg, self.row_log_mass_neg),
            )
        return (
            self._additive_refine(self.A_pos_init, self.theta_pos),
            self._additive_refine(self.A_neg_init, self.theta_neg),
        )

    def relation_regularization(self) -> torch.Tensor:
        if self.refinement_mode == "multiplicative":
            pos, neg = self.get_relations()
            return (pos - self.A_pos_init).pow(2).mean() + (neg - self.A_neg_init).pow(2).mean()
        if self.refinement_mode == "multiplicative_gain":
            pos, neg = self.get_relations()
            pos_prior = self._multiplicative_gain_prior(self.A_pos_init)
            neg_prior = self._multiplicative_gain_prior(self.A_neg_init)
            return (pos - pos_prior).pow(2).mean() + (neg - neg_prior).pow(2).mean()
        return self.theta_pos.pow(2).mean() + self.theta_neg.pow(2).mean()
