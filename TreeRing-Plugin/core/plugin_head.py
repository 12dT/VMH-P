from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conflict_gate import _inv_softplus, _logit, MonotonicConflictGate
from .feature_adapter import UniversalFeatureAdapter
from .geometry import LorentzGeometry
from .hyperbolic_tree import HyperbolicEvidenceTree
from .packet import EvidencePacket
from .relation_ring import AffectiveRelationRing
from .source_fusion import SourceReliability


@dataclass
class TreeRingConfig:
    tree_dim: int = 64
    dropout: float = 0.1
    geometry_mode: str = "hyperbolic"
    curvature: float = 0.25
    max_tangent_norm: float = 3.0
    num_internal: int = 2
    beta: float = 0.30
    tau_r: float = 0.07
    gamma: float = 0.10
    tau_e: float = 0.10
    relation_delta_max: float = 0.05
    relation_refinement_mode: str = "multiplicative"
    relation_refinement_clip: float = 5.0
    relation_gain_clip: float = 2.0
    relation_empty_row_mass: float = 0.05
    relation_empty_row_floor: float = 1e-3
    alpha_pos_init: float = 0.20
    alpha_neg_init: float = 0.20
    conflict_eta_init: float = 0.50
    conflict_kappa_init: float = 8.0
    conflict_mu_init: float = 0.35
    classwise_source_reliability: bool = True
    source_prob_bias_init: float = 0.25
    use_evidence_modulation: bool = False
    evidence_alpha_pos_init: float = 0.05
    evidence_alpha_neg_init: float = 0.05
    horizontal_only: bool = False
    use_sample_residual_gate: bool = True
    sample_gate_init: float = 0.50
    enable_base_calibration: bool = False
    base_scale_init: float = 1.0
    plugin_gate_init: float = 0.01
    plugin_scale_init: float = 1.0


class TreeRingHead(nn.Module):
    def __init__(self, feature_dims: Dict[str, int], num_classes: int, cfg: Optional[TreeRingConfig] = None):
        super().__init__()
        self.cfg = cfg or TreeRingConfig()
        if self.cfg.geometry_mode not in {"hyperbolic", "euclidean"}:
            raise ValueError("geometry_mode must be 'hyperbolic' or 'euclidean'")
        if self.cfg.relation_refinement_mode not in {"multiplicative", "multiplicative_gain", "additive"}:
            raise ValueError("relation_refinement_mode must be 'multiplicative', 'multiplicative_gain', or 'additive'")
        self.num_classes = int(num_classes)
        self.feature_dims = {str(k): int(v) for k, v in feature_dims.items()}

        self.ufa = nn.ModuleDict(
            {
                name: UniversalFeatureAdapter(in_dim, self.cfg.tree_dim, self.cfg.dropout)
                for name, in_dim in self.feature_dims.items()
            }
        )
        self.geometry = LorentzGeometry(self.cfg.curvature, self.cfg.max_tangent_norm)
        self.tree = HyperbolicEvidenceTree(
            num_classes=self.num_classes,
            tree_dim=self.cfg.tree_dim,
            num_internal=self.cfg.num_internal,
            curvature=self.cfg.curvature,
            max_tangent_norm=self.cfg.max_tangent_norm,
            beta=self.cfg.beta,
            tau_r=self.cfg.tau_r,
            gamma=self.cfg.gamma,
            geometry_mode=self.cfg.geometry_mode,
        )
        self.ring = AffectiveRelationRing(
            self.num_classes,
            delta_max=self.cfg.relation_delta_max,
            refinement_mode=self.cfg.relation_refinement_mode,
            refinement_clip=self.cfg.relation_refinement_clip,
            gain_clip=self.cfg.relation_gain_clip,
            empty_row_mass=self.cfg.relation_empty_row_mass,
            empty_row_floor=self.cfg.relation_empty_row_floor,
        )
        self.conflict = MonotonicConflictGate(
            eta_init=self.cfg.conflict_eta_init,
            kappa_init=self.cfg.conflict_kappa_init,
            mu_init=self.cfg.conflict_mu_init,
        )
        self.source_rel = SourceReliability(
            self.cfg.tree_dim,
            self.num_classes,
            classwise=self.cfg.classwise_source_reliability,
            prob_bias_init=self.cfg.source_prob_bias_init,
        )

        self.raw_alpha_pos = nn.Parameter(_inv_softplus(self.cfg.alpha_pos_init))
        self.raw_alpha_neg = nn.Parameter(_inv_softplus(self.cfg.alpha_neg_init))
        self.raw_evidence_alpha_pos = nn.Parameter(_inv_softplus(self.cfg.evidence_alpha_pos_init))
        self.raw_evidence_alpha_neg = nn.Parameter(_inv_softplus(self.cfg.evidence_alpha_neg_init))
        self.raw_plugin_gate = nn.Parameter(_logit(self.cfg.plugin_gate_init))
        self.raw_plugin_scale = nn.Parameter(_inv_softplus(self.cfg.plugin_scale_init))
        self.raw_plugin_class_scale = nn.Parameter(torch.zeros(self.num_classes))
        self.plugin_class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.raw_base_class_scale = nn.Parameter(torch.zeros(self.num_classes))
        self.raw_base_scale = nn.Parameter(_inv_softplus(self.cfg.base_scale_init))
        self.base_class_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.sample_residual_gate = nn.Sequential(
            nn.Linear(8, max(8, self.cfg.tree_dim // 8)),
            nn.GELU(),
            nn.Linear(max(8, self.cfg.tree_dim // 8), 1),
        )
        with torch.no_grad():
            last = self.sample_residual_gate[-1]
            last.weight.zero_()
            last.bias.copy_(_logit(self.cfg.sample_gate_init))

    @property
    def alpha_pos(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha_pos)

    @property
    def alpha_neg(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha_neg)

    @property
    def evidence_alpha_pos(self) -> torch.Tensor:
        return F.softplus(self.raw_evidence_alpha_pos)

    @property
    def evidence_alpha_neg(self) -> torch.Tensor:
        return F.softplus(self.raw_evidence_alpha_neg)

    @property
    def plugin_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_plugin_gate)

    @property
    def plugin_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_plugin_scale)

    @property
    def base_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_base_scale)

    def collect_sources(self, packet: EvidencePacket) -> Dict[str, torch.Tensor]:
        sources = packet.sources()
        out = {}
        for name, value in sources.items():
            if name in self.ufa and value is not None:
                out[name] = value
        if not out:
            raise ValueError("EvidencePacket does not contain any source known to this TreeRingHead")
        return out

    def non_ufa_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("ufa."):
                yield param

    def initialize_from_class_centers(self, class_centers: torch.Tensor, topology: Optional[Dict] = None):
        self.tree.init_from_centers(class_centers, topology=topology)
        self.ring.build_from_centers(class_centers)

    def load_structure(self, structure: Dict):
        topology = structure.get("topology")
        if topology is not None:
            self.tree.set_topology(topology)
        if "class_centers" in structure and "node_init" not in structure:
            self.initialize_from_class_centers(structure["class_centers"], topology=topology)
        else:
            if "node_init" in structure and "prototype_init" in structure:
                self.tree.load_initial_state(
                    structure["node_init"],
                    structure["prototype_init"],
                    edge_cost=structure.get("edge_cost"),
                )
            if "relation_pos_init" in structure and "relation_neg_init" in structure:
                self.ring.load_initial_relations(structure["relation_pos_init"], structure["relation_neg_init"])

    def calibrate_base_logits(self, base_logits: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_base_calibration:
            return base_logits
        class_scale = F.softplus(self.raw_base_class_scale).view(1, -1)
        class_scale = class_scale / F.softplus(torch.zeros((), device=class_scale.device, dtype=class_scale.dtype))
        return self.base_scale * class_scale * base_logits + self.base_class_bias.view(1, -1)

    def compute_sample_residual_gate(
        self,
        base_logits: torch.Tensor,
        tree_logits: torch.Tensor,
        plugin_logits: torch.Tensor,
        conflict_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.use_sample_residual_gate:
            return torch.ones_like(plugin_logits)
        base_probs = F.softmax(base_logits, dim=-1)
        tree_probs = F.softmax(tree_logits, dim=-1)
        plugin_probs = F.softmax(plugin_logits, dim=-1)

        base_top2 = base_probs.topk(k=min(2, self.num_classes), dim=-1).values
        plugin_top2 = plugin_probs.topk(k=min(2, self.num_classes), dim=-1).values
        if self.num_classes == 1:
            base_margin = torch.ones_like(base_top2[:, :1])
            plugin_margin = torch.ones_like(plugin_top2[:, :1])
        else:
            base_margin = base_top2[:, :1] - base_top2[:, 1:2]
            plugin_margin = plugin_top2[:, :1] - plugin_top2[:, 1:2]

        base_entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        plugin_entropy = -(plugin_probs.clamp_min(1e-8) * plugin_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        norm = torch.log(torch.tensor(float(self.num_classes), device=base_probs.device, dtype=base_probs.dtype))
        base_conf = 1.0 - base_entropy / norm
        plugin_conf = 1.0 - plugin_entropy / norm

        feature = torch.stack(
            [
                base_probs,
                tree_probs,
                plugin_probs,
                (plugin_probs - base_probs).abs(),
                base_conf.expand_as(base_probs),
                plugin_conf.expand_as(base_probs),
                (plugin_margin - base_margin).expand_as(base_probs),
                conflict_gate,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.sample_residual_gate(feature).squeeze(-1))

    def modulate_evidence(
        self,
        z: torch.Tensor,
        prob: torch.Tensor,
        A_pos: torch.Tensor,
        A_neg: torch.Tensor,
        conflict_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cfg.use_evidence_modulation:
            return z
        prototypes = self.tree.class_tangent_params
        pos_mix = prob @ A_pos
        neg_mix = prob @ A_neg
        pos_vec = pos_mix @ prototypes
        neg_vec = neg_mix @ prototypes
        conflict_strength = (1.0 - conflict_gate).mean(dim=-1, keepdim=True)
        z_mod = z + self.evidence_alpha_pos * pos_vec - self.evidence_alpha_neg * conflict_strength * neg_vec
        norm = z_mod.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        max_norm = torch.as_tensor(self.cfg.max_tangent_norm, device=z_mod.device, dtype=z_mod.dtype)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return z_mod * scale

    def export_structure(self) -> Dict:
        return {
            "topology": self.tree.export_topology(),
            "node_init": self.tree.node_tangent_params.detach().cpu(),
            "prototype_init": self.tree.class_tangent_params.detach().cpu(),
            "edge_cost": self.tree.edge_cost.detach().cpu(),
            "relation_pos_init": self.ring.A_pos_init.detach().cpu(),
            "relation_neg_init": self.ring.A_neg_init.detach().cpu(),
            "relation_refinement_mode": self.ring.refinement_mode,
        }

    def forward(self, packet: EvidencePacket, use_plugin: bool = True):
        if not use_plugin:
            return packet.base_logits, {"plugin_disabled": True}

        source_feats = self.collect_sources(packet)
        z, x, s_tree, probs = {}, {}, {}, {}
        for name, h in source_feats.items():
            z[name] = self.ufa[name](h)
            x[name] = self.tree.point_from_tangent(z[name])
            s_tree[name] = self.tree(x[name])
            probs[name] = F.softmax(s_tree[name] / max(self.cfg.tau_e, 1e-6), dim=-1)

        conf_info = self.conflict(probs)
        g_conf = conf_info["gate"]
        A_pos, A_neg = self.ring.get_relations()
        calibrated_base_logits = self.calibrate_base_logits(packet.base_logits)

        if self.cfg.use_evidence_modulation:
            mod_z, mod_x, mod_s_tree, mod_probs = {}, {}, {}, {}
            for name, prob in probs.items():
                mod_z[name] = self.modulate_evidence(z[name], prob, A_pos, A_neg, g_conf)
                mod_x[name] = self.tree.point_from_tangent(mod_z[name])
                mod_s_tree[name] = self.tree(mod_x[name])
                mod_probs[name] = F.softmax(mod_s_tree[name] / max(self.cfg.tau_e, 1e-6), dim=-1)
            z, x, s_tree, probs = mod_z, mod_x, mod_s_tree, mod_probs

        source_weights = self.source_rel(z, probs)
        if self.cfg.horizontal_only:
            seed_prob = F.softmax(calibrated_base_logits / max(self.cfg.tau_e, 1e-6), dim=-1)
            e_pos = seed_prob @ A_pos.t()
            e_neg = seed_prob @ A_neg.t()
            horizontal_delta = self.alpha_pos * e_pos - self.alpha_neg * (1.0 - g_conf) * e_neg
            tree_logits = calibrated_base_logits
            plugin_logits = calibrated_base_logits + horizontal_delta

            class_scale = F.softplus(self.raw_plugin_class_scale).view(1, -1)
            class_scale = class_scale / F.softplus(torch.zeros((), device=class_scale.device, dtype=class_scale.dtype))
            residual_logits = self.plugin_scale * class_scale * horizontal_delta + self.plugin_class_bias.view(1, -1)
            residual_gate = self.compute_sample_residual_gate(
                calibrated_base_logits,
                tree_logits,
                plugin_logits,
                g_conf,
            )
            final_logits = calibrated_base_logits + self.plugin_gate * residual_gate * residual_logits
            aux = {
                "tree_logits": tree_logits,
                "plugin_logits": plugin_logits,
                "residual_logits": residual_logits,
                "residual_gate": residual_gate,
                "calibrated_base_logits": calibrated_base_logits,
                "source_weights": source_weights,
                "source_probs": probs,
                "conflict_gate": g_conf,
                "conflict_info": conf_info,
                "A_pos": A_pos,
                "A_neg": A_neg,
                "plugin_gate": self.plugin_gate.detach(),
                "plugin_scale": self.plugin_scale.detach(),
                "plugin_class_scale": class_scale.detach(),
                "plugin_class_bias": self.plugin_class_bias.detach(),
                "base_scale": self.base_scale.detach(),
                "base_class_bias": self.base_class_bias.detach(),
                "source_prob_bias": self.source_rel.prob_bias.detach(),
                "alpha_pos": self.alpha_pos.detach(),
                "alpha_neg": self.alpha_neg.detach(),
                "evidence_alpha_pos": self.evidence_alpha_pos.detach(),
                "evidence_alpha_neg": self.evidence_alpha_neg.detach(),
                "z": z,
                "s_tree_by_source": s_tree,
                "s_pure_by_source": {"horizontal": plugin_logits},
            }
            return final_logits, aux

        s_pure = {}
        for name, prob in probs.items():
            e_pos = prob @ A_pos.t()
            e_neg = prob @ A_neg.t()
            s_pure[name] = s_tree[name] + self.alpha_pos * e_pos - self.alpha_neg * (1.0 - g_conf) * e_neg

        tree_logits = None
        plugin_logits = None
        for name in s_tree:
            weighted_tree = source_weights[name] * s_tree[name]
            weighted_plugin = source_weights[name] * s_pure[name]
            tree_logits = weighted_tree if tree_logits is None else tree_logits + weighted_tree
            plugin_logits = weighted_plugin if plugin_logits is None else plugin_logits + weighted_plugin

        class_scale = F.softplus(self.raw_plugin_class_scale).view(1, -1)
        class_scale = class_scale / F.softplus(torch.zeros((), device=class_scale.device, dtype=class_scale.dtype))
        residual_logits = self.plugin_scale * class_scale * plugin_logits + self.plugin_class_bias.view(1, -1)
        residual_gate = self.compute_sample_residual_gate(
            calibrated_base_logits,
            tree_logits,
            plugin_logits,
            g_conf,
        )
        final_logits = calibrated_base_logits + self.plugin_gate * residual_gate * residual_logits
        aux = {
            "tree_logits": tree_logits,
            "plugin_logits": plugin_logits,
            "residual_logits": residual_logits,
            "residual_gate": residual_gate,
            "calibrated_base_logits": calibrated_base_logits,
            "source_weights": source_weights,
            "source_probs": probs,
            "conflict_gate": g_conf,
            "conflict_info": conf_info,
            "A_pos": A_pos,
            "A_neg": A_neg,
            "plugin_gate": self.plugin_gate.detach(),
            "plugin_scale": self.plugin_scale.detach(),
            "plugin_class_scale": class_scale.detach(),
            "plugin_class_bias": self.plugin_class_bias.detach(),
            "base_scale": self.base_scale.detach(),
            "base_class_bias": self.base_class_bias.detach(),
            "source_prob_bias": self.source_rel.prob_bias.detach(),
            "alpha_pos": self.alpha_pos.detach(),
            "alpha_neg": self.alpha_neg.detach(),
            "evidence_alpha_pos": self.evidence_alpha_pos.detach(),
            "evidence_alpha_neg": self.evidence_alpha_neg.detach(),
            "z": z,
            "s_tree_by_source": s_tree,
            "s_pure_by_source": s_pure,
        }
        return final_logits, aux
