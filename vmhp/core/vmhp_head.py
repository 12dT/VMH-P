from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_adapter import UniversalFeatureAdapter
from .geometry import LorentzGeometry
from .horizontal_relation_field import HorizontalAffectiveRelationField
from .packet import EvidencePacket
from .vertical_hierarchy import VerticalHyperbolicSentimentHierarchy


@dataclass
class VMHPConfig:
    tree_dim: int = 64
    dropout: float = 0.10
    geometry_mode: str = "hyperbolic"
    curvature: float = 0.25
    max_tangent_norm: float = 3.0
    num_internal: int = 2
    beta: float = 0.30
    tau_r: float = 0.07
    edge_cost_weight: float = 0.10
    tau_e: float = 0.10
    tau_s: float = 0.10
    tau_g: float = 0.10
    eta: float = 0.50
    relation_delta_max: float = 0.05
    relation_refinement_mode: str = "multiplicative"
    relation_refinement_clip: float = 5.0
    relation_gain_clip: float = 2.0
    relation_empty_row_mass: float = 0.05
    relation_empty_row_floor: float = 1e-3
    residual_scale_init: float = 0.0
    horizontal_only: bool = False

    @classmethod
    def from_dict(cls, values: Optional[Dict]) -> "VMHPConfig":
        values = dict(values or {})
        if "gamma" in values and "edge_cost_weight" not in values:
            values["edge_cost_weight"] = values["gamma"]
        valid = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in valid})

    def to_dict(self) -> Dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _normalized_entropy(logits: torch.Tensor, tau: float, eps: float = 1e-8) -> torch.Tensor:
    prob = F.softmax(logits / max(float(tau), eps), dim=-1)
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
    norm = torch.log(torch.as_tensor(float(logits.shape[-1]), device=logits.device, dtype=logits.dtype)).clamp_min(eps)
    return entropy / norm


class VMHPHead(nn.Module):
    def __init__(self, feature_dims: Dict[str, int], num_classes: int, cfg: Optional[VMHPConfig] = None):
        super().__init__()
        self.cfg = cfg or VMHPConfig()
        if self.cfg.geometry_mode not in {"hyperbolic", "euclidean"}:
            raise ValueError("geometry_mode must be 'hyperbolic' or 'euclidean'")
        if self.cfg.relation_refinement_mode not in {"multiplicative", "multiplicative_gain", "additive"}:
            raise ValueError("relation_refinement_mode must be 'multiplicative', 'multiplicative_gain', or 'additive'")

        self.num_classes = int(num_classes)
        self.feature_dims = {str(name): int(dim) for name, dim in feature_dims.items()}

        self.ufa = nn.ModuleDict(
            {
                name: UniversalFeatureAdapter(in_dim, self.cfg.tree_dim, self.cfg.dropout)
                for name, in_dim in self.feature_dims.items()
            }
        )
        self.geometry = LorentzGeometry(self.cfg.curvature, self.cfg.max_tangent_norm)
        self.vhsh = VerticalHyperbolicSentimentHierarchy(
            num_classes=self.num_classes,
            tree_dim=self.cfg.tree_dim,
            num_internal=self.cfg.num_internal,
            curvature=self.cfg.curvature,
            max_tangent_norm=self.cfg.max_tangent_norm,
            beta=self.cfg.beta,
            tau_r=self.cfg.tau_r,
            gamma=self.cfg.edge_cost_weight,
            geometry_mode=self.cfg.geometry_mode,
        )
        self.harf = HorizontalAffectiveRelationField(
            self.num_classes,
            delta_max=self.cfg.relation_delta_max,
            refinement_mode=self.cfg.relation_refinement_mode,
            refinement_clip=self.cfg.relation_refinement_clip,
            gain_clip=self.cfg.relation_gain_clip,
            empty_row_mass=self.cfg.relation_empty_row_mass,
            empty_row_floor=self.cfg.relation_empty_row_floor,
        )

        self.residual_scale = nn.Parameter(torch.as_tensor(float(self.cfg.residual_scale_init)))

    @property
    def tree(self):
        return self.vhsh

    @property
    def ring(self):
        return self.harf

    def collect_sources(self, packet: EvidencePacket) -> Dict[str, torch.Tensor]:
        out = {}
        for name, value in packet.sources().items():
            if name in self.ufa and value is not None:
                out[name] = value
        if not out:
            raise ValueError("EvidencePacket does not contain any source known to this VMHPHead")
        return out

    def adapter_parameters(self):
        return self.ufa.parameters()

    def non_adapter_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("ufa."):
                yield param

    def non_ufa_parameters(self):
        return self.non_adapter_parameters()

    def initialize_from_class_centers(self, class_centers: torch.Tensor, topology: Optional[Dict] = None):
        self.vhsh.init_from_centers(class_centers, topology=topology)
        self.harf.build_from_centers(class_centers)

    def load_structure(self, structure: Dict):
        topology = structure.get("topology")
        if topology is not None:
            self.vhsh.set_topology(topology)
        if "class_centers" in structure and "node_init" not in structure:
            self.initialize_from_class_centers(structure["class_centers"], topology=topology)
        else:
            if "node_init" in structure and "prototype_init" in structure:
                self.vhsh.load_initial_state(
                    structure["node_init"],
                    structure["prototype_init"],
                    edge_cost=structure.get("edge_cost"),
                )
            if "relation_pos_init" in structure and "relation_neg_init" in structure:
                self.harf.load_initial_relations(structure["relation_pos_init"], structure["relation_neg_init"])

    def export_structure(self) -> Dict:
        return {
            "topology": self.vhsh.export_topology(),
            "node_init": self.vhsh.node_tangent_params.detach().cpu(),
            "prototype_init": self.vhsh.class_tangent_params.detach().cpu(),
            "edge_cost": self.vhsh.edge_cost.detach().cpu(),
            "relation_pos_init": self.harf.A_pos_init.detach().cpu(),
            "relation_neg_init": self.harf.A_neg_init.detach().cpu(),
            "relation_refinement_mode": self.harf.refinement_mode,
        }

    def _relation_matrix(self) -> Dict[str, torch.Tensor]:
        A_pos, A_neg = self.harf.get_relations()
        relation = A_pos - A_neg
        return {"A_pos": A_pos, "A_neg": A_neg, "relation": relation}

    def _confidence_gate(self, base_logits: torch.Tensor, structured_logits: torch.Tensor) -> torch.Tensor:
        base_entropy = _normalized_entropy(base_logits, self.cfg.tau_e)
        structured_entropy = _normalized_entropy(structured_logits, self.cfg.tau_e)
        return torch.sigmoid((base_entropy - structured_entropy) / max(float(self.cfg.tau_g), 1e-8))

    def _single_source_horizontal(self, seed_logits: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(seed_logits / max(float(self.cfg.tau_e), 1e-8), dim=-1)
        return seed_logits + float(self.cfg.eta) * (prob @ relation)

    def forward(self, packet: EvidencePacket, use_vmhp: bool = True, use_plugin: Optional[bool] = None):
        if use_plugin is not None:
            use_vmhp = bool(use_plugin)
        if not use_vmhp:
            return packet.base_logits, {"vmhp_disabled": True}

        base_logits = packet.base_logits
        relation_info = self._relation_matrix()
        relation = relation_info["relation"]

        if self.cfg.horizontal_only:
            structured_logits = self._single_source_horizontal(base_logits, relation)
            vertical_logits = base_logits
            source_weights = {"base": torch.ones_like(base_logits[:, :1])}
            source_confidence = {"base": 1.0 - _normalized_entropy(structured_logits, self.cfg.tau_e)}
            z = {}
            vertical_by_source = {"base": vertical_logits}
            relation_by_source = {"base": structured_logits}
        else:
            source_feats = self.collect_sources(packet)
            z = {}
            vertical_by_source = {}
            relation_by_source = {}
            confidence_rows = []
            names = []
            for name, h in source_feats.items():
                z[name] = self.ufa[name](h)
                x = self.vhsh.point_from_tangent(z[name])
                vertical_logits_for_source = self.vhsh(x)
                structured_logits_for_source = self._single_source_horizontal(vertical_logits_for_source, relation)
                vertical_by_source[name] = vertical_logits_for_source
                relation_by_source[name] = structured_logits_for_source
                confidence_rows.append(1.0 - _normalized_entropy(structured_logits_for_source, self.cfg.tau_e))
                names.append(name)

            confidence = torch.cat(confidence_rows, dim=1)
            weight_matrix = F.softmax(confidence / max(float(self.cfg.tau_s), 1e-8), dim=1)
            source_weights = {name: weight_matrix[:, idx : idx + 1] for idx, name in enumerate(names)}
            source_confidence = {name: confidence[:, idx : idx + 1] for idx, name in enumerate(names)}
            structured_logits = sum(source_weights[name] * relation_by_source[name] for name in names)
            vertical_logits = sum(source_weights[name] * vertical_by_source[name] for name in names)

        residual_gate = self._confidence_gate(base_logits, structured_logits)
        residual_logits = self.residual_scale * residual_gate * structured_logits
        final_logits = base_logits + residual_logits

        aux = {
            "base_logits": packet.base_logits,
            "vertical_logits": vertical_logits,
            "structured_logits": structured_logits,
            "relation_logits": structured_logits,
            "tree_logits": vertical_logits,
            "plugin_logits": structured_logits,
            "residual_logits": residual_logits,
            "residual_gate": residual_gate,
            "residual_scale": self.residual_scale.detach(),
            "source_weights": source_weights,
            "source_confidence": source_confidence,
            "z": z,
            "vertical_by_source": vertical_by_source,
            "relation_by_source": relation_by_source,
            "s_tree_by_source": vertical_by_source,
            "s_pure_by_source": relation_by_source,
            "A_pos": relation_info["A_pos"],
            "A_neg": relation_info["A_neg"],
            "relation_field": relation,
        }
        return final_logits, aux
