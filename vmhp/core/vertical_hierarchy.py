from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import EPS, LorentzGeometry


def _as_long_tensor(values: Iterable[int], device=None) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.long, device=device)


class VerticalHyperbolicSentimentHierarchy(nn.Module):
    def __init__(
        self,
        num_classes: int,
        tree_dim: int = 64,
        num_internal: int = 2,
        curvature: float = 0.25,
        max_tangent_norm: float = 3.0,
        beta: float = 0.3,
        tau_r: float = 0.07,
        gamma: float = 0.1,
        geometry_mode: str = "hyperbolic",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.tree_dim = int(tree_dim)
        self.num_internal = int(num_internal)
        self.beta = float(beta)
        self.tau_r = float(tau_r)
        self.gamma = float(gamma)
        self.geometry_mode = str(geometry_mode)
        if self.geometry_mode not in {"hyperbolic", "euclidean"}:
            raise ValueError("geometry_mode must be 'hyperbolic' or 'euclidean'")
        self.geometry = LorentzGeometry(curvature=curvature, max_norm=max_tangent_norm)

        self.node_tangent_params = nn.Parameter(torch.zeros(1 + self.num_internal + self.num_classes, self.tree_dim))
        self.class_tangent_params = nn.Parameter(torch.zeros(self.num_classes, self.tree_dim))

        self.register_buffer("class_to_internal", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("edge_parent", torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_child", torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_cost", torch.empty(0))
        self.register_buffer("root_edge_for_internal", torch.zeros(self.num_internal, dtype=torch.long))
        self.register_buffer("leaf_edge_for_class", torch.zeros(self.num_classes, dtype=torch.long))
        self.set_topology(self._default_topology())

    @property
    def num_nodes(self) -> int:
        return 1 + self.num_internal + self.num_classes

    def _default_topology(self) -> Dict:
        groups = [[] for _ in range(self.num_internal)]
        for cls in range(self.num_classes):
            groups[cls % self.num_internal].append(cls)
        return {"num_internal": self.num_internal, "groups": groups}

    def set_topology(self, topology: Dict):
        if topology is None:
            topology = self._default_topology()
        groups = topology.get("groups")
        class_to_internal = topology.get("class_to_internal")
        if groups is None and class_to_internal is None:
            raise ValueError("topology must include groups or class_to_internal")
        if groups is not None:
            self.num_internal = int(topology.get("num_internal", len(groups)))
            class_to_internal = [0 for _ in range(self.num_classes)]
            for internal_idx, classes in enumerate(groups):
                for cls in classes:
                    class_to_internal[int(cls)] = int(internal_idx)
        else:
            class_to_internal = [int(x) for x in class_to_internal]
            self.num_internal = int(topology.get("num_internal", max(class_to_internal) + 1))

        if len(class_to_internal) != self.num_classes:
            raise ValueError("class_to_internal length must equal num_classes")

        node_count = 1 + self.num_internal + self.num_classes
        if self.node_tangent_params.shape[0] != node_count:
            self.node_tangent_params = nn.Parameter(
                torch.zeros(node_count, self.tree_dim, device=self.node_tangent_params.device)
            )
        device = self.node_tangent_params.device
        self.class_to_internal = _as_long_tensor(class_to_internal, device=device)

        parents: List[int] = []
        children: List[int] = []
        root_edge_for_internal = []
        for internal_idx in range(self.num_internal):
            parents.append(0)
            children.append(1 + internal_idx)
            root_edge_for_internal.append(len(parents) - 1)
        leaf_edge_for_class = [0 for _ in range(self.num_classes)]
        for cls in range(self.num_classes):
            internal_idx = int(class_to_internal[cls])
            parents.append(1 + internal_idx)
            children.append(1 + self.num_internal + cls)
            leaf_edge_for_class[cls] = len(parents) - 1

        self.edge_parent = _as_long_tensor(parents, device=device)
        self.edge_child = _as_long_tensor(children, device=device)
        self.root_edge_for_internal = _as_long_tensor(root_edge_for_internal, device=device)
        self.leaf_edge_for_class = _as_long_tensor(leaf_edge_for_class, device=device)
        self.edge_cost = torch.ones(len(parents), device=device)

    def export_topology(self) -> Dict:
        groups = [[] for _ in range(self.num_internal)]
        for cls, internal_idx in enumerate(self.class_to_internal.detach().cpu().tolist()):
            groups[int(internal_idx)].append(cls)
        return {
            "num_classes": self.num_classes,
            "num_internal": self.num_internal,
            "groups": groups,
            "class_to_internal": self.class_to_internal.detach().cpu().tolist(),
        }

    def init_from_centers(self, class_centers: torch.Tensor, topology: Optional[Dict] = None):
        if topology is not None:
            self.set_topology(topology)
        if class_centers.shape != (self.num_classes, self.tree_dim):
            raise ValueError(
                "class_centers must have shape (%d, %d), got %s"
                % (self.num_classes, self.tree_dim, tuple(class_centers.shape))
            )
        class_centers = class_centers.to(device=self.node_tangent_params.device, dtype=self.node_tangent_params.dtype)
        node_init = torch.zeros(self.num_nodes, self.tree_dim, device=class_centers.device, dtype=class_centers.dtype)
        node_init[0] = class_centers.mean(dim=0)
        for internal_idx in range(self.num_internal):
            cls_mask = self.class_to_internal == internal_idx
            if cls_mask.any():
                node_init[1 + internal_idx] = class_centers[cls_mask].mean(dim=0)
            else:
                node_init[1 + internal_idx] = class_centers.mean(dim=0)
        node_init[1 + self.num_internal :] = class_centers
        with torch.no_grad():
            self.node_tangent_params.copy_(node_init)
            self.class_tangent_params.copy_(class_centers)
            self.edge_cost.copy_(self._compute_initial_edge_cost(node_init))

    def load_initial_state(self, node_init: torch.Tensor, prototype_init: torch.Tensor, edge_cost: Optional[torch.Tensor] = None):
        node_init = node_init.to(device=self.node_tangent_params.device, dtype=self.node_tangent_params.dtype)
        prototype_init = prototype_init.to(device=self.class_tangent_params.device, dtype=self.class_tangent_params.dtype)
        if node_init.shape != self.node_tangent_params.shape:
            raise ValueError("node_init shape mismatch")
        if prototype_init.shape != self.class_tangent_params.shape:
            raise ValueError("prototype_init shape mismatch")
        with torch.no_grad():
            self.node_tangent_params.copy_(node_init)
            self.class_tangent_params.copy_(prototype_init)
            if edge_cost is not None and self.geometry_mode == "hyperbolic":
                self.edge_cost.copy_(edge_cost.to(device=self.edge_cost.device, dtype=self.edge_cost.dtype))
            else:
                self.edge_cost.copy_(self._compute_initial_edge_cost(node_init))

    def _compute_initial_edge_cost(self, node_tangent: torch.Tensor) -> torch.Tensor:
        nodes = self.point_from_tangent(node_tangent)
        parent = nodes[self.edge_parent]
        child = nodes[self.edge_child]
        cost = self.distance(parent, child)
        return cost / cost.mean().clamp_min(EPS)

    def point_from_tangent(self, z: torch.Tensor) -> torch.Tensor:
        if self.geometry_mode == "hyperbolic":
            return self.geometry.exp_origin(z)
        return self.geometry.project_tangent(z)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.geometry_mode == "hyperbolic":
            return self.geometry.distance(x, y)
        return (x - y).norm(dim=-1)

    def pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.geometry_mode == "hyperbolic":
            return self.geometry.pairwise_distance(x, y)
        return torch.cdist(x.float(), y.float())

    def pairwise_tangent_distance(self, z: torch.Tensor) -> torch.Tensor:
        points = self.point_from_tangent(z)
        return self.pairwise_distance(points, points)

    def node_embeddings(self) -> torch.Tensor:
        return self.point_from_tangent(self.node_tangent_params)

    def prototype_embeddings(self) -> torch.Tensor:
        return self.point_from_tangent(self.class_tangent_params)

    def compute_routing(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        nodes = self.node_embeddings()
        edge_log_prob = x.new_empty(x.shape[0], self.edge_parent.numel())
        for parent_id in torch.unique(self.edge_parent).tolist():
            edge_ids = torch.nonzero(self.edge_parent == int(parent_id), as_tuple=False).view(-1)
            child_ids = self.edge_child[edge_ids]
            dist = self.pairwise_distance(x, nodes[child_ids])
            cost = self.edge_cost[edge_ids].unsqueeze(0)
            score = -(dist + self.gamma * cost) / max(self.tau_r, EPS)
            edge_log_prob[:, edge_ids] = F.log_softmax(score, dim=-1)
        return {"edge_log_prob": edge_log_prob, "nodes": nodes}

    def compute_path_scores(self, x: torch.Tensor, routing: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        routing = routing or self.compute_routing(x)
        edge_log_prob = routing["edge_log_prob"]
        internal_edge = self.root_edge_for_internal[self.class_to_internal]
        leaf_edge = self.leaf_edge_for_class
        return edge_log_prob[:, internal_edge] + edge_log_prob[:, leaf_edge]

    def compute_prototype_scores(self, x: torch.Tensor) -> torch.Tensor:
        prototypes = self.prototype_embeddings()
        return -self.beta * self.pairwise_distance(x, prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing = self.compute_routing(x)
        return self.compute_path_scores(x, routing) + self.compute_prototype_scores(x)

    def score_tangent(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(self.point_from_tangent(z))

    def label_tree_distances(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(device=self.class_to_internal.device, dtype=torch.long)
        internals = self.class_to_internal[labels]
        same_cls = labels[:, None] == labels[None, :]
        same_internal = internals[:, None] == internals[None, :]
        dist = torch.full((labels.numel(), labels.numel()), 4.0, device=labels.device)
        dist = torch.where(same_internal, torch.full_like(dist, 2.0), dist)
        dist = torch.where(same_cls, torch.zeros_like(dist), dist)
        return dist
