from typing import Dict, List

import numpy as np
import torch


def _fallback_kmeans(x: torch.Tensor, k: int, steps: int = 100) -> torch.Tensor:
    centers = x[:k].clone()
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(steps):
        dist = torch.cdist(x, centers)
        new_labels = dist.argmin(dim=-1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centers[idx] = x[mask].mean(dim=0)
    return labels


def build_tree_topology(class_centers: torch.Tensor, num_internal: int = 2, random_state: int = 42, n_init: int = 50) -> Dict:
    num_classes = int(class_centers.shape[0])
    num_internal = int(num_internal)
    if num_internal < 1 or num_internal > num_classes:
        raise ValueError("num_internal must be in [1, num_classes]")
    if num_internal == 1:
        labels = torch.zeros(num_classes, dtype=torch.long)
    elif num_internal == num_classes:
        labels = torch.arange(num_classes, dtype=torch.long)
    else:
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=num_internal, random_state=random_state, n_init=n_init)
            labels = torch.tensor(km.fit_predict(class_centers.detach().cpu().numpy()), dtype=torch.long)
        except Exception:
            torch.manual_seed(random_state)
            labels = _fallback_kmeans(class_centers.detach().cpu(), num_internal)

    groups: List[List[int]] = [[] for _ in range(num_internal)]
    for cls, group in enumerate(labels.tolist()):
        groups[int(group)].append(cls)
    return {
        "num_classes": num_classes,
        "num_internal": num_internal,
        "groups": groups,
        "class_to_internal": labels.tolist(),
    }


def _groups_from_labels(labels: torch.Tensor, num_classes: int, num_internal: int) -> Dict:
    groups: List[List[int]] = [[] for _ in range(num_internal)]
    for cls, group in enumerate(labels.tolist()):
        groups[int(group)].append(cls)
    return {
        "num_classes": int(num_classes),
        "num_internal": int(num_internal),
        "groups": groups,
        "class_to_internal": labels.tolist(),
    }


def build_relation_aware_tree_topology(
    class_centers: torch.Tensor,
    relation: torch.Tensor,
    num_internal: int = 3,
    center_weight: float = 0.35,
    relation_weight: float = 0.65,
    negative_weight: float = 0.60,
    random_state: int = 42,
) -> Dict:
    """Build a vertical topology from class centers and horizontal relations.

    The affinity remains host-derived: class centers provide geometric proximity,
    while the relation field encourages supportive pairs and discourages
    contrasting pairs. This is intended for structure analysis or an explicitly
    relation-aware topology variant, not as a silent replacement for the default
    K-means induction.
    """
    centers = class_centers.detach().cpu().float()
    rel = relation.detach().cpu().float()
    num_classes = int(centers.shape[0])
    num_internal = int(num_internal)
    if rel.shape != (num_classes, num_classes):
        raise ValueError("relation must have shape (%d, %d), got %s" % (num_classes, num_classes, tuple(rel.shape)))
    if num_internal < 1 or num_internal > num_classes:
        raise ValueError("num_internal must be in [1, num_classes]")
    if num_internal == 1:
        return _groups_from_labels(torch.zeros(num_classes, dtype=torch.long), num_classes, num_internal)
    if num_internal == num_classes:
        return _groups_from_labels(torch.arange(num_classes, dtype=torch.long), num_classes, num_internal)

    centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    center_aff = ((centers @ centers.t()).numpy() + 1.0) * 0.5
    rel_np = rel.numpy()
    pos_sym = (np.maximum(rel_np, 0.0) + np.maximum(rel_np.T, 0.0)) * 0.5
    neg_sym = (np.maximum(-rel_np, 0.0) + np.maximum(-rel_np.T, 0.0)) * 0.5
    relation_aff = np.maximum(pos_sym - float(negative_weight) * neg_sym, 0.0)

    affinity = float(center_weight) * center_aff + float(relation_weight) * relation_aff
    affinity = (affinity + affinity.T) * 0.5
    affinity = np.maximum(affinity, 0.0)
    np.fill_diagonal(affinity, 1.0)
    affinity = affinity + 1e-4 * (np.ones_like(affinity) - np.eye(num_classes))

    try:
        from sklearn.cluster import SpectralClustering

        sc = SpectralClustering(
            n_clusters=num_internal,
            affinity="precomputed",
            random_state=random_state,
            assign_labels="kmeans",
        )
        labels = torch.tensor(sc.fit_predict(affinity), dtype=torch.long)
    except Exception:
        distance = 1.0 - affinity
        try:
            from sklearn.cluster import AgglomerativeClustering

            try:
                model = AgglomerativeClustering(n_clusters=num_internal, metric="precomputed", linkage="average")
            except TypeError:
                model = AgglomerativeClustering(n_clusters=num_internal, affinity="precomputed", linkage="average")
            labels = torch.tensor(model.fit_predict(distance), dtype=torch.long)
        except Exception:
            labels = _fallback_kmeans(torch.tensor(affinity, dtype=torch.float32), num_internal)

    topology = _groups_from_labels(labels, num_classes, num_internal)
    topology["induction"] = {
        "mode": "relation_aware",
        "center_weight": float(center_weight),
        "relation_weight": float(relation_weight),
        "negative_weight": float(negative_weight),
        "random_state": int(random_state),
    }
    return topology
