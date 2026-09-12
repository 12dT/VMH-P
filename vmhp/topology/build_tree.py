from typing import Dict, List

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

