from typing import Dict, Iterable

import torch


def available_sources(features: Dict, source_names: Iterable[str] = ("text", "vision", "multimodal")):
    return [name for name in source_names if features.get(name) is not None]


@torch.no_grad()
def compute_class_centers(projected_sources: Dict[str, torch.Tensor], labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    names = [name for name, value in projected_sources.items() if value is not None]
    if not names:
        raise ValueError("no projected source features were provided")
    z_bar = None
    for name in names:
        z_bar = projected_sources[name] if z_bar is None else z_bar + projected_sources[name]
    z_bar = z_bar / float(len(names))

    centers = []
    for cls in range(int(num_classes)):
        mask = labels == cls
        if not mask.any():
            raise ValueError("class %d has no training samples; cannot initialize VMH-P structure" % cls)
        centers.append(z_bar[mask].mean(dim=0))
    return torch.stack(centers, dim=0)
