import os
import sys
from typing import Dict, Iterable, List

import torch


PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)


SOURCE_NAMES = ("text", "vision", "multimodal", "routed_text", "routed_vision")
PRIMARY_SOURCE_NAMES = ("text", "vision", "multimodal")


def load_feature_file(path: str) -> Dict:
    data = torch.load(path, map_location="cpu")
    if "vision" not in data and "image" in data:
        data["vision"] = data["image"]
    if "multimodal" not in data and "mm" in data:
        data["multimodal"] = data["mm"]
    if "labels" not in data and "label" in data:
        data["labels"] = data["label"]
    if "base_logits" not in data:
        raise KeyError("feature file must include base_logits")
    if "labels" not in data:
        raise KeyError("feature file must include labels")
    return data


def infer_feature_dims(features: Dict, source_names: Iterable[str] = SOURCE_NAMES) -> Dict[str, int]:
    dims = {}
    for name in source_names:
        value = features.get(name)
        if value is not None:
            dims[name] = int(value.shape[-1])
    if not dims:
        raise ValueError("feature file does not contain any source feature")
    return dims


def num_classes_from_features(*feature_sets: Dict) -> int:
    max_from_logits = max(int(f["base_logits"].shape[-1]) for f in feature_sets if f is not None)
    max_from_labels = max(int(f["labels"].max().item()) + 1 for f in feature_sets if f is not None)
    return max(max_from_logits, max_from_labels)


def iter_feature_batches(features: Dict, batch_size: int, shuffle: bool = False, device=None) -> Iterable[Dict]:
    n = int(features["labels"].shape[0])
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        batch = {}
        for key, value in features.items():
            if torch.is_tensor(value) and value.shape[0] == n:
                batch[key] = value[idx].to(device) if device is not None else value[idx]
            else:
                batch[key] = value
        yield batch


def packet_from_batch(batch: Dict):
    from core.packet import EvidencePacket

    extra_sources = {
        name: batch.get(name)
        for name in SOURCE_NAMES
        if name not in PRIMARY_SOURCE_NAMES and batch.get(name) is not None
    }
    return EvidencePacket(
        text=batch.get("text"),
        vision=batch.get("vision"),
        multimodal=batch.get("multimodal"),
        base_logits=batch["base_logits"],
        label=batch["labels"],
        meta=batch.get("meta"),
        extra_sources=extra_sources,
    )


def weighted_f1_acc(y_true: List[int], y_pred: List[int]):
    try:
        from sklearn.metrics import accuracy_score, f1_score

        return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="weighted"))
    except Exception:
        y_true_t = torch.tensor(y_true)
        y_pred_t = torch.tensor(y_pred)
        acc = (y_true_t == y_pred_t).float().mean().item()
        return acc, acc
