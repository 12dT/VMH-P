import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from tools.common import SOURCE_NAMES, iter_feature_batches, load_feature_file, num_classes_from_features, weighted_f1_acc


def concat_sources(batch: Dict, sources: List[str]) -> torch.Tensor:
    values = []
    for name in sources:
        value = batch.get(name)
        if value is not None:
            values.append(value.float())
    if not values:
        raise ValueError("no requested source is present in the feature batch")
    return torch.cat(values, dim=-1)


def infer_input_dim(features: Dict, sources: List[str]) -> int:
    dim = 0
    for name in sources:
        value = features.get(name)
        if value is not None:
            dim += int(value.shape[-1])
    if dim <= 0:
        raise ValueError("feature file does not contain any requested source")
    return dim


def metric_from_logits(features: Dict, logits: torch.Tensor):
    labels = features["labels"].long().cpu().tolist()
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    return weighted_f1_acc(labels, pred)


def score(metrics, rank_by: str):
    acc, f1 = metrics
    if rank_by == "acc":
        return acc
    if rank_by == "f1":
        return f1
    return acc + f1


class ResidualHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, kind: str, hidden_dim: int, dropout: float, gamma_init: float):
        super().__init__()
        if kind == "linear":
            self.head = nn.Linear(input_dim, num_classes)
        elif kind == "mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"unsupported residual head: {kind}")
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        return base_logits + self.gamma * self.head(x)


@torch.no_grad()
def evaluate_model(model, features, sources, batch_size, device):
    model.eval()
    labels_all = []
    pred_all = []
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        x = concat_sources(batch, sources)
        logits = model(x, batch["base_logits"].float())
        labels_all.extend(batch["labels"].detach().cpu().tolist())
        pred_all.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return weighted_f1_acc(labels_all, pred_all)


def train_residual(args, train, val, test, sources, device):
    input_dim = infer_input_dim(train, sources)
    num_classes = num_classes_from_features(train, val, test)
    model = ResidualHead(input_dim, num_classes, args.head, args.hidden_dim, args.dropout, args.gamma_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None
    bad_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in iter_feature_batches(train, args.batch_size, shuffle=True, device=device):
            labels = batch["labels"].long()
            x = concat_sources(batch, sources)
            logits = model(x, batch["base_logits"].float())
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * labels.numel()
            count += labels.numel()
        val_metric = evaluate_model(model, val, sources, args.batch_size, device)
        test_metric = evaluate_model(model, test, sources, args.batch_size, device)
        print(
            "epoch %03d loss %.6f val %.6f/%.6f test %.6f/%.6f gamma %.6f"
            % (
                epoch,
                total / max(count, 1),
                val_metric[0],
                val_metric[1],
                test_metric[0],
                test_metric[1],
                float(model.gamma.detach().cpu()),
            )
        )
        select_metric = test_metric if args.select_by_test else val_metric
        select_score = score(select_metric, args.rank_by)
        if best is None or select_score > best["selection_score"]:
            bad_epochs = 0
            best = {
                "epoch": epoch,
                "selection_score": select_score,
                "selection_split": "test" if args.select_by_test else "val",
                "val": {"accuracy": val_metric[0], "f1_w": val_metric[1]},
                "test": {"accuracy": test_metric[0], "f1_w": test_metric[1]},
                "state": copy.deepcopy(model.state_dict()),
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best is not None:
        model.load_state_dict(best["state"])
    return {
        "best": {k: v for k, v in best.items() if k != "state"},
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "seconds": time.time() - start_time,
    }


@torch.no_grad()
def collect_concat(features, sources, batch_size, device):
    chunks = []
    labels = []
    bases = []
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        chunks.append(concat_sources(batch, sources).detach().cpu())
        labels.append(batch["labels"].detach().cpu())
        bases.append(batch["base_logits"].detach().cpu())
    return torch.cat(chunks, dim=0), torch.cat(labels, dim=0), torch.cat(bases, dim=0)


def prototype_logits(x: torch.Tensor, centers: torch.Tensor, temperature: float):
    x = F.normalize(x.float(), dim=-1, eps=1e-12)
    centers = F.normalize(centers.float(), dim=-1, eps=1e-12)
    return x @ centers.t() / max(float(temperature), 1e-6)


def run_euclidean_prototype(args, train, val, test, sources, device):
    start_time = time.time()
    x_train, y_train, _ = collect_concat(train, sources, args.batch_size, device)
    num_classes = num_classes_from_features(train, val, test)
    centers = []
    for cls in range(num_classes):
        mask = y_train == cls
        if bool(mask.any()):
            centers.append(x_train[mask].mean(dim=0))
        else:
            centers.append(torch.zeros(x_train.shape[-1]))
    centers = torch.stack(centers, dim=0)
    val_x, val_y, val_base = collect_concat(val, sources, args.batch_size, device)
    test_x, test_y, test_base = collect_concat(test, sources, args.batch_size, device)
    val_proto = prototype_logits(val_x, centers, args.temperature)
    test_proto = prototype_logits(test_x, centers, args.temperature)

    scales = [float(x) for x in args.prototype_scales.split(",") if x.strip()]
    records = []
    for gamma in scales:
        val_logits = val_base.float() + gamma * val_proto
        test_logits = test_base.float() + gamma * test_proto
        val_metric = weighted_f1_acc(val_y.tolist(), val_logits.argmax(dim=-1).tolist())
        test_metric = weighted_f1_acc(test_y.tolist(), test_logits.argmax(dim=-1).tolist())
        records.append(
            {
                "gamma": gamma,
                "val": {"accuracy": val_metric[0], "f1_w": val_metric[1]},
                "test": {"accuracy": test_metric[0], "f1_w": test_metric[1]},
                "selection_score": score(test_metric if args.select_by_test else val_metric, args.rank_by),
            }
        )
    best = max(records, key=lambda item: item["selection_score"])
    head_only_val = weighted_f1_acc(val_y.tolist(), val_proto.argmax(dim=-1).tolist())
    head_only_test = weighted_f1_acc(test_y.tolist(), test_proto.argmax(dim=-1).tolist())
    return {
        "best": best,
        "head_only": {
            "val": {"accuracy": head_only_val[0], "f1_w": head_only_val[1]},
            "test": {"accuracy": head_only_test[0], "f1_w": head_only_test[1]},
        },
        "params": 0,
        "seconds": time.time() - start_time,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Train non-structured cached-feature heads for SET-P controls.")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", choices=["linear", "mlp", "euclidean_prototype"], required=True)
    parser.add_argument("--sources", default="text,vision,multimodal")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma-init", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--prototype-scales",
        default="0,0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0",
    )
    parser.add_argument("--rank-by", choices=["sum", "acc", "f1"], default="f1")
    parser.add_argument("--select-by-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    train = load_feature_file(args.train_features)
    val = load_feature_file(args.val_features)
    test = load_feature_file(args.test_features)
    requested = [x.strip() for x in args.sources.split(",") if x.strip()]
    sources = [name for name in requested if train.get(name) is not None]
    if not sources:
        sources = [name for name in SOURCE_NAMES if train.get(name) is not None]

    base_val = metric_from_logits(val, val["base_logits"])
    base_test = metric_from_logits(test, test["base_logits"])
    if args.head == "euclidean_prototype":
        result = run_euclidean_prototype(args, train, val, test, sources, device)
    else:
        result = train_residual(args, train, val, test, sources, device)
    result.update(
        {
            "format": "setp_cached_head_control_v1",
            "head": args.head,
            "sources": sources,
            "train_features": args.train_features,
            "val_features": args.val_features,
            "test_features": args.test_features,
            "rank_by": args.rank_by,
            "selection_split": "test" if args.select_by_test else "val",
            "base": {
                "val": {"accuracy": base_val[0], "f1_w": base_val[1]},
                "test": {"accuracy": base_test[0], "f1_w": base_test[1]},
            },
        }
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["best"], indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
