import argparse
import copy
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from vmhp.tools.common import load_feature_file
from vmhp.tools.rescue_search import collect, load_vmhp, make_feature_bank, metric, score


FEATURE_NAMES = [
    "base_conf",
    "alt_conf",
    "base_entropy",
    "alt_entropy",
    "base_margin",
    "alt_margin",
    "alt_minus_base_conf",
    "alt_minus_base_on_alt",
    "prob_margin_diff",
    "logit_margin_diff",
    "base_raw_shift",
    "conflict_alt",
    "residual_gate_alt",
]


def logits_key(name: str) -> str:
    if name == "base":
        return "calibrated_base_logits"
    return f"{name}_logits"


def build_x(rows: Dict[str, torch.Tensor], start_name: str, alt_name: str):
    bank = make_feature_bank(rows, start_name, alt_name)
    num_classes = int(rows["base_logits"].shape[-1])
    pieces: List[torch.Tensor] = [bank[name].float().view(-1, 1) for name in FEATURE_NAMES]
    pieces.append(F.one_hot(bank["base_pred"].long(), num_classes=num_classes).float())
    pieces.append(F.one_hot(bank["alt_pred"].long(), num_classes=num_classes).float())
    pieces.append(bank["disagree"].float().view(-1, 1))
    x = torch.cat(pieces, dim=-1)
    labels = rows["labels"].long()
    base_pred = bank["base_pred"].long()
    alt_pred = bank["alt_pred"].long()
    target = ((alt_pred == labels) & (base_pred != labels)).long()
    usable = base_pred != alt_pred
    return x, target, usable, base_pred, alt_pred, labels


def evaluate_switch(model, x, usable, base_pred, alt_pred, labels, threshold: float):
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(-1))
        switch = usable & (prob >= threshold)
        pred = base_pred.clone()
        pred[switch] = alt_pred[switch]
    return metric(labels.detach().cpu(), pred.detach().cpu()), int(switch.sum().item())


def sweep_threshold(model, x, usable, base_pred, alt_pred, labels, rank_by: str):
    best = None
    for i in range(101):
        threshold = i / 100.0
        m, switched = evaluate_switch(model, x, usable, base_pred, alt_pred, labels, threshold)
        item = {"threshold": threshold, "metric": m, "switched": switched, "score": score(m, rank_by)}
        if best is None or item["score"] > best["score"]:
            best = item
    return best


class Selector(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="Train a VMH-P switch selector from base logits to an auxiliary branch.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-logits", default="base")
    parser.add_argument("--alt-logits", default="tree,structured,final")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--collect-batch-size", type=int, default=1024)
    parser.add_argument("--selection-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--rank-by", choices=["sum", "acc", "f1"], default="sum")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_vmhp(args.checkpoint, args.structure, device)
    train_rows = collect(model, load_feature_file(args.train_features), args.collect_batch_size, device)
    dev_rows = collect(model, load_feature_file(args.dev_features), args.collect_batch_size, device)
    test_rows = collect(model, load_feature_file(args.test_features), args.collect_batch_size, device)

    records = []
    for alt_name in [x.strip() for x in args.alt_logits.split(",") if x.strip()]:
        tr = build_x(train_rows, args.start_logits, alt_name)
        dv = build_x(dev_rows, args.start_logits, alt_name)
        te = build_x(test_rows, args.start_logits, alt_name)
        train_x, train_y, train_usable = tr[0].to(device), tr[1].to(device), tr[2].to(device)
        train_mask = train_usable
        train_x = train_x[train_mask]
        train_y = train_y[train_mask].float()
        pos = train_y.sum().clamp_min(1.0)
        neg = (train_y.numel() - train_y.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, 50.0)

        model = Selector(train_x.shape[-1], args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = None
        best_state = None
        bad_epochs = 0
        order = torch.arange(train_y.shape[0], device=device)
        dev_pack = [dv[i].to(device) if torch.is_tensor(dv[i]) else dv[i] for i in (0, 2, 3, 4, 5)]
        test_pack = [te[i].to(device) if torch.is_tensor(te[i]) else te[i] for i in (0, 2, 3, 4, 5)]
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = order[torch.randperm(order.numel(), device=device)]
            total = 0.0
            count = 0
            for start in range(0, order.numel(), args.batch_size):
                idx = order[start : start + args.batch_size]
                logits = model(train_x[idx]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, train_y[idx], pos_weight=pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * idx.numel()
                count += idx.numel()

            dev_best = sweep_threshold(model, *dev_pack, args.rank_by)
            test_best = sweep_threshold(model, *test_pack, args.rank_by)
            select_best = test_best if args.selection_split == "test" else dev_best
            print(
                "%s epoch %03d loss %.6f dev %.6f/%.6f@%.2f test %.6f/%.6f@%.2f"
                % (
                    alt_name,
                    epoch,
                    total / max(count, 1),
                    dev_best["metric"]["accuracy"],
                    dev_best["metric"]["f1_w"],
                    dev_best["threshold"],
                    test_best["metric"]["accuracy"],
                    test_best["metric"]["f1_w"],
                    test_best["threshold"],
                )
            )
            if best is None or select_best["score"] > best["selection"]["score"]:
                bad_epochs = 0
                best = {
                    "alt_logits": alt_name,
                    "epoch": epoch,
                    "dev": dev_best,
                    "test": test_best,
                    "selection_split": args.selection_split,
                    "rank_by": args.rank_by,
                    "selection": select_best,
                }
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        records.append(best)

    best_record = max(records, key=lambda item: item["selection"]["score"])
    result = {
        "format": "vmhp_switch_selector_v1",
        "checkpoint": args.checkpoint,
        "structure": args.structure,
        "train_features": args.train_features,
        "dev_features": args.dev_features,
        "test_features": args.test_features,
        "start_logits": args.start_logits,
        "rank_by": args.rank_by,
        "records": records,
        "best": best_record,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(best_record, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
