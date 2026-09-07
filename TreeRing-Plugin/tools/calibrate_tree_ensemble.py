import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from tools.common import load_feature_file
from tools.rescue_search import collect, load_plugin, metric


BRANCH_TO_KEY = {
    "base": "base_logits",
    "calibrated_base": "calibrated_base_logits",
    "tree": "tree_logits",
    "plugin": "plugin_logits",
    "final": "final_logits",
}


def score(metrics: Dict[str, float], mode: str) -> float:
    if mode == "acc":
        return metrics["accuracy"]
    if mode == "f1":
        return metrics["f1_w"]
    return metrics["accuracy"] + metrics["f1_w"]


def parse_candidate(value: str) -> Tuple[str, str, str]:
    # name=/path/to/best_plugin.pt:tree
    name, spec = value.split("=", 1) if "=" in value else (None, value)
    path, branch = spec.rsplit(":", 1)
    if branch not in BRANCH_TO_KEY:
        raise ValueError("unknown branch %r; choose from %s" % (branch, sorted(BRANCH_TO_KEY)))
    if name is None:
        name = "%s:%s" % (os.path.basename(os.path.dirname(path)), branch)
    return name, path, branch


def evaluate_logits(labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    return metric(labels, logits.argmax(dim=-1))


def main():
    parser = argparse.ArgumentParser(description="Calibrate a pairwise TreeRing branch ensemble on cached features.")
    parser.add_argument("--structure", required=True)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=checkpoint.pt:branch")
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--rank-by", choices=["sum", "acc", "f1"], default="sum")
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    dev_features = load_feature_file(args.dev_features)
    test_features = load_feature_file(args.test_features)

    dev_rows = {}
    test_rows = {}
    labels = {}
    singles = {}
    for raw in args.candidate:
        name, checkpoint, branch = parse_candidate(raw)
        plugin = load_plugin(checkpoint, args.structure, device)
        dev = collect(plugin, dev_features, args.batch_size, device)
        test = collect(plugin, test_features, args.batch_size, device)
        key = BRANCH_TO_KEY[branch]
        dev_rows[name] = dev[key]
        test_rows[name] = test[key]
        labels["dev"] = dev["labels"]
        labels["test"] = test["labels"]
        singles[name] = {
            "checkpoint": checkpoint,
            "branch": branch,
            "dev": evaluate_logits(dev["labels"], dev[key]),
            "test": evaluate_logits(test["labels"], test[key]),
        }

    names = list(dev_rows)
    records: List[Dict] = []
    steps = int(round(1.0 / args.alpha_step))
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for idx in range(steps + 1):
                alpha = idx * args.alpha_step
                dev_logits = (1.0 - alpha) * dev_rows[left] + alpha * dev_rows[right]
                test_logits = (1.0 - alpha) * test_rows[left] + alpha * test_rows[right]
                record = {
                    "left": left,
                    "right": right,
                    "alpha_right": alpha,
                    "dev": evaluate_logits(labels["dev"], dev_logits),
                    "test": evaluate_logits(labels["test"], test_logits),
                }
                select_metrics = record[args.selection_split]
                record["selection_score"] = score(select_metrics, args.rank_by)
                records.append(record)

    best = max(records, key=lambda item: item["selection_score"])
    result = {
        "format": "treering_branch_ensemble_v1",
        "structure": args.structure,
        "dev_features": args.dev_features,
        "test_features": args.test_features,
        "selection_split": args.selection_split,
        "rank_by": args.rank_by,
        "singles": singles,
        "best": best,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(best, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
