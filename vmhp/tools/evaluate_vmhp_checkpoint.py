import argparse
import json
import os
import sys

import torch

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from vmhp.tools.common import load_feature_file
from vmhp.tools.train_vmhp import evaluate
from vmhp.tools.rescue_search import load_vmhp


def fmt(metrics):
    return {
        key: {"accuracy": float(value[0]), "f1_w": float(value[1])}
        for key, value in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a val-selected VMH-P checkpoint on val/test splits.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_vmhp(args.checkpoint, None, device)
    val = load_feature_file(args.val_features)
    test = load_feature_file(args.test_features)
    val_metrics = evaluate(model, val, args.batch_size, device)
    test_metrics = evaluate(model, test, args.batch_size, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    result = {
        "format": "vmhp_checkpoint_eval_v1",
        "checkpoint": args.checkpoint,
        "structure_path": checkpoint.get("structure_path"),
        "selection_split": checkpoint.get("selection_split"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_score": checkpoint.get("selection_score"),
        "val_features": args.val_features,
        "test_features": args.test_features,
        "val": fmt(val_metrics),
        "test": fmt(test_metrics),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
