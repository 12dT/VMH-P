import argparse
import json
import os
import sys

import torch

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from vmhp.tools.common import load_feature_file
from vmhp.tools.evaluate_vmhp_checkpoint import load_vmhp
from vmhp.tools.train_vmhp import evaluate


def fmt(metrics):
    return {
        key: {"accuracy": float(value[0]), "f1_w": float(value[1])}
        for key, value in metrics.items()
    }


def _copy_relation(model, structure):
    pos = structure["relation_pos_init"].to(
        device=model.ring.A_pos_init.device,
        dtype=model.ring.A_pos_init.dtype,
    )
    neg = structure["relation_neg_init"].to(
        device=model.ring.A_neg_init.device,
        dtype=model.ring.A_neg_init.dtype,
    )
    with torch.no_grad():
        model.ring.A_pos_init.copy_(pos)
        model.ring.A_neg_init.copy_(neg)


def apply_counterfactual(model, mode, structure_path):
    if mode == "full":
        return None
    structure = torch.load(structure_path, map_location="cpu") if structure_path else None

    with torch.no_grad():
        if mode == "vertical_only":
            model.ring.A_pos_init.zero_()
            model.ring.A_neg_init.zero_()
            model.ring.theta_pos.zero_()
            model.ring.theta_neg.zero_()
            return {"changed": ["relation_pos_init", "relation_neg_init", "theta_pos", "theta_neg"]}

        if mode == "random_relation":
            if structure is None:
                raise ValueError("--structure is required for random_relation")
            _copy_relation(model, structure)
            return {"changed": ["relation_pos_init", "relation_neg_init"], "structure_path": structure_path}

        if mode == "random_tree":
            if structure is None:
                raise ValueError("--structure is required for random_tree")
            topology = structure.get("topology")
            if topology is None:
                raise ValueError("random_tree structure must include topology")
            model.tree.set_topology(topology)
            edge_cost = model.tree._compute_initial_edge_cost(model.tree.node_tangent_params)
            model.tree.edge_cost.copy_(edge_cost)
            return {
                "changed": [
                    "class_to_internal",
                    "edge_parent",
                    "edge_child",
                    "root_edge_for_internal",
                    "leaf_edge_for_class",
                    "edge_cost",
                ],
                "structure_path": structure_path,
            }

    raise ValueError(f"unsupported counterfactual mode: {mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate structural counterfactuals with one frozen Full VMH-P checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["full", "vertical_only", "random_tree", "random_relation"], required=True)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_vmhp(args.checkpoint, device)
    applied = apply_counterfactual(model, args.mode, args.structure)
    model.eval()

    val = load_feature_file(args.val_features)
    test = load_feature_file(args.test_features)
    val_metrics = evaluate(model, val, args.batch_size, device)
    test_metrics = evaluate(model, test, args.batch_size, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    result = {
        "format": "vmhp_frozen_full_structure_counterfactual_v1",
        "checkpoint": args.checkpoint,
        "checkpoint_structure_path": checkpoint.get("structure_path"),
        "mode": args.mode,
        "override_structure_path": args.structure,
        "counterfactual": applied,
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
