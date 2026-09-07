import argparse
import json
import os
import sys
from typing import List

import torch

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from core.hyperbolic_tree import HyperbolicEvidenceTree
from core.relation_ring import safe_row_l1_norm


def groups_from_assignment(assignment: List[int], num_internal: int):
    groups = [[] for _ in range(num_internal)]
    for cls, group_id in enumerate(assignment):
        groups[int(group_id)].append(int(cls))
    return groups


def balanced_random_assignment(num_classes: int, num_internal: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_classes, generator=generator).tolist()
    assignment = [0 for _ in range(num_classes)]
    for pos, cls in enumerate(perm):
        assignment[int(cls)] = int(pos % num_internal)
    return assignment


def rebuild_tree_state(structure: dict, topology: dict):
    cfg = dict(structure.get("cfg", {}))
    num_classes = int(structure["num_classes"])
    tree_dim = int(cfg.get("tree_dim", structure["class_centers"].shape[-1]))
    tree = HyperbolicEvidenceTree(
        num_classes=num_classes,
        tree_dim=tree_dim,
        num_internal=int(topology["num_internal"]),
        curvature=float(cfg.get("curvature", 0.25)),
        max_tangent_norm=float(cfg.get("max_tangent_norm", 3.0)),
        beta=float(cfg.get("beta", 0.3)),
        tau_r=float(cfg.get("tau_r", 0.07)),
        gamma=float(cfg.get("gamma", 0.1)),
    )
    tree.init_from_centers(structure["class_centers"], topology=topology)
    structure["topology"] = tree.export_topology()
    structure["node_init"] = tree.node_tangent_params.detach().cpu()
    structure["prototype_init"] = tree.class_tangent_params.detach().cpu()
    structure["edge_cost"] = tree.edge_cost.detach().cpu()
    structure["cfg"]["num_internal"] = int(topology["num_internal"])
    return structure


def zero_relations(structure: dict):
    structure["relation_pos_init"] = torch.zeros_like(structure["relation_pos_init"])
    structure["relation_neg_init"] = torch.zeros_like(structure["relation_neg_init"])
    return structure


def random_relations(structure: dict, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    pos = torch.rand(structure["relation_pos_init"].shape, generator=generator)
    neg = torch.rand(structure["relation_neg_init"].shape, generator=generator)
    offdiag = torch.ones_like(pos) - torch.eye(pos.shape[0], dtype=pos.dtype)
    structure["relation_pos_init"] = safe_row_l1_norm(pos * offdiag)
    structure["relation_neg_init"] = safe_row_l1_norm(neg * offdiag)
    return structure


def shuffled_relations(structure: dict, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    num_classes = int(structure["num_classes"])
    perm = torch.randperm(num_classes, generator=generator)
    structure["relation_pos_init"] = structure["relation_pos_init"][perm][:, perm].clone()
    structure["relation_neg_init"] = structure["relation_neg_init"][perm][:, perm].clone()
    return structure


def uniform_relations(structure: dict):
    num_classes = int(structure["num_classes"])
    offdiag = torch.ones(num_classes, num_classes) - torch.eye(num_classes)
    rel = safe_row_l1_norm(offdiag)
    structure["relation_pos_init"] = rel.clone()
    structure["relation_neg_init"] = rel.clone()
    return structure


def main():
    parser = argparse.ArgumentParser(description="Create SET-P structure variants for controlled ablations.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "vertical_only",
            "random_relation",
            "shuffled_relation",
            "uniform_relation",
            "random_tree",
            "flat_tree",
            "horizontal_only",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    structure = torch.load(args.input, map_location="cpu")
    structure = {k: v.clone() if torch.is_tensor(v) else v for k, v in structure.items()}
    structure["cfg"] = dict(structure.get("cfg", {}))
    num_classes = int(structure["num_classes"])
    num_internal = int(structure.get("topology", {}).get("num_internal", structure["cfg"].get("num_internal", 2)))

    if args.variant == "vertical_only":
        structure = zero_relations(structure)
    elif args.variant == "random_relation":
        structure = random_relations(structure, args.seed)
    elif args.variant == "shuffled_relation":
        structure = shuffled_relations(structure, args.seed)
    elif args.variant == "uniform_relation":
        structure = uniform_relations(structure)
    elif args.variant == "random_tree":
        assignment = balanced_random_assignment(num_classes, num_internal, args.seed)
        topology = {
            "num_classes": num_classes,
            "num_internal": num_internal,
            "groups": groups_from_assignment(assignment, num_internal),
            "class_to_internal": assignment,
        }
        structure = rebuild_tree_state(structure, topology)
    elif args.variant in {"flat_tree", "horizontal_only"}:
        assignment = [0 for _ in range(num_classes)]
        topology = {
            "num_classes": num_classes,
            "num_internal": 1,
            "groups": [list(range(num_classes))],
            "class_to_internal": assignment,
        }
        structure = rebuild_tree_state(structure, topology)

    structure["ablation_variant"] = args.variant
    structure["ablation_seed"] = int(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(structure, args.output)
    sidecar = os.path.splitext(args.output)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "input": args.input,
                "output": args.output,
                "variant": args.variant,
                "seed": int(args.seed),
                "topology": structure.get("topology"),
            },
            fh,
            indent=2,
        )
    print("saved", args.output)


if __name__ == "__main__":
    main()
