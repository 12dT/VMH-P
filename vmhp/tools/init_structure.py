import argparse
import json
import os

import torch

from vmhp.tools.common import iter_feature_batches, load_feature_file
from vmhp.core.vmhp_head import VMHPConfig, VMHPHead
from vmhp.topology.build_tree import build_tree_topology
from vmhp.topology.class_centers import compute_class_centers


@torch.no_grad()
def project_all_sources(model, features, batch_size, device):
    projected = {name: [] for name in model.ufa.keys()}
    labels = []
    model.eval()
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        labels.append(batch["labels"].detach().cpu())
        for name in model.ufa:
            if batch.get(name) is not None:
                projected[name].append(model.ufa[name](batch[name]).detach().cpu())
    projected = {name: torch.cat(values, dim=0) for name, values in projected.items() if values}
    return projected, torch.cat(labels, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Induce VMH-P vertical and horizontal structures from cached host features.")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--ufa-warmup", required=True)
    parser.add_argument("--output-dir", default="runs/structures/default")
    parser.add_argument("--num-internal", type=int, default=2)
    parser.add_argument("--tree-induction", default="kmeans", choices=["kmeans"])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    train = load_feature_file(args.train_features)
    warm = torch.load(args.ufa_warmup, map_location="cpu")
    cfg_dict = dict(warm.get("cfg", {}))
    cfg_dict["num_internal"] = args.num_internal
    cfg = VMHPConfig.from_dict(cfg_dict)
    model = VMHPHead(warm["feature_dims"], warm["num_classes"], cfg).to(device)
    model.ufa.load_state_dict(warm["ufa"], strict=True)

    projected, labels = project_all_sources(model, train, args.batch_size, device)
    class_centers = compute_class_centers(projected, labels, warm["num_classes"])
    topology = build_tree_topology(class_centers, num_internal=args.num_internal, random_state=42, n_init=50)

    model.initialize_from_class_centers(class_centers.to(device), topology=topology)
    structure = model.export_structure()
    structure["class_centers"] = class_centers.cpu()
    structure["feature_dims"] = warm["feature_dims"]
    structure["num_classes"] = warm["num_classes"]
    structure["cfg"] = cfg.to_dict()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(structure, os.path.join(args.output_dir, "structure.pt"))
    torch.save(class_centers, os.path.join(args.output_dir, "class_centers.pt"))
    torch.save(structure["node_init"], os.path.join(args.output_dir, "node_init.pt"))
    torch.save(structure["prototype_init"], os.path.join(args.output_dir, "prototype_init.pt"))
    torch.save(structure["edge_cost"], os.path.join(args.output_dir, "edge_cost.pt"))
    torch.save(structure["relation_pos_init"], os.path.join(args.output_dir, "relation_pos_init.pt"))
    torch.save(structure["relation_neg_init"], os.path.join(args.output_dir, "relation_neg_init.pt"))
    with open(os.path.join(args.output_dir, "topology.json"), "w", encoding="utf-8") as fh:
        json.dump(topology, fh, indent=2)
    print("saved induced VMH-P structure to", args.output_dir)


if __name__ == "__main__":
    main()
