import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

PLUGIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)

from core.plugin_head import TreeRingConfig, TreeRingHead
from tools.common import iter_feature_batches, load_feature_file, packet_from_batch, weighted_f1_acc


def entropy(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp_min(1e-8)
    return -(prob * prob.log()).sum(dim=-1)


def margin(prob: torch.Tensor) -> torch.Tensor:
    top2 = prob.topk(k=min(2, prob.shape[-1]), dim=-1).values
    if prob.shape[-1] == 1:
        return torch.ones_like(top2[:, 0])
    return top2[:, 0] - top2[:, 1]


def load_plugin(checkpoint_path: str, structure_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    structure = torch.load(structure_path or checkpoint.get("structure_path"), map_location="cpu")
    cfg_dict = dict(checkpoint.get("cfg", structure.get("cfg", {})))
    if "classwise_source_reliability" not in cfg_dict:
        cfg_dict["classwise_source_reliability"] = False
    if "use_sample_residual_gate" not in cfg_dict:
        cfg_dict["use_sample_residual_gate"] = False
    if "enable_base_calibration" not in cfg_dict:
        cfg_dict["enable_base_calibration"] = False
    if "relation_refinement_mode" not in cfg_dict:
        cfg_dict["relation_refinement_mode"] = "additive"
    if "relation_refinement_clip" not in cfg_dict:
        cfg_dict["relation_refinement_clip"] = 5.0
    cfg = TreeRingConfig(**cfg_dict)
    plugin = TreeRingHead(checkpoint["feature_dims"], checkpoint["num_classes"], cfg).to(device)
    plugin.load_structure(structure)
    state = dict(checkpoint["plugin"])
    if "source_rel.scorer.0.weight" in state and "source_rel.encoder.0.weight" not in state:
        state["source_rel.encoder.0.weight"] = state["source_rel.scorer.0.weight"]
        state["source_rel.encoder.0.bias"] = state["source_rel.scorer.0.bias"]
        state["source_rel.global_scorer.weight"] = state["source_rel.scorer.2.weight"]
        state["source_rel.global_scorer.bias"] = state["source_rel.scorer.2.bias"]
    plugin.load_state_dict(state, strict=False)
    plugin.eval()
    return plugin


@torch.no_grad()
def collect(plugin, features: Dict, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    rows = {
        "labels": [],
        "base_logits": [],
        "calibrated_base_logits": [],
        "tree_logits": [],
        "plugin_logits": [],
        "final_logits": [],
        "conflict": [],
        "residual_gate": [],
    }
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        packet = packet_from_batch(batch)
        final_logits, aux = plugin(packet)
        rows["labels"].append(batch["labels"].detach().cpu())
        rows["base_logits"].append(packet.base_logits.detach().cpu())
        rows["calibrated_base_logits"].append(aux.get("calibrated_base_logits", packet.base_logits).detach().cpu())
        rows["tree_logits"].append(aux["tree_logits"].detach().cpu())
        rows["plugin_logits"].append(aux["plugin_logits"].detach().cpu())
        rows["final_logits"].append(final_logits.detach().cpu())
        rows["conflict"].append(aux["conflict_gate"].detach().cpu())
        rows["residual_gate"].append(aux.get("residual_gate", torch.ones_like(aux["plugin_logits"])).detach().cpu())
    return {name: torch.cat(values, dim=0) for name, values in rows.items()}


def logits_key(name: str) -> str:
    if name == "base":
        return "calibrated_base_logits"
    return f"{name}_logits"


def make_feature_bank(rows: Dict[str, torch.Tensor], start_name: str, alt_name: str) -> Dict[str, torch.Tensor]:
    base_logits = rows[logits_key(start_name)]
    alt_logits = rows[logits_key(alt_name)]
    raw_base_logits = rows["base_logits"]
    base_prob = F.softmax(base_logits, dim=-1)
    raw_base_prob = F.softmax(raw_base_logits, dim=-1)
    alt_prob = F.softmax(alt_logits, dim=-1)
    base_pred = base_prob.argmax(dim=-1)
    alt_pred = alt_prob.argmax(dim=-1)
    base_selected = base_prob.gather(1, base_pred.view(-1, 1)).squeeze(1)
    alt_selected = alt_prob.gather(1, alt_pred.view(-1, 1)).squeeze(1)
    alt_on_base = alt_prob.gather(1, base_pred.view(-1, 1)).squeeze(1)
    base_on_alt = base_prob.gather(1, alt_pred.view(-1, 1)).squeeze(1)
    residual_gate_alt = rows["residual_gate"].gather(1, alt_pred.view(-1, 1)).squeeze(1)
    conflict_alt = rows["conflict"].gather(1, alt_pred.view(-1, 1)).squeeze(1)
    return {
        "base_pred": base_pred,
        "alt_pred": alt_pred,
        "base_conf": base_selected,
        "alt_conf": alt_selected,
        "base_entropy": entropy(base_prob),
        "alt_entropy": entropy(alt_prob),
        "base_margin": margin(base_prob),
        "alt_margin": margin(alt_prob),
        "alt_minus_base_conf": alt_selected - base_selected,
        "alt_minus_base_on_alt": alt_selected - base_on_alt,
        "base_raw_shift": (base_prob - raw_base_prob).abs().sum(dim=-1),
        "prob_margin_diff": margin(alt_prob) - margin(base_prob),
        "logit_margin_diff": margin(alt_logits) - margin(base_logits),
        "conflict_alt": conflict_alt,
        "residual_gate_alt": residual_gate_alt,
        "disagree": (base_pred != alt_pred).float(),
    }


def apply_rule(pred: torch.Tensor, bank: Dict[str, torch.Tensor], rule: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = bank[rule["feature"]]
    if rule["op"] == "ge":
        mask = scores >= rule["threshold"]
    else:
        mask = scores <= rule["threshold"]
    if rule["from_class"] >= 0:
        mask = mask & (bank["base_pred"] == rule["from_class"])
    if rule["to_class"] >= 0:
        mask = mask & (bank["alt_pred"] == rule["to_class"])
    mask = mask & (bank["base_pred"] != bank["alt_pred"])
    out = pred.clone()
    out[mask] = bank["alt_pred"][mask]
    return out, mask


def metric(labels: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    labels = labels.long()
    pred = pred.long()
    classes = torch.arange(int(labels.max().item()) + 1)
    acc = (labels == pred).float().mean()
    f1_terms = []
    weights = []
    for cls in classes:
        true_c = labels == cls
        pred_c = pred == cls
        tp = (true_c & pred_c).float().sum()
        fp = (~true_c & pred_c).float().sum()
        fn = (true_c & ~pred_c).float().sum()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
        support = true_c.float().sum()
        f1_terms.append(f1)
        weights.append(support)
    f1_tensor = torch.stack(f1_terms)
    weight_tensor = torch.stack(weights)
    f1_w = (f1_tensor * weight_tensor / weight_tensor.sum().clamp_min(1.0)).sum()
    return {"accuracy": float(acc.item()), "f1_w": float(f1_w.item())}


def score(metrics: Dict[str, float], mode: str) -> float:
    if mode == "acc":
        return metrics["accuracy"]
    if mode == "f1":
        return metrics["f1_w"]
    return metrics["accuracy"] + metrics["f1_w"]


def candidate_rules(bank: Dict[str, torch.Tensor], num_classes: int, max_thresholds: int) -> List[Dict]:
    features = [
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
    rules = []
    classes = [-1] + list(range(num_classes))
    for feature in features:
        values = bank[feature][bank["base_pred"] != bank["alt_pred"]]
        if values.numel() == 0:
            continue
        qs = torch.linspace(0.05, 0.95, steps=max_thresholds)
        thresholds = torch.quantile(values.float(), qs).unique().tolist()
        for threshold in thresholds:
            for op in ("le", "ge"):
                for from_class in classes:
                    for to_class in classes:
                        if from_class < 0 and to_class < 0:
                            continue
                        rules.append(
                            {
                                "feature": feature,
                                "op": op,
                                "threshold": float(threshold),
                                "from_class": int(from_class),
                                "to_class": int(to_class),
                            }
                        )
    return rules


def greedy_search(dev, test, start_name: str, alt_name: str, max_rules: int, max_thresholds: int, rank_by: str):
    num_classes = int(dev["labels"].max().item()) + 1
    dev_bank = make_feature_bank(dev, start_name, alt_name)
    test_bank = make_feature_bank(test, start_name, alt_name)
    dev_pred = dev[logits_key(start_name)].argmax(dim=-1)
    test_pred = test[logits_key(start_name)].argmax(dim=-1)
    base_dev_metric = metric(dev["labels"], dev_pred)
    base_test_metric = metric(test["labels"], test_pred)
    best_metric = base_dev_metric
    rules_out = []
    tried = candidate_rules(dev_bank, num_classes, max_thresholds)

    for _ in range(max_rules):
        best = None
        for rule in tried:
            cand_dev_pred, dev_mask = apply_rule(dev_pred, dev_bank, rule)
            if int(dev_mask.sum().item()) == 0:
                continue
            cand_metric = metric(dev["labels"], cand_dev_pred)
            cand_score = score(cand_metric, rank_by)
            if cand_score <= score(best_metric, rank_by):
                continue
            cand_test_pred, test_mask = apply_rule(test_pred, test_bank, rule)
            item = {
                "rule": rule,
                "dev_metric": cand_metric,
                "test_metric": metric(test["labels"], cand_test_pred),
                "dev_switched": int(dev_mask.sum().item()),
                "test_switched": int(test_mask.sum().item()),
                "dev_pred": cand_dev_pred,
                "test_pred": cand_test_pred,
                "score": cand_score,
            }
            if best is None or item["score"] > best["score"]:
                best = item
        if best is None:
            break
        rules_out.append(
            {
                "rule": best["rule"],
                "dev_metric": best["dev_metric"],
                "test_metric": best["test_metric"],
                "dev_switched": best["dev_switched"],
                "test_switched": best["test_switched"],
            }
        )
        dev_pred = best["dev_pred"]
        test_pred = best["test_pred"]
        best_metric = best["dev_metric"]

    return {
        "alt_logits": alt_name,
        "start_logits": start_name,
        "rank_by": rank_by,
        "base_dev": base_dev_metric,
        "base_test": base_test_metric,
        "final_dev": metric(dev["labels"], dev_pred),
        "final_test": metric(test["labels"], test_pred),
        "rules": rules_out,
    }


def main():
    parser = argparse.ArgumentParser(description="TreeRing selective tree/listening rescue on cached features.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--dev-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-logits", default="base")
    parser.add_argument("--alt-logits", default="tree,plugin,final")
    parser.add_argument("--rank-by", default="sum", choices=["sum", "acc", "f1"])
    parser.add_argument("--max-rules", type=int, default=4)
    parser.add_argument("--max-thresholds", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    plugin = load_plugin(args.checkpoint, args.structure, device)
    dev = collect(plugin, load_feature_file(args.dev_features), args.batch_size, device)
    test = collect(plugin, load_feature_file(args.test_features), args.batch_size, device)

    records = []
    for start_name in [x.strip() for x in args.start_logits.split(",") if x.strip()]:
        for alt_name in [x.strip() for x in args.alt_logits.split(",") if x.strip()]:
            if start_name == alt_name:
                continue
            records.append(greedy_search(dev, test, start_name, alt_name, args.max_rules, args.max_thresholds, args.rank_by))
    best = max(records, key=lambda item: score(item["final_dev"], args.rank_by))
    result = {
        "format": "treering_rescue_search_v1",
        "checkpoint": args.checkpoint,
        "structure": args.structure,
        "rank_by": args.rank_by,
        "records": records,
        "best": best,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(best, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
