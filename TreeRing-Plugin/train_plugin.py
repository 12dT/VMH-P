import argparse
import copy
import os
import sys

import torch

PLUGIN_ROOT = os.path.abspath(os.path.dirname(__file__))
if PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, PLUGIN_ROOT)
TOOLS_DIR = os.path.join(PLUGIN_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from core.losses import treering_loss
from core.plugin_head import TreeRingConfig, TreeRingHead
from tools.common import infer_feature_dims, iter_feature_batches, load_feature_file, num_classes_from_features, packet_from_batch, weighted_f1_acc


def build_class_weight(labels: torch.Tensor, num_classes: int, mode: str, device: torch.device):
    if mode == "none":
        return None
    counts = torch.bincount(labels.long().cpu(), minlength=num_classes).float().clamp_min(1.0)
    inv = counts.sum() / (float(num_classes) * counts)
    if mode == "sqrt_balanced":
        inv = inv.sqrt()
    inv = inv / inv.mean().clamp_min(1e-8)
    return inv.to(device)


def metric_score(metrics, mode: str):
    acc, f1 = metrics
    if mode == "acc":
        return acc
    if mode == "f1":
        return f1
    return acc + f1


@torch.no_grad()
def evaluate(plugin, features, batch_size, device):
    plugin.eval()
    rows = {"final": ([], []), "tree": ([], []), "plugin": ([], []), "base": ([], [])}
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        packet = packet_from_batch(batch)
        final_logits, aux = plugin(packet)
        logits_map = {
            "final": final_logits,
            "tree": aux["tree_logits"],
            "plugin": aux["plugin_logits"],
            "base": packet.base_logits,
        }
        labels = batch["labels"].detach().cpu().tolist()
        for key, logits in logits_map.items():
            rows[key][0].extend(labels)
            rows[key][1].extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return {key: weighted_f1_acc(y, p) for key, (y, p) in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--test-features", default=None)
    parser.add_argument("--structure", required=True, help="Path to the induced VMH-P structure.pt file.")
    parser.add_argument("--ufa-warmup", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--plugin-lr", type=float, default=2e-4)
    parser.add_argument("--ufa-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-tree", type=float, default=0.5)
    parser.add_argument("--lambda-plugin", type=float, default=0.5)
    parser.add_argument("--lambda-thc", type=float, default=0.1)
    parser.add_argument("--lambda-rel", type=float, default=0.01)
    parser.add_argument("--class-weight", choices=["none", "balanced", "sqrt_balanced"], default="none")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--select-by-test", action="store_true", help="Diagnostic mode: save the best test checkpoint.")
    parser.add_argument("--rank-by", choices=["sum", "acc", "f1"], default="f1")
    parser.add_argument("--plugin-gate-init", type=float, default=None)
    parser.add_argument("--plugin-scale-init", type=float, default=None)
    parser.add_argument("--geometry-mode", choices=["hyperbolic", "euclidean"], default=None)
    parser.add_argument("--relation-refinement-mode", choices=["multiplicative", "multiplicative_gain", "additive"], default=None)
    parser.add_argument("--relation-refinement-clip", type=float, default=None)
    parser.add_argument("--relation-gain-clip", type=float, default=None)
    parser.add_argument("--relation-empty-row-mass", type=float, default=None)
    parser.add_argument("--relation-empty-row-floor", type=float, default=None)
    parser.add_argument("--enable-base-calibration", action="store_true")
    parser.add_argument("--disable-sample-residual-gate", action="store_true")
    parser.add_argument("--disable-classwise-source-reliability", action="store_true")
    parser.add_argument("--source-prob-bias-init", type=float, default=None)
    parser.add_argument("--enable-evidence-modulation", action="store_true")
    parser.add_argument("--evidence-alpha-pos-init", type=float, default=None)
    parser.add_argument("--evidence-alpha-neg-init", type=float, default=None)
    parser.add_argument("--horizontal-only", action="store_true")
    parser.add_argument("--sample-gate-init", type=float, default=None)
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
    test = load_feature_file(args.test_features) if args.test_features else None
    structure = torch.load(args.structure, map_location="cpu")
    warm = torch.load(args.ufa_warmup, map_location="cpu") if args.ufa_warmup else None

    cfg = TreeRingConfig(**structure.get("cfg", {}))
    if args.geometry_mode is not None:
        cfg.geometry_mode = args.geometry_mode
    if args.relation_refinement_mode is not None:
        cfg.relation_refinement_mode = args.relation_refinement_mode
    if args.relation_refinement_clip is not None:
        cfg.relation_refinement_clip = float(args.relation_refinement_clip)
    if args.relation_gain_clip is not None:
        cfg.relation_gain_clip = float(args.relation_gain_clip)
    if args.relation_empty_row_mass is not None:
        cfg.relation_empty_row_mass = float(args.relation_empty_row_mass)
    if args.relation_empty_row_floor is not None:
        cfg.relation_empty_row_floor = float(args.relation_empty_row_floor)
    cfg.enable_base_calibration = bool(args.enable_base_calibration)
    if args.disable_sample_residual_gate:
        cfg.use_sample_residual_gate = False
    if args.disable_classwise_source_reliability:
        cfg.classwise_source_reliability = False
    if args.source_prob_bias_init is not None:
        cfg.source_prob_bias_init = float(args.source_prob_bias_init)
    if args.enable_evidence_modulation:
        cfg.use_evidence_modulation = True
    if args.evidence_alpha_pos_init is not None:
        cfg.evidence_alpha_pos_init = float(args.evidence_alpha_pos_init)
    if args.evidence_alpha_neg_init is not None:
        cfg.evidence_alpha_neg_init = float(args.evidence_alpha_neg_init)
    if args.horizontal_only:
        cfg.horizontal_only = True
    if args.sample_gate_init is not None:
        cfg.sample_gate_init = float(args.sample_gate_init)
    feature_dims = structure.get("feature_dims") or (warm.get("feature_dims") if warm else infer_feature_dims(train))
    num_classes = int(structure.get("num_classes") or (warm.get("num_classes") if warm else num_classes_from_features(train, val)))
    class_weight = build_class_weight(train["labels"], num_classes, args.class_weight, device)
    plugin = TreeRingHead(feature_dims, num_classes, cfg).to(device)
    if warm is not None:
        plugin.ufa.load_state_dict(warm["ufa"], strict=True)
    plugin.load_structure(structure)
    if args.plugin_gate_init is not None:
        from core.conflict_gate import _logit

        with torch.no_grad():
            plugin.raw_plugin_gate.copy_(_logit(args.plugin_gate_init).to(device=plugin.raw_plugin_gate.device))
    if args.plugin_scale_init is not None:
        from core.conflict_gate import _inv_softplus

        with torch.no_grad():
            plugin.raw_plugin_scale.copy_(_inv_softplus(args.plugin_scale_init).to(device=plugin.raw_plugin_scale.device))

    optimizer = torch.optim.AdamW(
        [
            {"params": plugin.ufa.parameters(), "lr": args.ufa_lr},
            {"params": plugin.non_ufa_parameters(), "lr": args.plugin_lr},
        ],
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        plugin.train()
        total = 0.0
        count = 0
        for batch in iter_feature_batches(train, args.batch_size, shuffle=True, device=device):
            labels = batch["labels"].long()
            packet = packet_from_batch(batch)
            final_logits, aux = plugin(packet)
            loss_info = treering_loss(
                final_logits,
                aux,
                labels,
                plugin=plugin,
                lambda_tree=args.lambda_tree,
                lambda_plugin=args.lambda_plugin,
                lambda_thc=args.lambda_thc,
                lambda_rel=args.lambda_rel,
                class_weight=class_weight,
                label_smoothing=args.label_smoothing,
            )
            optimizer.zero_grad()
            loss_info["loss"].backward()
            torch.nn.utils.clip_grad_norm_(plugin.parameters(), 5.0)
            optimizer.step()
            total += float(loss_info["loss"].item()) * labels.numel()
            count += labels.numel()

        val_metrics = evaluate(plugin, val, args.batch_size, device)
        test_metrics = evaluate(plugin, test, args.batch_size, device) if (test is not None and args.select_by_test) else None
        val_acc, val_f1 = val_metrics["final"]
        print(
            "epoch %03d loss %.6f val_final %.6f/%.6f val_tree %.6f/%.6f val_plugin %.6f/%.6f gate %.6f"
            % (
                epoch,
                total / max(count, 1),
                val_acc,
                val_f1,
                val_metrics["tree"][0],
                val_metrics["tree"][1],
                val_metrics["plugin"][0],
                val_metrics["plugin"][1],
                float(plugin.plugin_gate.detach().cpu()),
            )
        )
        if test_metrics is not None:
            print(
                "epoch %03d test_final %.6f/%.6f test_tree %.6f/%.6f test_plugin %.6f/%.6f"
                % (
                    epoch,
                    test_metrics["final"][0],
                    test_metrics["final"][1],
                    test_metrics["tree"][0],
                    test_metrics["tree"][1],
                    test_metrics["plugin"][0],
                    test_metrics["plugin"][1],
                )
            )
        select_metrics = test_metrics if args.select_by_test and test_metrics is not None else val_metrics
        select_score = metric_score(select_metrics["final"], args.rank_by)
        if select_score > best_f1:
            bad_epochs = 0
            best_f1 = select_score
            best_state = {
                "plugin": copy.deepcopy(plugin.state_dict()),
                "cfg": cfg.__dict__,
                "feature_dims": feature_dims,
                "num_classes": num_classes,
                "structure_path": args.structure,
                "best_val_metrics": val_metrics,
                "best_test_metrics": test_metrics,
                "selection_split": "test" if args.select_by_test and test_metrics is not None else "val",
                "selection_metric": args.rank_by,
                "selection_score": select_score,
                "class_weight": class_weight.detach().cpu() if class_weight is not None else None,
                "class_weight_mode": args.class_weight,
                "label_smoothing": args.label_smoothing,
                "loss_weights": {
                    "lambda_tree": args.lambda_tree,
                    "lambda_plugin": args.lambda_plugin,
                    "lambda_thc": args.lambda_thc,
                    "lambda_rel": args.lambda_rel,
                },
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_plugin.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        plugin.load_state_dict(best_state["plugin"], strict=True)
    final_val = evaluate(plugin, val, args.batch_size, device)
    print("best val", final_val)
    if test is not None:
        print("test", evaluate(plugin, test, args.batch_size, device))


if __name__ == "__main__":
    main()
