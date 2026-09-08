import argparse
import copy
import os

import torch

from vmhp.core.losses import vmhp_loss
from vmhp.core.vmhp_head import VMHPConfig, VMHPHead
from vmhp.tools.common import (
    infer_feature_dims,
    iter_feature_batches,
    load_feature_file,
    num_classes_from_features,
    packet_from_batch,
    weighted_f1_acc,
)


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
def evaluate(model, features, batch_size, device):
    model.eval()
    rows = {"final": ([], []), "tree": ([], []), "structured": ([], []), "plugin": ([], []), "base": ([], [])}
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        packet = packet_from_batch(batch)
        final_logits, aux = model(packet)
        logits_map = {
            "final": final_logits,
            "tree": aux["tree_logits"],
            "structured": aux["structured_logits"],
            "plugin": aux["plugin_logits"],
            "base": packet.base_logits,
        }
        labels = batch["labels"].detach().cpu().tolist()
        for key, logits in logits_map.items():
            rows[key][0].extend(labels)
            rows[key][1].extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return {key: weighted_f1_acc(y, p) for key, (y, p) in rows.items()}


def main():
    parser = argparse.ArgumentParser(description="Train VMH-P on cached frozen-host representations.")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--test-features", default=None)
    parser.add_argument("--structure", required=True, help="Path to the induced VMH-P structure.pt file.")
    parser.add_argument("--ufa-warmup", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--structured-lr", type=float, default=3e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--plugin-lr", type=float, default=None, help="Deprecated alias for --structured-lr.")
    parser.add_argument("--ufa-lr", type=float, default=None, help="Deprecated alias for --adapter-lr.")
    parser.add_argument("--train-adapters", action="store_true", help="Fine-tune evidence adapters during VMH-P training.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-str", type=float, default=0.10)
    parser.add_argument("--lambda-vh", type=float, default=0.10)
    parser.add_argument("--lambda-hr", type=float, default=0.01)
    parser.add_argument("--class-weight", choices=["none", "balanced", "sqrt_balanced"], default="none")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--select-by-test", action="store_true", help="Diagnostic mode: save the best test checkpoint.")
    parser.add_argument("--rank-by", choices=["sum", "acc", "f1"], default="sum")
    parser.add_argument("--geometry-mode", choices=["hyperbolic", "euclidean"], default=None)
    parser.add_argument("--num-internal", type=int, default=None)
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--lambda-minus", type=float, default=None)
    parser.add_argument("--residual-scale-init", type=float, default=None)
    parser.add_argument("--relation-refinement-mode", choices=["multiplicative", "multiplicative_gain", "additive"], default=None)
    parser.add_argument("--relation-refinement-clip", type=float, default=None)
    parser.add_argument("--relation-gain-clip", type=float, default=None)
    parser.add_argument("--relation-empty-row-mass", type=float, default=None)
    parser.add_argument("--relation-empty-row-floor", type=float, default=None)
    parser.add_argument("--enable-base-calibration", action="store_true")
    parser.add_argument("--horizontal-only", action="store_true")
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

    cfg = VMHPConfig.from_dict(structure.get("cfg", {}))
    if args.geometry_mode is not None:
        cfg.geometry_mode = args.geometry_mode
    if args.num_internal is not None:
        cfg.num_internal = int(args.num_internal)
    if args.curvature is not None:
        cfg.curvature = float(args.curvature)
    if args.eta is not None:
        cfg.eta = float(args.eta)
    if args.lambda_minus is not None:
        cfg.lambda_minus = float(args.lambda_minus)
    if args.residual_scale_init is not None:
        cfg.residual_scale_init = float(args.residual_scale_init)
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
    if args.horizontal_only:
        cfg.horizontal_only = True

    structured_lr = args.plugin_lr if args.plugin_lr is not None else args.structured_lr
    adapter_lr = args.ufa_lr if args.ufa_lr is not None else args.adapter_lr
    feature_dims = structure.get("feature_dims") or (warm.get("feature_dims") if warm else infer_feature_dims(train))
    num_classes = int(structure.get("num_classes") or (warm.get("num_classes") if warm else num_classes_from_features(train, val)))
    class_weight = build_class_weight(train["labels"], num_classes, args.class_weight, device)

    model = VMHPHead(feature_dims, num_classes, cfg).to(device)
    if warm is not None:
        model.ufa.load_state_dict(warm["ufa"], strict=True)
    model.load_structure(structure)
    if warm is not None and not args.train_adapters:
        for param in model.adapter_parameters():
            param.requires_grad_(False)

    param_groups = []
    adapter_params = [param for param in model.adapter_parameters() if param.requires_grad]
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": adapter_lr})
    param_groups.append({"params": [param for param in model.non_adapter_parameters() if param.requires_grad], "lr": structured_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_score = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in iter_feature_batches(train, args.batch_size, shuffle=True, device=device):
            labels = batch["labels"].long()
            packet = packet_from_batch(batch)
            final_logits, aux = model(packet)
            loss_info = vmhp_loss(
                final_logits,
                aux,
                labels,
                model=model,
                lambda_str=args.lambda_str,
                lambda_vh=args.lambda_vh,
                lambda_hr=args.lambda_hr,
                class_weight=class_weight,
                label_smoothing=args.label_smoothing,
            )
            optimizer.zero_grad()
            loss_info["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss_info["loss"].item()) * labels.numel()
            count += labels.numel()

        val_metrics = evaluate(model, val, args.batch_size, device)
        test_metrics = evaluate(model, test, args.batch_size, device) if (test is not None and args.select_by_test) else None
        print(
            "epoch %03d loss %.6f val_final %.6f/%.6f val_vertical %.6f/%.6f val_struct %.6f/%.6f residual %.6f"
            % (
                epoch,
                total / max(count, 1),
                val_metrics["final"][0],
                val_metrics["final"][1],
                val_metrics["tree"][0],
                val_metrics["tree"][1],
                val_metrics["structured"][0],
                val_metrics["structured"][1],
                float(model.residual_scale.detach().cpu()),
            )
        )
        if test_metrics is not None:
            print(
                "epoch %03d test_final %.6f/%.6f test_vertical %.6f/%.6f test_struct %.6f/%.6f"
                % (
                    epoch,
                    test_metrics["final"][0],
                    test_metrics["final"][1],
                    test_metrics["tree"][0],
                    test_metrics["tree"][1],
                    test_metrics["structured"][0],
                    test_metrics["structured"][1],
                )
            )

        select_metrics = test_metrics if args.select_by_test and test_metrics is not None else val_metrics
        select_score = metric_score(select_metrics["final"], args.rank_by)
        if select_score > best_score:
            bad_epochs = 0
            best_score = select_score
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "cfg": cfg.to_dict(),
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
                "train_adapters": bool(args.train_adapters),
                "adapter_lr": adapter_lr,
                "structured_lr": structured_lr,
                "loss_weights": {
                    "lambda_str": args.lambda_str,
                    "lambda_vh": args.lambda_vh,
                    "lambda_hr": args.lambda_hr,
                },
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_vmhp.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"], strict=True)
    final_val = evaluate(model, val, args.batch_size, device)
    print("best val", final_val)
    if test is not None:
        print("test", evaluate(model, test, args.batch_size, device))


if __name__ == "__main__":
    main()
