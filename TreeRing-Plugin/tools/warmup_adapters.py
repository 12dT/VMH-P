import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import infer_feature_dims, iter_feature_batches, load_feature_file, num_classes_from_features, weighted_f1_acc
from core.plugin_head import TreeRingConfig, TreeRingHead


def project_mean(plugin: TreeRingHead, batch):
    zs = []
    for name in plugin.ufa:
        if batch.get(name) is not None:
            zs.append(plugin.ufa[name](batch[name]))
    if not zs:
        raise ValueError("batch has no usable source features")
    return torch.stack(zs, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate(plugin, classifier, features, batch_size, device):
    plugin.eval()
    classifier.eval()
    y_true, y_pred = [], []
    for batch in iter_feature_batches(features, batch_size, shuffle=False, device=device):
        logits = classifier(project_mean(plugin, batch))
        pred = logits.argmax(dim=-1)
        y_true.extend(batch["labels"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    return weighted_f1_acc(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tree-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    train = load_feature_file(args.train_features)
    val = load_feature_file(args.val_features)
    feature_dims = infer_feature_dims(train)
    num_classes = num_classes_from_features(train, val)

    cfg = TreeRingConfig(tree_dim=args.tree_dim, dropout=args.dropout)
    plugin = TreeRingHead(feature_dims, num_classes, cfg).to(device)
    classifier = nn.Linear(args.tree_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(list(plugin.ufa.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=1e-4)

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        plugin.train()
        classifier.train()
        total = 0.0
        count = 0
        for batch in iter_feature_batches(train, args.batch_size, shuffle=True, device=device):
            labels = batch["labels"].long()
            logits = classifier(project_mean(plugin, batch))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(plugin.ufa.parameters()) + list(classifier.parameters()), 5.0)
            optimizer.step()
            total += float(loss.item()) * labels.numel()
            count += labels.numel()
        val_acc, val_f1 = evaluate(plugin, classifier, val, args.batch_size, device)
        print("epoch %03d warm_loss %.6f val_acc %.6f val_f1 %.6f" % (epoch, total / max(count, 1), val_acc, val_f1))
        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            best_state = {
                "ufa": copy.deepcopy(plugin.ufa.state_dict()),
                "classifier": copy.deepcopy(classifier.state_dict()),
                "feature_dims": feature_dims,
                "num_classes": num_classes,
                "cfg": cfg.__dict__,
                "best_val_f1": best_f1,
            }
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(best_state, args.output)
    print("saved", args.output, "best_val_f1", best_f1)


if __name__ == "__main__":
    main()
