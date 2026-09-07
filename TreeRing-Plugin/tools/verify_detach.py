import argparse
import os

import torch

from common import infer_feature_dims, iter_feature_batches, load_feature_file, num_classes_from_features, packet_from_batch
from core.plugin_head import TreeRingConfig, TreeRingHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-num", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    features = load_feature_file(args.features)
    structure = torch.load(args.structure, map_location="cpu") if args.structure else {}
    ckpt = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    cfg_dict = dict(structure.get("cfg", {}) or (ckpt.get("cfg", {}) if ckpt else {}))
    if ckpt is not None and "relation_refinement_mode" not in cfg_dict:
        cfg_dict["relation_refinement_mode"] = "additive"
    cfg = TreeRingConfig(**cfg_dict)
    feature_dims = structure.get("feature_dims") or (ckpt.get("feature_dims") if ckpt else infer_feature_dims(features))
    num_classes = int(structure.get("num_classes") or (ckpt.get("num_classes") if ckpt else num_classes_from_features(features)))
    plugin = TreeRingHead(feature_dims, num_classes, cfg).to(device)
    if structure:
        plugin.load_structure(structure)
    if ckpt is not None:
        plugin.load_state_dict(ckpt["plugin"], strict=True)
    plugin.eval()

    max_abs = 0.0
    with torch.no_grad():
        for batch in iter_feature_batches(features, args.batch_size, shuffle=False, device=device):
            packet = packet_from_batch(batch)
            disabled, _ = plugin(packet, use_plugin=False)
            max_abs = max(max_abs, float((disabled - packet.base_logits).abs().max().detach().cpu()))
    print("detach max_abs %.10f" % max_abs)
    if max_abs >= 1e-6:
        raise SystemExit("detach verification failed")


if __name__ == "__main__":
    main()
