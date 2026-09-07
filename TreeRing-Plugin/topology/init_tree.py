from core.plugin_head import TreeRingConfig, TreeRingHead
from .build_tree import build_tree_topology


def build_structure_from_centers(class_centers, feature_dims, num_classes, cfg=None, num_internal=2):
    cfg = cfg or TreeRingConfig(num_internal=num_internal)
    cfg.num_internal = int(num_internal)
    topology = build_tree_topology(class_centers, num_internal=num_internal, random_state=42, n_init=50)
    plugin = TreeRingHead(feature_dims, num_classes, cfg)
    plugin.initialize_from_class_centers(class_centers, topology=topology)
    structure = plugin.export_structure()
    structure["class_centers"] = class_centers.detach().cpu()
    structure["feature_dims"] = feature_dims
    structure["num_classes"] = int(num_classes)
    structure["cfg"] = cfg.__dict__
    return structure
