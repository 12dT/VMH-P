from vmhp.core.vmhp_head import VMHPConfig, VMHPHead
from .build_tree import build_tree_topology


def build_structure_from_centers(class_centers, feature_dims, num_classes, cfg=None, num_internal=2):
    cfg = cfg or VMHPConfig(num_internal=num_internal)
    cfg.num_internal = int(num_internal)
    topology = build_tree_topology(class_centers, num_internal=num_internal, random_state=42, n_init=50)
    model = VMHPHead(feature_dims, num_classes, cfg)
    model.initialize_from_class_centers(class_centers, topology=topology)
    structure = model.export_structure()
    structure["class_centers"] = class_centers.detach().cpu()
    structure["feature_dims"] = feature_dims
    structure["num_classes"] = int(num_classes)
    structure["cfg"] = cfg.to_dict()
    return structure
