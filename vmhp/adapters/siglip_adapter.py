from .clip_adapter import FrozenDualTowerAdapter


class SigLIPAdapter(FrozenDualTowerAdapter):
    def __init__(self, encoder, classifier, device=None, detach_backbone: bool = True):
        super().__init__(encoder, classifier, device=device, detach_backbone=detach_backbone, backbone_name="siglip")
