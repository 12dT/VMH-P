import torch

from core.packet import EvidencePacket
from .base_adapter import BaseBackboneAdapter


class FrozenDualTowerAdapter(BaseBackboneAdapter):
    def __init__(self, encoder, classifier, device=None, detach_backbone: bool = True, backbone_name: str = "clip"):
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        self.detach_backbone = bool(detach_backbone)
        self.backbone_name = backbone_name

    def extract(self, batch) -> EvidencePacket:
        text_inputs, image_inputs, labels = batch
        if self.device is not None:
            text_inputs = text_inputs.to(self.device)
            image_inputs = image_inputs.to(self.device)
            labels = labels.to(self.device)
        with torch.set_grad_enabled(not self.detach_backbone):
            text_feat, image_feat = self.encoder(text_inputs, image_inputs)
            text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
            image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
            base_logits = self.classifier(torch.cat([text_feat, image_feat], dim=-1))
        if self.detach_backbone:
            text_feat = text_feat.detach()
            image_feat = image_feat.detach()
            base_logits = base_logits.detach()
        return EvidencePacket(text_feat, image_feat, None, base_logits, labels, {"backbone": self.backbone_name})

    def freeze_backbone(self):
        self.encoder.eval()
        self.classifier.eval()
        for module in (self.encoder, self.classifier):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_for_joint_tuning(self):
        self.encoder.train()
        self.classifier.train()
        for module in (self.encoder, self.classifier):
            for param in module.parameters():
                param.requires_grad = True
