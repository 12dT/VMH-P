import contextlib
from typing import Optional

import torch

from core.packet import EvidencePacket
from .base_adapter import BaseBackboneAdapter


class D2RAdapter(BaseBackboneAdapter):
    """Expose D2R evidence without changing its representation pipeline.

    Main TreeRing inputs stay backbone-agnostic:
    text = pre-interaction BERT global evidence,
    vision = pre-interaction ViT global evidence,
    multimodal = D2R final Block-Fusion feature.
    D2R's own routed branch features are intentionally not packet sources.
    """

    def __init__(self, backbone, device: Optional[torch.device] = None, detach_backbone: bool = True):
        self.backbone = backbone
        self.device = device
        self.detach_backbone = bool(detach_backbone)

    def _move(self, tensor):
        if self.device is None or tensor is None:
            return tensor
        return tensor.to(self.device) if torch.is_tensor(tensor) else tensor

    def _unpack_backbone_output(self, output):
        if isinstance(output, dict):
            features = dict(output.get("features", {}))
            features.setdefault("text_global", output["text_global"])
            features.setdefault("vision_global", output["vision_global"])
            features.setdefault("fusion_feature", output["fusion_feature"])
            return output["logits"], features
        loss, logits, features = output
        return logits, features

    def forward(self, input_ids, attention_mask, token_type_ids, images, labels=None) -> EvidencePacket:
        input_ids = self._move(input_ids)
        attention_mask = self._move(attention_mask)
        token_type_ids = self._move(token_type_ids)
        images = self._move(images)
        labels = self._move(labels)

        grad_enabled = (not self.detach_backbone) and any(p.requires_grad for p in self.backbone.parameters())
        guard = contextlib.nullcontext() if grad_enabled else torch.no_grad()
        with guard:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                images=images,
                return_features=True,
            )
        logits, features = self._unpack_backbone_output(output)
        text = features["text_global"]
        vision = features["vision_global"]
        multimodal = features["fusion_feature"]
        if self.detach_backbone:
            logits = logits.detach()
            text = text.detach()
            vision = vision.detach()
            multimodal = multimodal.detach()
        return EvidencePacket(
            text=text,
            vision=vision,
            multimodal=multimodal,
            base_logits=logits,
            label=labels,
            meta={
                "backbone": "d2r",
                "feature_dims": {
                    "text": int(text.shape[-1]),
                    "vision": int(vision.shape[-1]),
                    "multimodal": int(multimodal.shape[-1]),
                },
            },
        )

    def extract(self, batch) -> EvidencePacket:
        input_ids, attention_mask, token_type_ids, _img_mask, labels, images = batch
        return self.forward(input_ids, attention_mask, token_type_ids, images, labels=labels)

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_for_joint_tuning(self):
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
