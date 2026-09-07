import contextlib
from typing import Optional

import torch

from core.packet import EvidencePacket
from .base_adapter import BaseBackboneAdapter


class CLMLFAdapter(BaseBackboneAdapter):
    def __init__(self, backbone, device: Optional[torch.device] = None, detach_backbone: bool = True):
        self.backbone = backbone
        self.device = device
        self.detach_backbone = bool(detach_backbone)

    def _move(self, tensor):
        if self.device is None or tensor is None:
            return tensor
        if torch.is_tensor(tensor):
            return tensor.to(self.device)
        return tensor

    def _make_model_param(self, texts, bert_attention_mask, images, text_image_mask):
        from model import ModelParam

        param = ModelParam()
        param.set_data_param(
            texts=texts,
            bert_attention_mask=bert_attention_mask,
            images=images,
            text_image_mask=text_image_mask,
        )
        return param

    def extract(self, batch) -> EvidencePacket:
        (
            texts_origin,
            bert_attention_mask,
            image_origin,
            text_image_mask,
            labels,
            *_,
        ) = batch
        texts_origin = self._move(texts_origin)
        bert_attention_mask = self._move(bert_attention_mask)
        image_origin = self._move(image_origin)
        text_image_mask = self._move(text_image_mask)
        labels = self._move(labels)

        param = self._make_model_param(texts_origin, bert_attention_mask, image_origin, text_image_mask)
        grad_enabled = (not self.detach_backbone) and any(p.requires_grad for p in self.backbone.parameters())
        guard = contextlib.nullcontext() if grad_enabled else torch.no_grad()
        with guard:
            logits, aux = self.backbone(param, return_features=True)
        if self.detach_backbone:
            logits = logits.detach()
            aux = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in aux.items()}
        return EvidencePacket(
            text=aux.get("text_feature"),
            vision=aux.get("vision_feature"),
            multimodal=aux.get("fusion_feature"),
            base_logits=logits,
            label=labels,
            meta={"backbone": "clmlf"},
        )

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_for_joint_tuning(self):
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
