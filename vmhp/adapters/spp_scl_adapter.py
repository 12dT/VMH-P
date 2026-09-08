import contextlib
from typing import Optional

import torch

from vmhp.core.packet import EvidencePacket
from .base_adapter import BaseBackboneAdapter


class SPPSCLAdapter(BaseBackboneAdapter):
    def __init__(self, backbone, device: Optional[torch.device] = None, detach_backbone: bool = True):
        self.backbone = backbone
        self.device = device
        self.detach_backbone = bool(detach_backbone)

    def _move_batch(self, batch):
        if self.device is None:
            return batch
        if isinstance(batch, (list, tuple)):
            return type(batch)(x.to(self.device) if torch.is_tensor(x) else x for x in batch)
        return batch

    def extract(self, batch) -> EvidencePacket:
        batch = self._move_batch(batch)
        labels = batch[-1] if isinstance(batch, (list, tuple)) and torch.is_tensor(batch[-1]) else None
        grad_enabled = (not self.detach_backbone) and any(p.requires_grad for p in self.backbone.parameters())
        guard = contextlib.nullcontext() if grad_enabled else torch.no_grad()
        with guard:
            output, cl_out, t_align, i_align = self.backbone(*batch[:-1]) if labels is not None else self.backbone(*batch)
        if self.detach_backbone:
            output = output.detach()
            cl_out = cl_out.detach()
            t_align = t_align.detach()
            i_align = i_align.detach()
        return EvidencePacket(
            text=t_align,
            vision=i_align,
            multimodal=cl_out,
            base_logits=output,
            label=labels,
            meta={"backbone": "spp_scl"},
        )

    def freeze_backbone(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_for_joint_tuning(self):
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
