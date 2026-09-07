from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class EvidencePacket:
    text: Optional[torch.Tensor]
    vision: Optional[torch.Tensor]
    multimodal: Optional[torch.Tensor]
    base_logits: torch.Tensor
    label: Optional[torch.Tensor] = None
    meta: Optional[Dict] = None
    extra_sources: Optional[Dict[str, torch.Tensor]] = None

    def sources(self) -> Dict[str, torch.Tensor]:
        out = {}
        if self.text is not None:
            out["text"] = self.text
        if self.vision is not None:
            out["vision"] = self.vision
        if self.multimodal is not None:
            out["multimodal"] = self.multimodal
        if self.extra_sources:
            for name, value in self.extra_sources.items():
                if value is not None:
                    out[name] = value
        return out
