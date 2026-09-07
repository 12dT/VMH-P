import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from adapters.d2r_adapter import D2RAdapter
from core.packet import EvidencePacket
from core.plugin_head import TreeRingConfig, TreeRingHead


class DummyD2R(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, images, return_features=False):
        base = input_ids.float()
        text = base[:, :4]
        vision = images.float()[:, :5]
        fusion = torch.cat([text[:, :3], vision[:, :3]], dim=-1)
        logits = self.proj(text)
        loss = torch.nn.functional.cross_entropy(logits, labels.long())
        if return_features:
            return {
                "loss": loss,
                "logits": logits,
                "text_global": text,
                "vision_global": vision,
                "fusion_feature": fusion,
                "features": {
                    "text_global": text,
                    "vision_global": vision,
                    "fusion_feature": fusion,
                    "routed_text": text + 1.0,
                    "routed_vision": vision + 1.0,
                },
            }
        return loss, logits


def make_batch(batch_size=2):
    return (
        torch.arange(batch_size * 6).view(batch_size, 6),
        torch.ones(batch_size, 6),
        torch.zeros(batch_size, 6),
        torch.ones(batch_size),
        torch.tensor([0, 1]),
        torch.arange(batch_size * 7).view(batch_size, 7),
    )


def test_d2r_adapter_packet_semantics_and_detach():
    backbone = DummyD2R()
    batch = make_batch()
    _, original_logits = backbone(
        input_ids=batch[0],
        attention_mask=batch[1],
        token_type_ids=batch[2],
        labels=batch[4],
        images=batch[5],
        return_features=False,
    )

    adapter = D2RAdapter(backbone, detach_backbone=True)
    packet = adapter.extract(batch)
    assert isinstance(packet, EvidencePacket)
    assert torch.allclose(packet.base_logits, original_logits)
    assert packet.text.shape == (2, 4)
    assert packet.vision.shape == (2, 5)
    assert packet.multimodal.shape == (2, 6)
    for value in (packet.text, packet.vision, packet.multimodal, packet.base_logits):
        assert torch.isfinite(value).all()
    assert packet.meta["backbone"] == "d2r"
    assert set(packet.sources()) == {"text", "vision", "multimodal"}

    adapter.freeze_backbone()
    packet = adapter.extract(batch)
    loss = packet.base_logits.sum()
    assert not loss.requires_grad
    assert all(param.grad is None for param in backbone.parameters())


def test_d2r_detached_treering_returns_base_logits_and_core_has_no_d2r_branch():
    packet = D2RAdapter(DummyD2R(), detach_backbone=True).extract(make_batch())
    cfg = TreeRingConfig(tree_dim=8, num_internal=2)
    head = TreeRingHead({"text": 4, "vision": 5, "multimodal": 6}, 3, cfg)
    head.initialize_from_class_centers(torch.randn(3, 8) * 0.1)
    disabled, _ = head(packet, use_plugin=False)
    assert torch.allclose(disabled, packet.base_logits, atol=1e-7)

    core_dir = os.path.join(ROOT, "core")
    for dirpath, _dirnames, filenames in os.walk(core_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                with open(os.path.join(dirpath, filename), encoding="utf-8") as fh:
                    text = fh.read().lower()
                assert "backbone == \"d2r\"" not in text
                assert "backbone == 'd2r'" not in text
