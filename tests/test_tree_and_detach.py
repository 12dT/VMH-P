import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from vmhp.core.packet import EvidencePacket
from vmhp.core.vmhp_head import VMHPConfig, VMHPHead


def test_tree_output_and_detach_property():
    cfg = VMHPConfig(tree_dim=8, num_internal=2)
    head = VMHPHead({"text": 10, "vision": 12, "multimodal": 14}, 3, cfg)
    centers = torch.randn(3, 8) * 0.1
    head.initialize_from_class_centers(centers)
    packet = EvidencePacket(
        text=torch.randn(4, 10),
        vision=torch.randn(4, 12),
        multimodal=torch.randn(4, 14),
        base_logits=torch.randn(4, 3),
        label=torch.tensor([0, 1, 2, 1]),
    )
    final, aux = head(packet)
    assert final.shape == (4, 3)
    assert aux["tree_logits"].shape == (4, 3)
    disabled, _ = head(packet, use_vmhp=False)
    assert torch.allclose(disabled, packet.base_logits, atol=1e-7)


def test_routing_probabilities_sum_to_one():
    cfg = VMHPConfig(tree_dim=8, num_internal=2)
    head = VMHPHead({"text": 10}, 3, cfg)
    head.initialize_from_class_centers(torch.randn(3, 8) * 0.1)
    z = head.ufa["text"](torch.randn(5, 10))
    x = head.geometry.exp_origin(z)
    routing = head.vhsh.compute_routing(x)["edge_log_prob"].exp()
    for parent_id in torch.unique(head.vhsh.edge_parent).tolist():
        edge_ids = torch.nonzero(head.vhsh.edge_parent == int(parent_id), as_tuple=False).view(-1)
        assert torch.allclose(routing[:, edge_ids].sum(dim=-1), torch.ones(5), atol=1e-6)
