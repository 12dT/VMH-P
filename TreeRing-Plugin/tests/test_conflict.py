import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from core.conflict_gate import MonotonicConflictGate


def test_conflict_is_zero_for_identical_distributions():
    gate = MonotonicConflictGate()
    p = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    out = gate({"text": p, "vision": p})
    assert out["global_conflict"].abs().max() < 1e-6
    assert out["class_disagree"].abs().max() < 1e-6


def test_conflict_gate_is_monotonic():
    gate = MonotonicConflictGate(kappa_init=8.0, mu_init=0.35)
    low = torch.tensor([[0.6, 0.3, 0.1]])
    mid = torch.tensor([[0.2, 0.7, 0.1]])
    high = torch.tensor([[0.05, 0.05, 0.9]])
    out_low = gate({"text": low, "vision": low})
    out_mid = gate({"text": low, "vision": mid})
    out_high = gate({"text": low, "vision": high})
    assert out_low["global_conflict"].mean() <= out_mid["global_conflict"].mean() + 1e-6
    assert out_mid["global_conflict"].mean() <= out_high["global_conflict"].mean() + 1e-6
    assert out_low["gate"].mean() <= out_high["gate"].mean() + 1e-6
