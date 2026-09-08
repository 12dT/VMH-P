import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from vmhp.core.geometry import expmap0, lorentz_inner, pairwise_lorentz_distance


def test_expmap_stays_on_lorentz_manifold():
    c = 0.25
    z = torch.randn(8, 16) * 0.1
    x = expmap0(z, c=c)
    inner = lorentz_inner(x, x).squeeze(-1)
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / c), atol=1e-4)


def test_pairwise_distance_is_symmetric_and_finite():
    x = expmap0(torch.randn(5, 8) * 0.2)
    d = pairwise_lorentz_distance(x, x)
    assert torch.isfinite(d).all()
    assert torch.allclose(d, d.t(), atol=1e-5)
    assert d.diag().max() < 1e-2
