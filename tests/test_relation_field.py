import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from vmhp.core.horizontal_relation_field import HorizontalAffectiveRelationField


def test_relation_field_constraints():
    field = HorizontalAffectiveRelationField(3, delta_max=0.05)
    field.build_from_centers(torch.randn(3, 8))
    pos, neg = field.get_relations()
    assert torch.allclose(torch.diag(pos), torch.zeros(3), atol=1e-7)
    assert torch.allclose(torch.diag(neg), torch.zeros(3), atol=1e-7)
    assert (pos >= 0).all()
    assert (neg >= 0).all()
    for mat in (pos, neg):
        row_sum = mat.sum(dim=-1)
        nonzero = row_sum > 1e-8
        assert torch.allclose(row_sum[nonzero], torch.ones_like(row_sum[nonzero]), atol=1e-6)


def test_multiplicative_refinement_preserves_relation_support():
    field = HorizontalAffectiveRelationField(3, refinement_mode="multiplicative")
    pos_init = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ]
    )
    neg_init = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    field.load_initial_relations(pos_init, neg_init)
    pos, neg = field.get_relations()
    assert torch.allclose(pos, pos_init)
    assert torch.allclose(neg, neg_init)

    with torch.no_grad():
        field.theta_pos.normal_()
        field.theta_neg.normal_()
    refined_pos, refined_neg = field.get_relations()
    assert torch.equal(refined_pos > 0, pos_init > 0)
    assert torch.equal(refined_neg > 0, neg_init > 0)


def test_multiplicative_gain_learns_single_edge_mass_and_empty_rows():
    field = HorizontalAffectiveRelationField(
        3,
        refinement_mode="multiplicative_gain",
        empty_row_mass=0.05,
        empty_row_floor=1e-3,
    )
    pos_init = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    neg_init = torch.zeros(3, 3)
    field.load_initial_relations(pos_init, neg_init)

    pos, neg = field.get_relations()
    assert torch.allclose(pos[0].sum(), torch.tensor(1.0))
    assert torch.allclose(pos[1].sum(), torch.tensor(0.05), atol=1e-6)
    assert torch.allclose(neg.sum(dim=-1), torch.full((3,), 0.05), atol=1e-6)

    with torch.no_grad():
        field.row_log_mass_pos[0].fill_(0.5)
    refined_pos, _ = field.get_relations()
    assert refined_pos[0].sum() > pos[0].sum()
    assert torch.equal(refined_pos[0] > 0, pos_init[0] > 0)
