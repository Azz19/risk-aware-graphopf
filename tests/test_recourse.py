import numpy as np
from pypower.idx_gen import GEN_STATUS, PG, PMAX, PMIN

from graphopf.experiments import (
    apply_symmetric_generator_backoff,
    directional_participation,
)


def _case():
    gen = np.zeros((3, 21))
    gen[:, GEN_STATUS] = 1
    gen[:, PMIN] = [0.0, 20.0, 40.0]
    gen[:, PMAX] = [100.0, 120.0, 140.0]
    gen[:, PG] = [20.0, 70.0, 130.0]
    return {"gen": gen}


def test_symmetric_backoff_tightens_limits():
    out = apply_symmetric_generator_backoff(_case(), 0.10)
    np.testing.assert_allclose(out["gen"][:, PMIN], [10.0, 30.0, 50.0])
    np.testing.assert_allclose(out["gen"][:, PMAX], [90.0, 110.0, 130.0])


def test_directional_participation_uses_downward_reserve_for_positive_error():
    alpha = directional_participation(_case(), mismatch=10.0)
    expected = np.array([20.0, 50.0, 90.0])
    np.testing.assert_allclose(alpha, expected / expected.sum())


def test_directional_participation_uses_upward_reserve_for_negative_error():
    alpha = directional_participation(_case(), mismatch=-10.0)
    expected = np.array([80.0, 50.0, 10.0])
    np.testing.assert_allclose(alpha, expected / expected.sum())


def test_directional_participation_sums_to_one():
    assert np.isclose(directional_participation(_case(), 5.0).sum(), 1.0)
    assert np.isclose(directional_participation(_case(), -5.0).sum(), 1.0)
