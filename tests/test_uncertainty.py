import numpy as np

from graphopf.uncertainty import exponential_correlation, sample_gaussian, sample_student_t


def test_gaussian_and_student_t_match_requested_variance():
    g = sample_gaussian(np.random.default_rng(7), 200_000, np.array([0.1, 0.25, 0.5]))
    t = sample_student_t(np.random.default_rng(8), 200_000, np.array([0.1, 0.25, 0.5]), df=5)
    target = np.array([0.1, 0.25, 0.5]) ** 2
    np.testing.assert_allclose(g.var(axis=0), target, rtol=0.035, atol=2e-4)
    np.testing.assert_allclose(t.var(axis=0), target, rtol=0.055, atol=4e-4)


def test_student_t_is_heavier_tailed_than_gaussian():
    g = sample_gaussian(np.random.default_rng(10), 200_000, np.ones(1))[:, 0]
    t = sample_student_t(np.random.default_rng(11), 200_000, np.ones(1), df=5)[:, 0]
    assert np.mean(np.abs(t) > 4.0) > np.mean(np.abs(g) > 4.0)


def test_exponential_correlation_is_recovered_empirically():
    d = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    target = exponential_correlation(d, 1.5)
    x = sample_gaussian(np.random.default_rng(12), 150_000, np.ones(3), target)
    np.testing.assert_allclose(np.corrcoef(x, rowvar=False), target, atol=0.015)


def test_iid_student_t_components_are_independent():
    x = sample_student_t(
        np.random.default_rng(13), 200_000, np.ones(3), df=5, correlation=None
    )
    corr = np.corrcoef(x, rowvar=False)
    off_diag = corr - np.eye(3)
    assert np.max(np.abs(off_diag)) < 0.015

    # Tail events should also be approximately independent, which catches a
    # shared radial scale that ordinary linear correlation can miss.
    tail = np.abs(x) > 2.5
    p0 = tail[:, 0].mean()
    p1 = tail[:, 1].mean()
    joint = np.mean(tail[:, 0] & tail[:, 1])
    assert abs(joint - p0 * p1) < 0.002
