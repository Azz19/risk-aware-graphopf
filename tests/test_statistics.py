from graphopf.metrics import wilson_interval


def test_wilson_interval_contains_empirical_rate():
    lo, hi = wilson_interval(50, 1000)
    assert lo < 0.05 < hi


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 100)
    assert 0.0 <= lo <= hi <= 1.0
