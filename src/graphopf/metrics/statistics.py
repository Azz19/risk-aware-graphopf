from math import sqrt

from scipy.stats import norm


def wilson_interval(successes: int, trials: int, confidence: float = 0.95):
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    p = successes / trials
    z = norm.ppf(0.5 + confidence / 2.0)
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)
