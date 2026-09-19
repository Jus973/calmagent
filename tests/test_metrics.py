import math

import numpy as np

from calm_coder.bench.baselines import pass_at_k
from calm_coder.bench.metrics import mcnemar_exact, paired_bootstrap, wilson


def test_wilson_known_value():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)
    assert wilson(0, 10)[1] == 0.0


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(0, 6), 2 / 64)
    assert mcnemar_exact(3, 3) == 1.0


def test_pass_at_k():
    assert pass_at_k(8, 0, 1) == 0.0 and pass_at_k(8, 8, 1) == 1.0
    assert math.isclose(pass_at_k(8, 2, 1), 0.25)
    assert math.isclose(pass_at_k(8, 1, 8), 1.0)


def test_paired_bootstrap_centered():
    a, b = np.array([1, 1, 0, 1.0]), np.array([0, 1, 0, 0.0])
    m, lo, hi = paired_bootstrap(a, b, reps=2000)
    assert m == 0.5 and lo <= m <= hi
