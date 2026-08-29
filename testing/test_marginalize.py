import numpy as np
import cvxpy as cp

from pdg.alg.interior_pt import _marginalize, _marginalize_slow


def _rand_shape_rng(rng, n_axes=3, min_card=2, max_card=4):
    return tuple(int(rng.integers(min_card, max_card + 1)) for _ in range(n_axes))


def test_marginalize_matches_slow_small_cases():
    rng = np.random.default_rng(0)

    for _ in range(5):
        shape = _rand_shape_rng(rng, n_axes=3, min_card=2, max_card=3)
        n = int(np.prod(shape))
        mu_np = rng.random(n)
        mu_np /= mu_np.sum()

        idx_choices = [
            [0],
            [1],
            [2],
            [0, 2],
            [],
        ]

        for IDXs in idx_choices:
            mu = cp.Parameter(n)
            mu.value = mu_np

            fast = _marginalize(mu, shape, IDXs)
            slow = _marginalize_slow(mu, shape, IDXs)

            x = cp.Variable(fast.size)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(x - fast)))
            prob.solve()

            y = cp.Variable(slow.size)
            prob2 = cp.Problem(cp.Minimize(cp.sum_squares(y - slow)))
            prob2.solve()

            fval = x.value
            sval = y.value

            assert fval.shape == sval.shape
            assert np.allclose(fval, sval, atol=1e-10)

