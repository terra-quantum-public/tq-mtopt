"""
Tests for tq_mtopt.representation.

We use simple functions whose CP rank is known:
  - rank-1 separable: f(x) = prod_k x_k
  - rank-1 separable shifted: f(x) = prod_k (x_k + c)
  - rank-2 (sum of two rank-1 terms): f(x) = prod_k x_k + prod_k (x_k - 1)

For these functions a TRCRepresentation with r >= true_rank built from a
well-chosen skeleton should interpolate exactly at skeleton points and
approximate well at all other grid points.
"""

import numpy as np

from tq_mtopt.grid import Grid
from tq_mtopt.optimization import (
    TensorRankOptimization,
    MatrixTrainOptimization,
    random_grid_points,
)
from tq_mtopt.representation import (
    build_trc_representation,
    TTRepresentation,
    build_tt_representation,
    MTRepresentation,
    build_mt_representation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_primitives(
    d: int, n: int = 20, lo: float = 1.0, hi: float = 3.0
) -> list[Grid]:
    """d uniform 1-D grids over [lo, hi] with n points, avoiding zero."""
    return [Grid(np.linspace(lo, hi, n), k) for k in range(d)]


def rank1_fn(x: np.ndarray, **_) -> float:
    return float(np.prod(x))


def rank2_fn(x: np.ndarray, **_) -> float:
    return float(np.prod(x) + np.prod(x - 1.0))


# ---------------------------------------------------------------------------
# TRCRepresentation unit tests (using a hand-crafted skeleton)
# ---------------------------------------------------------------------------


class TestTRCRepresentationUnit:
    def test_evaluate_rank1_exact(self):
        """Rank-1 function with r=1 skeleton: representation is exact on grid points."""
        d, n = 3, 15
        primitives = make_primitives(d, n)

        # Skeleton = one point that IS on the grid.
        sk_val = primitives[0].grid[n // 2, 0]  # middle grid value
        skeleton_vals = np.array([[sk_val, sk_val, sk_val]])
        skeleton = Grid(skeleton_vals, list(range(d)))

        rep = build_trc_representation(skeleton, primitives, rank1_fn)

        # Test only at actual grid points (evaluation is exact on-grid for rank-1).
        for i in [0, 3, 7, 14]:
            pt = np.array([primitives[k].grid[i, 0] for k in range(d)])
            true_val = rank1_fn(pt)
            approx_val = rep.evaluate(pt)
            assert abs(approx_val - true_val) / (abs(true_val) + 1e-12) < 1e-6, (
                f"rank1: at {pt}, true={true_val:.6f}, approx={approx_val:.6f}"
            )

    def test_evaluate_batch_matches_pointwise(self):
        """evaluate_batch should match repeated evaluate calls."""
        d, n = 4, 10
        primitives = make_primitives(d, n)
        # Use actual grid values for the skeleton.
        sk_vals = np.array([[primitives[k].grid[3, 0] for k in range(d)]])
        skeleton = Grid(sk_vals, list(range(d)))

        rep = build_trc_representation(skeleton, primitives, rank1_fn)

        # Test on actual grid points only.
        rng = np.random.default_rng(0)
        idxs = rng.integers(0, n, size=(50, d))
        X = np.array(
            [[primitives[k].grid[idxs[i, k], 0] for k in range(d)] for i in range(50)]
        )
        batch = rep.evaluate_batch(X)
        pointwise = np.array([rep.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, pointwise, rtol=1e-10)

    def test_interpolation_at_skeleton_points(self):
        """Approximation must equal f exactly at every skeleton point."""
        d, n = 3, 20
        primitives = make_primitives(d, n)

        # Use r=2 skeleton.
        skeleton_vals = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
        skeleton = Grid(skeleton_vals, list(range(d)))

        rep = build_trc_representation(skeleton, primitives, rank1_fn)

        for pt in skeleton_vals:
            true_val = rank1_fn(pt)
            approx_val = rep.evaluate(pt)
            assert abs(approx_val - true_val) / (abs(true_val) + 1e-12) < 1e-8, (
                f"interpolation at skeleton: true={true_val}, approx={approx_val}"
            )

    def test_evaluate_batch_accepts_grid(self):
        """evaluate_batch should accept a Grid object directly."""
        d, n = 2, 10
        primitives = make_primitives(d, n)
        skeleton = Grid(np.array([[2.0, 2.0]]), list(range(d)))
        rep = build_trc_representation(skeleton, primitives, rank1_fn)

        test_grid = Grid(np.array([[1.5, 2.5], [2.0, 3.0]]), [0, 1])
        result = rep.evaluate_batch(test_grid)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# Integration tests: build representation from TRC/MTC optimizer output
# ---------------------------------------------------------------------------


class TestBuildFromOptimizer:
    def _run_and_build(self, optimizer, primitives, fn, r, n_epochs=6):
        init_skeleton = random_grid_points(primitives, n_samples=r, seed=0)
        final_skeleton = optimizer.optimize(init_skeleton, fn, n_epochs)
        rep = build_trc_representation(final_skeleton, primitives, fn)
        return rep, final_skeleton

    def test_trc_rank1_low_error(self):
        """TRC with r=1 on a rank-1 function should give near-zero approx error on grid points."""
        d, n = 4, 20
        primitives = make_primitives(d, n)
        optimizer = TensorRankOptimization(primitives)

        rep, _ = self._run_and_build(optimizer, primitives, rank1_fn, r=1)

        # Evaluate on a random sample of actual grid points.
        rng = np.random.default_rng(42)
        idxs = rng.integers(0, n, size=(200, d))
        X_test = np.array(
            [[primitives[k].grid[idxs[i, k], 0] for k in range(d)] for i in range(200)]
        )
        true_vals = np.array([rank1_fn(x) for x in X_test])
        approx_vals = rep.evaluate_batch(X_test)
        rel_errors = np.abs(approx_vals - true_vals) / (np.abs(true_vals) + 1e-12)
        assert rel_errors.mean() < 0.01, f"mean rel error = {rel_errors.mean():.4f}"

    def test_trc_rank2_improves_with_r(self):
        """For a rank-2 function, representation error should decrease as r grows."""
        d, n = 3, 20
        primitives = make_primitives(d, n)

        rng = np.random.default_rng(99)
        idxs = rng.integers(0, n, size=(200, d))
        X_test = np.array(
            [[primitives[k].grid[idxs[i, k], 0] for k in range(d)] for i in range(200)]
        )
        true_vals = np.array([rank2_fn(x) for x in X_test])

        errors = {}
        for r in [1, 2, 4]:
            optimizer = TensorRankOptimization(primitives)
            rep, _ = self._run_and_build(
                optimizer, primitives, rank2_fn, r=r, n_epochs=8
            )
            approx_vals = rep.evaluate_batch(X_test)
            errors[r] = np.abs(approx_vals - true_vals).mean()

        assert errors[2] <= errors[1] * 1.05, (
            f"r=2 error ({errors[2]:.4f}) should not be worse than r=1 ({errors[1]:.4f})"
        )
        assert errors[4] <= errors[2] * 1.05

    def test_mtc_representation_same_api(self):
        """MTC skeleton can be passed to build_trc_representation; API is identical."""
        d, n, r = 5, 15, 3
        primitives = make_primitives(d, n)
        optimizer = MatrixTrainOptimization(primitives)

        rep, _ = self._run_and_build(optimizer, primitives, rank1_fn, r=r, n_epochs=5)

        # Verify shape consistency.
        assert len(rep.factor_matrices) == d
        assert rep.weights.shape == (r,)
        for k, F_k in enumerate(rep.factor_matrices):
            assert F_k.shape == (n, r), f"F_{k} shape mismatch: {F_k.shape}"

    def test_representation_no_extra_fn_calls_after_build(self):
        """After build, evaluate must not call the original function."""
        d, n, r = 3, 10, 2
        primitives = make_primitives(d, n)

        call_count = [0]

        def counted_fn(x, **_):
            call_count[0] += 1
            return rank1_fn(x)

        optimizer = TensorRankOptimization(primitives)
        init = random_grid_points(primitives, n_samples=r, seed=1)
        final = optimizer.optimize(init, counted_fn, num_epochs=3)
        rep = build_trc_representation(final, primitives, counted_fn)

        count_after_build = call_count[0]

        # Evaluate at 100 points — call count must not change.
        rng = np.random.default_rng(7)
        X = rng.uniform(1.0, 3.0, size=(100, d))
        _ = rep.evaluate_batch(X)

        assert call_count[0] == count_after_build, (
            "evaluate_batch must not call the objective function"
        )


# ---------------------------------------------------------------------------
# MTRepresentation tests
# ---------------------------------------------------------------------------


def _build_mt_rep(primitives, fn, r, n_epochs=6, seed=0):
    """Run MTC optimizer then build the MT representation from its skeleton."""
    init_skeleton = random_grid_points(primitives, n_samples=r, seed=seed)
    optimizer = MatrixTrainOptimization(primitives)
    final_skeleton = optimizer.optimize(init_skeleton, fn, n_epochs)
    return build_mt_representation(final_skeleton, primitives, fn, num_sweeps=3)


class TestMTRepresentation:
    def _make_primitives(self, d: int, n: int = 15) -> list[Grid]:
        return [Grid(np.linspace(1.0, 3.0, n), k) for k in range(d)]

    def test_is_mt_representation(self):
        """build_mt_representation returns an MTRepresentation (not TTRepresentation)."""
        d, n = 3, 10
        primitives = self._make_primitives(d, n)
        rep = _build_mt_rep(primitives, rank1_fn, r=2)
        assert isinstance(rep, MTRepresentation)
        assert not isinstance(rep, TTRepresentation)

    def test_skeleton_grids_shape(self):
        """skeleton_grids has d-1 entries, each of shape (<=r, d)."""
        d, n, r = 4, 10, 3
        primitives = self._make_primitives(d, n)
        rep = _build_mt_rep(primitives, rank1_fn, r=r)

        assert len(rep.skeleton_grids) == d - 1
        for k, S_k in enumerate(rep.skeleton_grids):
            assert S_k.ndim == 2
            assert S_k.shape[1] == d, (
                f"bond {k}: expected {d} coords, got {S_k.shape[1]}"
            )
            assert S_k.shape[0] <= r, (
                f"bond {k}: too many pivots ({S_k.shape[0]} > {r})"
            )

    def test_skeleton_points_within_grid_bounds(self):
        """All pivot coordinates in skeleton_grids lie within the primitive grid ranges."""
        d, n, r = 3, 12, 4
        primitives = self._make_primitives(d, n)
        rep = _build_mt_rep(primitives, rank1_fn, r=r, n_epochs=4, seed=1)

        for k, S_k in enumerate(rep.skeleton_grids):
            for coord in range(d):
                lo = primitives[coord].grid[0, 0]
                hi = primitives[coord].grid[-1, 0]
                assert np.all(S_k[:, coord] >= lo - 1e-12), (
                    f"bond {k}, coord {coord}: pivot below lower bound"
                )
                assert np.all(S_k[:, coord] <= hi + 1e-12), (
                    f"bond {k}, coord {coord}: pivot above upper bound"
                )

    def test_mt_and_tt_both_accurate_on_rank1(self):
        """MT (from optimizer skeleton) and TT (independent cross) both achieve low
        error on a rank-1 function.  They use different algorithms so will not agree
        pointwise, but both must be accurate."""
        d, n, r = 3, 10, 3
        primitives = self._make_primitives(d, n)
        rep_mt = _build_mt_rep(primitives, rank1_fn, r=r, n_epochs=4, seed=42)
        rep_tt = build_tt_representation(
            rank1_fn, primitives, rank=r, num_sweeps=4, seed=42
        )

        rng = np.random.default_rng(99)
        idxs = rng.integers(0, n, size=(50, d))
        X = np.array(
            [[primitives[k].grid[idxs[i, k], 0] for k in range(d)] for i in range(50)]
        )
        true_vals = np.array([rank1_fn(x) for x in X])

        for name, rep in [("MT", rep_mt), ("TT", rep_tt)]:
            approx_vals = rep.evaluate_batch(X)
            rel_errors = np.abs(approx_vals - true_vals) / (np.abs(true_vals) + 1e-12)
            assert rel_errors.mean() < 0.05, (
                f"{name} mean rel error too high: {rel_errors.mean():.4f}"
            )

    def test_rank1_low_error(self):
        """MT representation at rank=2 approximates a rank-1 function with low error."""
        d, n = 4, 15
        primitives = self._make_primitives(d, n)
        rep = _build_mt_rep(primitives, rank1_fn, r=2, n_epochs=5, seed=7)

        rng = np.random.default_rng(0)
        idxs = rng.integers(0, n, size=(200, d))
        X = np.array(
            [[primitives[k].grid[idxs[i, k], 0] for k in range(d)] for i in range(200)]
        )
        true_vals = np.array([rank1_fn(x) for x in X])
        approx_vals = rep.evaluate_batch(X)

        rel_errors = np.abs(approx_vals - true_vals) / (np.abs(true_vals) + 1e-12)
        assert rel_errors.mean() < 0.05, f"mean rel error = {rel_errors.mean():.4f}"

    def test_no_fn_calls_after_build(self):
        """evaluate_batch must not call the original function after build."""
        d, n, r = 3, 8, 2
        primitives = self._make_primitives(d, n)
        call_count = [0]

        def counted_fn(x, **_):
            call_count[0] += 1
            return rank1_fn(x)

        rep = _build_mt_rep(primitives, counted_fn, r=r, n_epochs=3, seed=5)
        count_after_build = call_count[0]

        rng = np.random.default_rng(3)
        idxs = rng.integers(0, n, size=(100, d))
        X = np.array(
            [[primitives[k].grid[idxs[i, k], 0] for k in range(d)] for i in range(100)]
        )
        _ = rep.evaluate_batch(X)

        assert call_count[0] == count_after_build, (
            "evaluate_batch must not call the objective function"
        )
