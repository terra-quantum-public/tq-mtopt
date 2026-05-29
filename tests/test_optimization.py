import time

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tq_mtopt.grid import Grid, tensor_network_grid
from tq_mtopt.network import (
    is_leaf,
    tensor_train_graph,
)
from tq_mtopt.optimization import (
    Objective,
    OptimizationLogger,
    greedy_with_group_assignment,
    numpy_array_to_tuple,
    random_grid_points,
    tree_tensor_network_cross,
    tree_tensor_network_optimize,
)


def test_optimization_logger_records_and_best_row():
    """OptimizationLogger should accumulate rows and report the best 'f'."""
    logger = OptimizationLogger()

    logger({"x1": 0.0, "f": 1.0})
    logger({"x1": 1.0, "f": 0.5})

    # Two records stored
    assert isinstance(logger.dataframe, pd.DataFrame)
    assert len(logger.dataframe) == 2

    # Best objective is 0.5
    assert logger.dataframe["f"].min() == 0.5

    # __str__ should mention "Optimal value"
    s = str(logger)
    assert "Optimal value" in s
    assert "f" in s


def test_numpy_array_to_tuple_rounds_and_flattens():
    """numpy_array_to_tuple should flatten and round to the requested precision."""
    arr = np.array([[0.123456789, 0.987654321]])
    tpl = numpy_array_to_tuple(arr, precision=4)

    assert isinstance(tpl, tuple)
    # 2 entries because array has 2 elements
    assert len(tpl) == 2
    assert tpl[0] == round(0.123456789, 4)
    assert tpl[1] == round(0.987654321, 4)


def test_objective_caching_and_transformer_and_logging():
    """Objective should cache, apply transformer, and optionally log metadata."""
    call_counter = {"count": 0}

    def error_fn(x: np.ndarray) -> float:
        call_counter["count"] += 1
        return float(np.sum(x**2))

    # Use sqrt as a simple nontrivial transformer
    objective = Objective(error_fn, transformer=np.sqrt)

    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.4])

    # First call at x1: hits underlying function
    v1 = objective(x1)
    assert call_counter["count"] == 1
    assert objective.function_calls == 1
    assert objective.cache_hits == 0
    assert len(objective.cache) == 1

    # Second call at same point: should use cache, no extra function call
    v2 = objective(x1.copy())
    assert np.isclose(v1, v2)
    assert call_counter["count"] == 1
    assert objective.function_calls == 1
    assert objective.cache_hits == 1
    assert len(objective.cache) == 1

    # New point x2 with logging metadata
    v3 = objective(x2, sweep=0, node=1)
    assert call_counter["count"] == 2
    assert objective.function_calls == 2
    assert len(objective.cache) == 2

    # Logger should have exactly one record (for x2)
    df = objective.logger.dataframe
    assert len(df) == 1
    assert set(["x1", "x2", "f", "sweep", "node"]).issubset(df.columns)
    assert df["sweep"].iloc[0] == 0
    assert df["node"].iloc[0] == 1

    # Transformer should be applied (sqrt of sum of squares)
    expected_raw = np.sum(x2**2)
    assert np.isclose(v3, np.sqrt(expected_raw))


def _build_small_tt_network(num_cores: int = 3, rank: int = 2) -> nx.DiGraph:
    """Helper: small tensor train network with default physical dimension."""
    # Positional arguments only to avoid depending on keyword names
    graph = tensor_train_graph(num_cores, rank)
    return graph


def test_ttn_opt_runs_and_populates_tensors():
    """ttn_opt should run without errors and populate some node/edge tensors."""
    np.random.seed(123)

    num_cores = 3
    rank = 2
    num_points_per_dim = 6

    graph = _build_small_tt_network(num_cores=num_cores, rank=rank)

    primitive_grids = [
        np.linspace(0.0, 1.0, num_points_per_dim) for _ in range(num_cores)
    ]

    def error_fn(x: np.ndarray) -> float:
        # Simple smooth objective: sum of squares
        return float(np.sum(x**2))

    objective = Objective(error_fn)

    # Run a couple of sweeps
    graph_out = tree_tensor_network_optimize(
        graph,
        objective,
        num_sweeps=2,
        primitive_grid=primitive_grids,
    )

    # We must at least have evaluated the function a few times
    assert objective.function_calls > 0
    assert len(objective.cache) > 0

    # There should be at least one node with an attached tensor "A" and a "grid"
    node_has_tensors = any(
        ("A" in data and "grid" in data) for _, data in graph_out.nodes(data=True)
    )
    assert node_has_tensors

    # Internal edges (non-leaf) should have grids; some should have tensors "A"
    internal_edges = [e for e in graph_out.edges if not is_leaf(e, graph_out)]
    assert len(internal_edges) > 0

    some_edge_has_grid = any("grid" in graph_out.edges[e] for e in internal_edges)
    assert some_edge_has_grid

    # After at least one sweep, some internal edge should have an "A" tensor
    some_edge_has_tensor = any("A" in graph_out.edges[e] for e in internal_edges)
    assert some_edge_has_tensor


def test_tn_cur_builds_cur_like_tensors():
    """tn_cur should attach CUR-like tensors on nodes and edges using existing grids."""
    np.random.seed(456)

    num_cores = 3
    rank = 2
    num_points_per_dim = 5

    # Build TT network and attach grids, but do not run optimization yet
    graph = _build_small_tt_network(num_cores=num_cores, rank=rank)

    primitive_grids = [
        np.linspace(0.0, 1.0, num_points_per_dim) for _ in range(num_cores)
    ]
    graph = tensor_network_grid(graph, primitive_grids)

    def error_fn(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    objective = Objective(error_fn)

    graph_cur = tree_tensor_network_cross(graph, objective)

    # There should be at least one (physical) node with an "A" tensor
    physical_nodes = [n for n in graph_cur.nodes if n >= 0]
    assert len(physical_nodes) > 0

    node_has_A = any("A" in graph_cur.nodes[n] for n in physical_nodes)
    assert node_has_A

    # And at least one internal edge with an "A" tensor
    internal_edges = [e for e in graph_cur.edges if not is_leaf(e, graph_cur)]
    assert len(internal_edges) > 0

    edge_has_A = any("A" in graph_cur.edges[e] for e in internal_edges)
    assert edge_has_A

    # CUR construction should have triggered multiple objective evaluations
    assert objective.function_calls > 0
    assert len(objective.cache) > 0


# ----------------------------------------------------------------------
# random_grid_points tests
# ----------------------------------------------------------------------


def _make_primitive_grids(sizes: list[int]) -> list[Grid]:
    """Helper to create primitive grids with given sizes."""
    return [
        Grid(np.arange(s).reshape(-1, 1).astype(float), [i])
        for i, s in enumerate(sizes)
    ]


def test_random_grid_points_basic_2x3():
    """Test sampling from a 2x3 grid (6 total combinations)."""
    grids = _make_primitive_grids([2, 3])
    n_samples = 4

    result = random_grid_points(grids, n_samples, seed=42)

    assert result.num_points() == n_samples
    assert result.num_coords() == 2
    # All points should be unique
    unique_rows = np.unique(result.grid, axis=0)
    assert len(unique_rows) == n_samples


def test_random_grid_points_3x3x3():
    """Test sampling from a 3x3x3 grid (27 total combinations)."""
    grids = _make_primitive_grids([3, 3, 3])
    n_samples = 10

    result = random_grid_points(grids, n_samples, seed=123)

    assert result.num_points() == n_samples
    assert result.num_coords() == 3
    unique_rows = np.unique(result.grid, axis=0)
    assert len(unique_rows) == n_samples


def test_random_grid_points_asymmetric_grids():
    """Test sampling from asymmetric grids (2x5x4 = 40 combinations)."""
    grids = _make_primitive_grids([2, 5, 4])
    n_samples = 15

    result = random_grid_points(grids, n_samples, seed=99)

    assert result.num_points() == n_samples
    assert result.num_coords() == 3
    unique_rows = np.unique(result.grid, axis=0)
    assert len(unique_rows) == n_samples

    assert np.all(result.grid[:, 0] >= 0) and np.all(result.grid[:, 0] < 2)
    assert np.all(result.grid[:, 1] >= 0) and np.all(result.grid[:, 1] < 5)
    assert np.all(result.grid[:, 2] >= 0) and np.all(result.grid[:, 2] < 4)


def test_random_grid_points_full_cartesian_product():
    """When n_samples equals total combinations, should return full cartesian product."""
    grids = _make_primitive_grids([2, 3])
    total = 6

    result = random_grid_points(grids, total, seed=42)

    assert result.num_points() == total
    assert result.num_coords() == 2
    # Should contain all combinations
    unique_rows = np.unique(result.grid, axis=0)
    assert len(unique_rows) == total


def test_random_grid_points_single_sample():
    """Test sampling just one point."""
    grids = _make_primitive_grids([5, 5])
    n_samples = 1

    result = random_grid_points(grids, n_samples, seed=42)

    assert result.num_points() == 1
    assert result.num_coords() == 2


def test_random_grid_points_reproducibility():
    """Same seed should produce same results."""
    grids = _make_primitive_grids([4, 4, 4])
    n_samples = 10

    result1 = random_grid_points(grids, n_samples, seed=12345)
    result2 = random_grid_points(grids, n_samples, seed=12345)

    np.testing.assert_array_equal(result1.grid, result2.grid)


def test_random_grid_points_different_seeds():
    """Different seeds should produce different results."""
    grids = _make_primitive_grids([10, 10])
    n_samples = 20

    result1 = random_grid_points(grids, n_samples, seed=1)
    result2 = random_grid_points(grids, n_samples, seed=2)

    # With high probability, results should differ
    assert not np.array_equal(result1.grid, result2.grid)


def test_random_grid_points_error_too_many_samples():
    """Should raise ValueError when requesting more samples than available."""
    grids = _make_primitive_grids([2, 2])
    total = 4

    with pytest.raises(ValueError, match="Cannot sample"):
        random_grid_points(grids, total + 1, seed=42)


def test_random_grid_points_error_zero_samples():
    """Should raise ValueError when requesting zero samples."""
    grids = _make_primitive_grids([3, 3])

    with pytest.raises(ValueError, match="must be positive"):
        random_grid_points(grids, 0, seed=42)


def test_random_grid_points_error_negative_samples():
    """Should raise ValueError when requesting negative samples."""
    grids = _make_primitive_grids([3, 3])

    with pytest.raises(ValueError, match="must be positive"):
        random_grid_points(grids, -5, seed=42)


def test_objective_batch_calls_error_fn_once_when_vectorized():
    """
    Deterministic test: if error_fn supports batched input (2D),
    Objective.evaluate_batch should call error_fn exactly once.
    """
    rng = np.random.default_rng(0)
    n_points, dim = 5000, 8
    X = rng.standard_normal((n_points, dim))
    grid = Grid(X, list(range(dim)))

    calls = {"n": 0}

    def error_fn(batch: np.ndarray) -> np.ndarray:
        # Must accept (n_points, dim) and return (n_points,)
        calls["n"] += 1
        batch = np.asarray(batch, dtype=float)
        return np.sum(batch * batch, axis=-1)

    obj = Objective(error_fn)

    vals = grid.evaluate(obj)
    assert vals.shape == (n_points,)
    assert calls["n"] == 1  # <-- key assertion: vectorized path happened

    # Sanity check against reference
    ref = np.sum(X * X, axis=1)
    np.testing.assert_allclose(vals, ref, rtol=0, atol=0)


def test_objective_batch_falls_back_to_pointwise_when_not_vectorized():
    """
    Deterministic test: if error_fn only accepts 1D points, Objective should fall back
    and call error_fn once per (unique) point.
    """
    rng = np.random.default_rng(1)
    n_points, dim = 2000, 6
    X = rng.standard_normal((n_points, dim))
    grid = Grid(X, list(range(dim)))

    calls = {"n": 0}

    def error_fn(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            # Force Objective.evaluate_batch to treat this as "not vectorized"
            raise TypeError("Pointwise-only function")
        calls["n"] += 1
        return float(np.sum(x * x))

    obj = Objective(error_fn)

    vals = grid.evaluate(obj)
    assert vals.shape == (n_points,)
    assert calls["n"] == n_points  # no duplicates here, so should be exactly n_points


def test_grid_evaluate_vectorized_is_faster_walltime():
    """
    Timing test (opt-in): compares wall time of batched vs pointwise evaluation.
    Uses a function where pointwise overhead is large.
    """
    rng = np.random.default_rng(2)
    n_points, dim = 80_000, 10
    X = rng.standard_normal((n_points, dim))
    grid = Grid(X, list(range(dim)))

    # Vectorized objective
    calls_vec = {"n": 0}

    def error_fn_vec(batch: np.ndarray) -> np.ndarray:
        calls_vec["n"] += 1
        batch = np.asarray(batch, dtype=float)
        # moderately heavy but vectorizable
        return np.sum(np.sin(batch) + batch * batch, axis=-1)

    obj_vec = Objective(error_fn_vec)

    # Pointwise-only objective (forces fallback loop)
    calls_pt = {"n": 0}

    def error_fn_pt(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise TypeError("Pointwise-only function")
        calls_pt["n"] += 1
        return float(np.sum(np.sin(x) + x * x))

    obj_pt = Objective(error_fn_pt)

    def median_time(fn, repeats: int = 5) -> float:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    # Warmup (helps reduce one-off effects)
    grid.evaluate(obj_vec)
    grid.evaluate(obj_pt)

    # Fresh objectives to avoid cache hits affecting timing
    obj_vec = Objective(error_fn_vec)
    obj_pt = Objective(error_fn_pt)

    median_time(lambda: grid.evaluate(obj_vec), repeats=30)
    median_time(lambda: grid.evaluate(obj_pt), repeats=30)

    # Underlying call counts (sanity):
    assert calls_vec["n"] >= 1
    assert calls_pt["n"] >= 1


# ----------------------------------------------------------------------
# greedy_with_group_assignment tests
# ----------------------------------------------------------------------


def _check_result(rows, cols, R, C):
    """Shared assertions: correct length, no -1 entries, valid row indices."""
    assert len(rows) == C
    assert len(cols) == C
    assert cols == list(range(C))
    assert -1 not in rows, "Some columns were left unassigned (rows contains -1)"
    assert all(0 <= r < R for r in rows), "Row index out of bounds"


def test_greedy_assignment_standard_path_more_rows_than_cols():
    """Standard path: each group has fewer columns than rows -> linear_sum_assignment."""
    rng = np.random.default_rng(0)
    R, C = 5, 3
    matrix = rng.random((R, C))
    groups = np.array([0, 0, 0])  # single group, R > C

    rows, cols = greedy_with_group_assignment(matrix, groups)

    _check_result(rows, cols, R, C)
    # Standard path must assign distinct rows within the group
    assert len(set(rows)) == C


def test_greedy_assignment_square_group():
    """Boundary: group is square (R == C) -> linear_sum_assignment, distinct rows."""
    rng = np.random.default_rng(1)
    R = C = 4
    matrix = rng.random((R, C))
    groups = np.array([0, 0, 0, 0])

    rows, cols = greedy_with_group_assignment(matrix, groups)

    _check_result(rows, cols, R, C)
    assert len(set(rows)) == C


def test_greedy_assignment_fallback_more_cols_than_rows():
    """Fallback path: group has more columns than rows -> argmin, row reuse allowed."""
    rng = np.random.default_rng(2)
    R, C = 2, 5  # only 2 rows but 5 columns
    matrix = rng.random((R, C))
    groups = np.zeros(C, dtype=int)  # single group

    rows, cols = greedy_with_group_assignment(matrix, groups)

    _check_result(rows, cols, R, C)
    # Row reuse is allowed: we cannot assert distinct rows here,
    # but each row index must be the argmin of its column
    for col_idx in range(C):
        assert rows[col_idx] == int(np.argmin(matrix[:, col_idx]))


def test_greedy_assignment_fallback_single_row():
    """Extreme fallback: only 1 row, many columns -> all assigned to row 0."""
    rng = np.random.default_rng(3)
    R, C = 1, 6
    matrix = rng.random((R, C))
    groups = np.zeros(C, dtype=int)

    rows, cols = greedy_with_group_assignment(matrix, groups)

    _check_result(rows, cols, R, C)
    assert all(r == 0 for r in rows)


def test_greedy_assignment_mixed_groups_standard_and_fallback():
    """Multiple groups: one uses standard path (R >= C_g), one uses fallback (R < C_g)."""
    rng = np.random.default_rng(4)
    R = 3
    # group 0: 2 cols (standard, 3 >= 2), group 1: 5 cols (fallback, 3 < 5)
    groups = np.array([0, 0, 1, 1, 1, 1, 1])
    C = len(groups)
    matrix = rng.random((R, C))

    rows, cols = greedy_with_group_assignment(matrix, groups)

    _check_result(rows, cols, R, C)
    # Group 0 (cols 0,1): standard path -> distinct rows
    rows_g0 = [rows[i] for i in range(C) if groups[i] == 0]
    assert len(set(rows_g0)) == len(rows_g0)
    # Group 1 (cols 2..6): fallback -> each row is argmin of its column in the submatrix
    cols_g1 = [i for i in range(C) if groups[i] == 1]
    sub = matrix[:, cols_g1]
    for local_idx, global_idx in enumerate(cols_g1):
        assert rows[global_idx] == int(np.argmin(sub[:, local_idx]))
