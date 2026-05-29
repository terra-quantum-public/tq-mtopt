"""Unit tests for tq_mtopt/qtt.py.

Coverage:
  - qtt_digits_to_index: digit<->index round-trips, MSB/LSB, base>2, batch
  - qtt_index_to_unit_interval: endpoint modes, edge cases
  - qtt_coordinate_map / qtt_z_permuted_coordinate_map: ordering and properties
  - QTTDecoder.decode_indices / decode: bounds mapping, permutation handling
  - make_qtt_objective: single-point, vectorised batch, non-vectorised fallback
"""

from __future__ import annotations

import numpy as np
import pytest

from tq_mtopt.qtt import (
    QTTDecoder,
    make_qtt_objective,
    qtt_coordinate_map,
    qtt_digits_to_index,
    qtt_index_to_unit_interval,
    qtt_primitive_grids,
    qtt_z_permuted_coordinate_map,
)


# -----------------------------------------------------------------------
# qtt_digits_to_index
# -----------------------------------------------------------------------


def test_digits_to_index_binary_msb_known_values():
    digits = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    idx = qtt_digits_to_index(digits, base=2, msb_first=True)
    np.testing.assert_array_equal(idx, [0, 5, 6, 7])


def test_digits_to_index_binary_lsb_known_values():
    # LSB first: [1,0,0] -> 1*1 + 0*2 + 0*4 = 1
    digits = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    idx = qtt_digits_to_index(digits, base=2, msb_first=False)
    np.testing.assert_array_equal(idx, [1, 2, 3])


def test_digits_to_index_base3():
    # [2, 1] base=3 MSB: 2*3 + 1 = 7
    digits = np.array([[2, 1], [1, 0], [0, 2]])
    idx = qtt_digits_to_index(digits, base=3, msb_first=True)
    np.testing.assert_array_equal(idx, [7, 3, 2])


def test_digits_to_index_single_digit():
    digits = np.array([[0], [1], [3]])
    idx = qtt_digits_to_index(digits, base=4, msb_first=True)
    np.testing.assert_array_equal(idx, [0, 1, 3])


def test_digits_to_index_1d_input():
    # 1D input: shape (L,) treated as a single digit vector
    digits = np.array([1, 0, 1])
    idx = qtt_digits_to_index(digits, base=2, msb_first=True)
    assert int(idx) == 5


def test_digits_to_index_full_roundtrip_binary():
    """Encode every integer 0..2^L-1 to digits and back."""
    L, base = 5, 2
    N = base**L
    for i in range(N):
        d = np.array([(i >> (L - 1 - b)) & 1 for b in range(L)], dtype=int)
        recovered = int(qtt_digits_to_index(d, base=base, msb_first=True))
        assert recovered == i, f"roundtrip failed for i={i}: got {recovered}"


def test_digits_to_index_full_roundtrip_base3():
    """Encode every integer 0..3^3-1 to base-3 digits and back."""
    L, base = 3, 3
    N = base**L
    for i in range(N):
        tmp = i
        d = []
        for _ in range(L):
            d.append(tmp % base)
            tmp //= base
        d = np.array(d[::-1], dtype=int)  # MSB first
        recovered = int(qtt_digits_to_index(d, base=base, msb_first=True))
        assert recovered == i


def test_digits_to_index_invalid_digit_raises():
    with pytest.raises(ValueError, match="digits must be in"):
        qtt_digits_to_index(np.array([[0, 2]]), base=2)


def test_digits_to_index_empty_last_axis_raises():
    with pytest.raises(ValueError, match="length L>0"):
        qtt_digits_to_index(np.zeros((3, 0), dtype=int), base=2)


# -----------------------------------------------------------------------
# qtt_index_to_unit_interval
# -----------------------------------------------------------------------


def test_index_to_unit_interval_endpoint_true():
    # 0 -> 0.0, N-1 -> 1.0
    N = 5
    u = qtt_index_to_unit_interval(np.array([0, 2, 4]), N, endpoint=True)
    np.testing.assert_allclose(u, [0.0, 0.5, 1.0])


def test_index_to_unit_interval_endpoint_false():
    # i -> i/N
    N = 4
    u = qtt_index_to_unit_interval(np.array([0, 1, 2, 3]), N, endpoint=False)
    np.testing.assert_allclose(u, [0.0, 0.25, 0.5, 0.75])


def test_index_to_unit_interval_single_point_endpoint_true():
    # num_points=1, endpoint=True -> always 0
    u = qtt_index_to_unit_interval(np.array([0]), num_points=1, endpoint=True)
    np.testing.assert_allclose(u, [0.0])


def test_index_to_unit_interval_zero_num_points_raises():
    with pytest.raises(ValueError, match="num_points must be positive"):
        qtt_index_to_unit_interval(np.array([0]), num_points=0)


# -----------------------------------------------------------------------
# qtt_coordinate_map
# -----------------------------------------------------------------------


def test_coordinate_map_var_major_order():
    # 3 vars, levels=[1,2,1]: expected [(0,0),(1,0),(1,1),(2,0)]
    m = qtt_coordinate_map(3, [1, 2, 1])
    assert m == [(0, 0), (1, 0), (1, 1), (2, 0)]


def test_coordinate_map_uniform_levels():
    m = qtt_coordinate_map(2, 3)
    assert m == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_coordinate_map_is_permutation():
    num_vars, levels = 4, 3
    m = qtt_coordinate_map(num_vars, levels)
    expected = {(k, ell) for k in range(num_vars) for ell in range(levels)}
    assert set(m) == expected
    assert len(m) == num_vars * levels


# -----------------------------------------------------------------------
# qtt_z_permuted_coordinate_map
# -----------------------------------------------------------------------


def test_z_permuted_z1_equals_var_major():
    num_vars, levels = 4, 3
    assert qtt_z_permuted_coordinate_map(num_vars, levels, z=1) == qtt_coordinate_map(
        num_vars, levels
    )


def test_z_permuted_full_interleave():
    # z >= num_vars -> all vars interleaved digit by digit
    m = qtt_z_permuted_coordinate_map(4, 3, z=10)
    expected = [(k, ell) for ell in range(3) for k in range(4)]
    assert m == expected


def test_z_permuted_z3_known_example():
    # num_vars=6, levels=2, z=3: documented example in qtt.py
    m = qtt_z_permuted_coordinate_map(6, 2, z=3)
    expected = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 0),
        (4, 0),
        (5, 0),
        (3, 1),
        (4, 1),
        (5, 1),
    ]
    assert m == expected


def test_z_permuted_is_permutation():
    num_vars, levels, z = 6, 4, 3
    m = qtt_z_permuted_coordinate_map(num_vars, levels, z)
    expected = {(k, ell) for k in range(num_vars) for ell in range(levels)}
    assert set(m) == expected
    assert len(m) == num_vars * levels


def test_z_permuted_non_divisible_num_vars():
    # num_vars=5, z=3: groups are [0,1,2] and [3,4]
    m = qtt_z_permuted_coordinate_map(5, 2, z=3)
    expected = [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 0),
        (4, 0),
        (3, 1),
        (4, 1),
    ]
    assert m == expected


def test_z_permuted_invalid_z_raises():
    with pytest.raises(ValueError, match="z must be >= 1"):
        qtt_z_permuted_coordinate_map(3, 2, z=0)


# -----------------------------------------------------------------------
# qtt_primitive_grids
# -----------------------------------------------------------------------


def test_primitive_grids_length_and_range():
    num_vars, levels, base = 3, 4, 2
    grids = qtt_primitive_grids(num_vars, levels, base)
    assert len(grids) == num_vars * levels
    for g in grids:
        vals = g.grid.flatten()
        np.testing.assert_array_equal(vals, np.arange(base))


# -----------------------------------------------------------------------
# QTTDecoder.decode_indices (var-major, no bounds)
# -----------------------------------------------------------------------


def test_decoder_decode_indices_var_major_single_point():
    # num_vars=2, levels=2, base=2
    # var-major order: [x0_d0, x0_d1, x1_d0, x1_d1]
    # digits [1,1,0,1] -> var0: [1,1] -> 3, var1: [0,1] -> 1
    dec = QTTDecoder(num_vars=2, levels=2, base=2)
    q = np.array([1, 1, 0, 1])
    idx = dec.decode_indices(q)
    np.testing.assert_array_equal(idx, [3, 1])


def test_decoder_decode_indices_batch():
    dec = QTTDecoder(num_vars=2, levels=2, base=2)
    # Two points: [1,1,0,1] and [0,0,1,0]
    Q = np.array([[1, 1, 0, 1], [0, 0, 1, 0]])
    idx = dec.decode_indices(Q)
    assert idx.shape == (2, 2)
    np.testing.assert_array_equal(idx[0], [3, 1])
    np.testing.assert_array_equal(idx[1], [0, 2])


def test_decoder_decode_indices_all_zeros():
    dec = QTTDecoder(num_vars=3, levels=3, base=2)
    q = np.zeros((1, 9), dtype=int)
    idx = dec.decode_indices(q)
    np.testing.assert_array_equal(idx, [[0, 0, 0]])


def test_decoder_decode_indices_all_ones_binary():
    # All digits 1: index = base^L - 1 for each var
    L, base = 4, 2
    dec = QTTDecoder(num_vars=2, levels=L, base=base)
    q = np.ones((1, 2 * L), dtype=int)
    idx = dec.decode_indices(q)
    np.testing.assert_array_equal(idx, [[base**L - 1, base**L - 1]])


# -----------------------------------------------------------------------
# QTTDecoder.decode (bounds mapping)
# -----------------------------------------------------------------------


def test_decoder_decode_no_bounds_returns_float_indices():
    dec = QTTDecoder(num_vars=2, levels=2, base=2)
    q = np.array([1, 1, 0, 1])  # var0->3, var1->1
    x = dec.decode(q)
    np.testing.assert_allclose(x, [3.0, 1.0])


def test_decoder_decode_bounds_lower_upper():
    # index 0 -> lower bound, index N-1 -> upper bound (endpoint=True)
    L, base = 3, 2
    bounds = [(-5.0, 5.0), (0.0, 1.0)]
    dec = QTTDecoder(num_vars=2, levels=L, base=base, bounds=bounds, endpoint=True)

    # All-zero digits -> index 0 for both vars -> lower bounds
    q_low = np.zeros(2 * L, dtype=int)
    x_low = dec.decode(q_low)
    np.testing.assert_allclose(x_low, [-5.0, 0.0])

    # All-one digits -> index 7 for both vars -> upper bounds
    q_high = np.ones(2 * L, dtype=int)
    x_high = dec.decode(q_high)
    np.testing.assert_allclose(x_high, [5.0, 1.0])


def test_decoder_decode_bounds_midpoint():
    # L=1, base=2: indices 0 and 1, endpoint=True -> u in {0, 1}
    # L=2, base=2: index 2 -> u=2/3, mapped to a + 2/3*(b-a)
    L, base = 2, 2
    bounds = [(0.0, 3.0)]
    dec = QTTDecoder(num_vars=1, levels=L, base=base, bounds=bounds, endpoint=True)
    # digits [1,0] -> index 2 -> u = 2/3 -> x = 0 + 3*(2/3) = 2.0
    q = np.array([1, 0])
    x = dec.decode(q)
    np.testing.assert_allclose(x, [2.0])


def test_decoder_decode_batch_bounds():
    L, base = 2, 2
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    dec = QTTDecoder(num_vars=2, levels=L, base=base, bounds=bounds, endpoint=True)
    # Point 1: all zeros -> (-1, -1)
    # Point 2: all ones  -> ( 1,  1)
    Q = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    X = dec.decode(Q)
    np.testing.assert_allclose(X[0], [-1.0, -1.0])
    np.testing.assert_allclose(X[1], [1.0, 1.0])


# -----------------------------------------------------------------------
# QTTDecoder with permutation (z-permuted)
# -----------------------------------------------------------------------


def test_decoder_z_permuted_decode_indices():
    # num_vars=4, levels=2, z=2
    # perm: group 0 (vars 0,1): (0,0),(1,0),(0,1),(1,1)
    #       group 1 (vars 2,3): (2,0),(3,0),(2,1),(3,1)
    # digit vector: [d00, d10, d01, d11, d20, d30, d21, d31]
    # var0 uses positions 0 (ell=0) and 2 (ell=1)
    # var1 uses positions 1 (ell=0) and 3 (ell=1)
    perm = qtt_z_permuted_coordinate_map(4, 2, z=2)
    dec = QTTDecoder(num_vars=4, levels=2, base=2, permutation=perm)

    # Set var0=[1,1]->3, var1=[0,1]->1, var2=[1,0]->2, var3=[0,0]->0
    # Positions in digit vector:
    #   var0 ell=0 -> pos 0: 1, ell=1 -> pos 2: 1
    #   var1 ell=0 -> pos 1: 0, ell=1 -> pos 3: 1
    #   var2 ell=0 -> pos 4: 1, ell=1 -> pos 6: 0
    #   var3 ell=0 -> pos 5: 0, ell=1 -> pos 7: 0
    q = np.array([1, 0, 1, 1, 1, 0, 0, 0])
    idx = dec.decode_indices(q)
    np.testing.assert_array_equal(idx, [3, 1, 2, 0])


def test_decoder_z_permuted_vs_var_major_different_for_same_digits():
    """Same flat digit vector decoded differently by z-permuted vs var-major."""
    num_vars, levels, base = 4, 2, 2
    perm = qtt_z_permuted_coordinate_map(num_vars, levels, z=2)

    dec_std = QTTDecoder(num_vars=num_vars, levels=levels, base=base)
    dec_z = QTTDecoder(num_vars=num_vars, levels=levels, base=base, permutation=perm)

    q = np.array([1, 0, 0, 1, 1, 0, 1, 0])  # arbitrary non-trivial digits
    idx_std = dec_std.decode_indices(q)
    idx_z = dec_z.decode_indices(q)

    # They should generally differ (different grouping of digits per var)
    assert not np.array_equal(idx_std, idx_z)


def test_decoder_permutation_invalid_length_raises():
    perm = [(0, 0), (0, 1)]  # too short for num_vars=2, levels=2 (needs 4)
    with pytest.raises(ValueError, match="length"):
        QTTDecoder(num_vars=2, levels=2, base=2, permutation=perm)


def test_decoder_permutation_invalid_content_raises():
    # Wrong pairs: duplicate (0,0) instead of (1,1)
    perm = [(0, 0), (0, 1), (1, 0), (0, 0)]
    with pytest.raises(ValueError, match="permutation of all"):
        QTTDecoder(num_vars=2, levels=2, base=2, permutation=perm)


def test_decoder_z_permuted_roundtrip_with_bounds():
    """Encode known physical points to digit space and decode back."""
    num_vars, levels, base = 6, 4, 2
    bounds = [(-5.0, 5.0)] * num_vars
    perm = qtt_z_permuted_coordinate_map(num_vars, levels, z=3)
    dec = QTTDecoder(
        num_vars=num_vars,
        levels=levels,
        base=base,
        bounds=bounds,
        endpoint=True,
        permutation=perm,
    )

    # Extreme points: all-zero digits -> all lower bounds
    q_low = np.zeros(num_vars * levels, dtype=int)
    x_low = dec.decode(q_low)
    np.testing.assert_allclose(x_low, [-5.0] * num_vars)

    # All-one digits -> all upper bounds
    q_high = np.ones(num_vars * levels, dtype=int)
    x_high = dec.decode(q_high)
    np.testing.assert_allclose(x_high, [5.0] * num_vars)


# -----------------------------------------------------------------------
# make_qtt_objective
# -----------------------------------------------------------------------


def _simple_decoder(num_vars: int = 2, levels: int = 3, base: int = 2):
    bounds = [(0.0, 1.0)] * num_vars
    return QTTDecoder(
        num_vars=num_vars, levels=levels, base=base, bounds=bounds, endpoint=True
    )


def test_make_qtt_objective_single_point():
    """1D input (single digit vector) should return a scalar."""
    dec = _simple_decoder()
    calls = {"n": 0}

    def phys_fn(x: np.ndarray) -> float:
        calls["n"] += 1
        return float(np.sum(x))

    obj = make_qtt_objective(phys_fn, dec)

    total_cores = dec.total_cores  # 2*3 = 6
    q = np.zeros(total_cores, dtype=int)
    result = obj(q)

    assert np.isscalar(result) or np.asarray(result).ndim == 0
    assert calls["n"] == 1


def test_make_qtt_objective_batch_vectorized():
    """2D input with a vectorized physical fn should call it exactly once."""
    dec = _simple_decoder(num_vars=2, levels=2, base=2)
    calls = {"n": 0}

    def phys_fn(X: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return np.sum(X, axis=-1)

    obj = make_qtt_objective(phys_fn, dec)

    total_cores = dec.total_cores  # 4
    Q = np.zeros((5, total_cores), dtype=int)
    result = obj(Q)

    assert np.asarray(result).shape == (5,)
    assert calls["n"] == 1


def test_make_qtt_objective_batch_fallback():
    """2D input with a pointwise-only physical fn should fall back to per-row calls."""
    dec = _simple_decoder(num_vars=2, levels=2, base=2)
    calls = {"n": 0}

    def phys_fn(x: np.ndarray) -> float:
        if np.asarray(x).ndim != 1:
            raise TypeError("pointwise only")
        calls["n"] += 1
        return float(np.sum(x))

    obj = make_qtt_objective(phys_fn, dec)

    n_points = 4
    total_cores = dec.total_cores
    Q = np.zeros((n_points, total_cores), dtype=int)
    result = obj(Q)

    assert np.asarray(result).shape == (n_points,)
    assert calls["n"] == n_points


def test_make_qtt_objective_output_consistent_with_decoder():
    """Objective evaluated at all-zero digits should equal physical_fn at lower bounds."""
    num_vars, levels, base = 3, 2, 2
    bounds = [(-2.0, 2.0), (0.0, 4.0), (1.0, 3.0)]
    dec = QTTDecoder(
        num_vars=num_vars, levels=levels, base=base, bounds=bounds, endpoint=True
    )

    def phys_fn(x: np.ndarray) -> float:
        return float(np.prod(x))

    obj = make_qtt_objective(phys_fn, dec)

    q_low = np.zeros(num_vars * levels, dtype=int)
    result = float(obj(q_low))

    # all-zero digits -> lower bounds -> product = (-2)*0*1 = 0
    expected = float(np.prod([b[0] for b in bounds]))
    np.testing.assert_allclose(result, expected)


def test_make_qtt_objective_wrong_ndim_raises():
    dec = _simple_decoder()
    obj = make_qtt_objective(lambda x: float(np.sum(x)), dec)
    with pytest.raises(ValueError, match="1D or 2D"):
        obj(np.zeros((2, 3, 4), dtype=int))
