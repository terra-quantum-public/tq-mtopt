"""
Function representation module.

Three formats are provided:

CP / tensor-rank cross (TRC)
-----------------------------
    f_approx(x) = sum_p  w_p * prod_k  F_k[i_k, p]

Built from an optimizer skeleton (r full-dimensional pivot points).  The
factor matrices F_k (shape N_k x r) and weight vector w (shape r) are stored;
evaluation is O(r * d) per point.

TT / tensor-train cross
------------------------
    f_approx(x) = G_0[0,i_0,:] @ G_1[:,i_1,:] @ ... @ G_{d-1}[:,i_{d-1},0]

Built directly from f via alternating maxvol sweeps (no optimizer needed).
Each core G_k has shape (r_{k-1}, N_k, r_k) with max bond dimension r.
Following the paper's A^{-1} formulation, each core is computed as

    G_k = C @ pinv(C_sel)   where C_sel = C[maxvol_rows, :]

which guarantees exact interpolation at the selected cross rows.

MT / matrix-train cross (chain graph)
--------------------------------------
Built in a single left-to-right pass from the MTC optimizer skeleton
(r full-dimensional pivot points): the skeleton fixes both multi-index
sets I_{k-1} and J_k at every site, with no max-volume selection and no
alternating sweeps.  The cross section at bond k is the r x r matrix
A_k[a, b] = C^(k)[a, idx(a,k), b] and A_k^{-1} = pinv(A_k) (truncated-SVD
pseudo-inverse).  Cost: r*N for the boundary sites plus r^2*N per interior
site.  Evaluation contracts fiber tensors and cross inverses along the chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from tq_mtopt.grid import Grid
from tq_mtopt.maxvol import maxvol
from tq_mtopt.optimization import create_mutations, evaluate_grid


# ---------------------------------------------------------------------------
# Shared index helper
# ---------------------------------------------------------------------------


def _nearest_index(gk: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the index of the nearest grid point for each value in *values*."""
    idx = np.clip(np.searchsorted(gk, values, side="left"), 1, len(gk) - 1)
    left_dist = np.abs(values - gk[idx - 1])
    right_dist = np.abs(values - gk[idx])
    return np.where(left_dist <= right_dist, idx - 1, idx).astype(int)


# ===========================================================================
# CP / TRC representation
# ===========================================================================


@dataclass
class TRCRepresentation:
    """
    CP-format cross approximation of a function on a product grid.

    Parameters
    ----------
    factor_matrices:
        List of d arrays of shape ``(N_k, r)``.  ``factor_matrices[k][i, p]``
        is f evaluated with coordinate k set to ``primitive_grids[k][i]``
        and all other coordinates taken from skeleton point p.
    weights:
        Length-r weight vector satisfying f_approx(skeleton[p]) = f(skeleton[p]).
    primitive_grids:
        The 1-D grids for each coordinate.
    skeleton:
        The r full-dimensional pivot points (shape r x d).
    """

    factor_matrices: list[np.ndarray]
    weights: np.ndarray
    primitive_grids: list[Grid]
    skeleton: Grid

    def _coord_index(self, k: int, values: np.ndarray) -> np.ndarray:
        return _nearest_index(self.primitive_grids[k].grid[:, 0], values)

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        v = np.ones(self.weights.shape[0])
        for k, F_k in enumerate(self.factor_matrices):
            i_k = int(self._coord_index(k, x[k : k + 1])[0])
            v *= F_k[i_k, :]
        return float(self.weights @ v)

    def evaluate_batch(self, X: Union[np.ndarray, Grid]) -> np.ndarray:
        if isinstance(X, Grid):
            X = X.grid
        X = np.asarray(X, dtype=float)
        V = np.ones((X.shape[0], self.weights.shape[0]))
        for k, F_k in enumerate(self.factor_matrices):
            V *= F_k[self._coord_index(k, X[:, k]), :]
        return V @ self.weights


def build_trc_representation(
    skeleton: Grid,
    primitive_grids: list[Grid],
    function: Callable,
    rcond: float = 1e-8,
    **kwargs,
) -> TRCRepresentation:
    """
    Build a CP-format representation from a TRC or MTC optimizer skeleton.

    Computes factor matrices F_k (one fiber pass per coordinate), normalises
    columns to keep the cross weight matrix M well-conditioned, then solves
    M @ w = b via truncated SVD.

    Parameters
    ----------
    skeleton:
        Final skeleton grid of shape ``(r, d)``.
    primitive_grids:
        List of d one-dimensional :class:`Grid` objects.
    function:
        The callable used during optimisation.
    rcond:
        Condition-number cutoff for the weight lstsq solve.
    **kwargs:
        Forwarded to every ``function`` call.
    """
    r = skeleton.num_points()
    d = skeleton.num_coords()

    # Factor matrices F_k of shape (N_k, r).
    factor_matrices: list[np.ndarray] = []
    for k in range(d):
        candidates, _ = create_mutations(skeleton, primitive_grids[k])
        F_k = evaluate_grid(
            candidates, function, primitive_grids[k].num_points(), **kwargs
        )
        factor_matrices.append(F_k)

    # Column-normalise so M entries stay O(1).
    factor_matrices_norm = []
    for F_k in factor_matrices:
        scale_k = np.maximum(np.max(np.abs(F_k), axis=0), 1e-300)
        factor_matrices_norm.append(F_k / scale_k[np.newaxis, :])

    # Cross weight matrix M_norm of shape (r, r).
    M_norm = np.ones((r, r))
    for k, F_k_norm in enumerate(factor_matrices_norm):
        gk_vals = primitive_grids[k].grid[:, 0]
        sk_vals = skeleton.grid[:, k]
        idxs = np.clip(
            np.searchsorted(gk_vals, sk_vals, side="left"), 0, len(gk_vals) - 1
        )
        M_norm *= F_k_norm[idxs, :]

    b = skeleton.evaluate(function, **kwargs)
    w_eff, _, _, _ = np.linalg.lstsq(M_norm, b, rcond=rcond)

    return TRCRepresentation(
        factor_matrices=factor_matrices_norm,
        weights=w_eff,
        primitive_grids=primitive_grids,
        skeleton=skeleton,
    )


# ===========================================================================
# TT / MT shared helpers and core classes
# ===========================================================================


class TTRepresentation:
    """
    TT-format cross approximation of a function on a product grid.

    Evaluation:
        f_approx(x) = G_0[0,i_0,:] @ G_1[:,i_1,:] @ ... @ G_{d-1}[:,i_{d-1},0]

    where core G_k has shape (r_{k-1}, N_k, r_k).
    """

    def __init__(self, cores: list[np.ndarray], primitive_grids: list[Grid]):
        self.cores = cores
        self.primitive_grids = primitive_grids

    def _coord_index(self, k: int, value: float) -> int:
        return int(
            _nearest_index(
                self.primitive_grids[k].grid[:, 0],
                np.array([value]),
            )[0]
        )

    def evaluate(self, x: np.ndarray) -> float:
        v = np.ones(1)
        for k, G_k in enumerate(self.cores):
            i_k = self._coord_index(k, float(x[k]))
            v = v @ G_k[:, i_k, :]
        return float(v.ravel()[0])

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(X[j]) for j in range(len(X))])


class MTRepresentation:
    """
    Chain-graph matrix-train (MT) cross approximation.

    Stores the fundamental objects of the paper's MT formulation
    (Section 2, Eq. 1) rather than precomputed TT cores:

    skeleton_grids[k]   (k = 0..d-2): shape (r_k, d)
        Full-dimensional pivot points at bond k (grid-snapped optimizer
        skeleton; identical at every bond for the single-pass construction).

    factor_matrices[k]  (k = 0..d-1): shape (r_{k-1}, N_k, r_k)
        The fiber matrix F_k evaluated from f using the left context I_{k-1}
        and right context J_k, both fixed by the skeleton.

    cross_inverses[k]   (k = 0..d-2): shape (r_k, r_{k+1})
        A_k^{-1} = pinv(A_k), the truncated-SVD pseudo-inverse of the
        skeleton-selected cross section — the A^{-1} factor from the paper.

    Evaluation contracts the chain without pre-multiplying F_k and A_k:
        v = np.ones(1)
        for k in range(d):
            v = v @ factor_matrices[k][:, i_k, :]   # (r_{k-1},) → (r_k,)
            if k < d-1:
                v = v @ cross_inverses[k]            # (r_k,) → (r_{k+1},)
    """

    def __init__(
        self,
        skeleton_grids: list[np.ndarray],
        factor_matrices: list[np.ndarray],
        cross_inverses: list[np.ndarray],
        primitive_grids: list[Grid],
    ):
        self.skeleton_grids = skeleton_grids  # d-1 arrays, shape (r, d)
        self.factor_matrices = factor_matrices  # d arrays, shape (r_left, N_k, r_right)
        self.cross_inverses = cross_inverses  # d-1 arrays, shape (r_right, r_new)
        self.primitive_grids = primitive_grids

    def _coord_index(self, k: int, value: float) -> int:
        return int(
            _nearest_index(
                self.primitive_grids[k].grid[:, 0],
                np.array([value]),
            )[0]
        )

    def evaluate(self, x: np.ndarray) -> float:
        v = np.ones(1)
        for k, F_k in enumerate(self.factor_matrices):
            i_k = self._coord_index(k, float(x[k]))
            v = v @ F_k[:, i_k, :]  # fiber slice: (r_left,) → (r_right,)
            if k < len(self.cross_inverses):
                v = v @ self.cross_inverses[k]  # cross inverse: (r_right,) → (r_new,)
        return float(v.ravel()[0])

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(X[j]) for j in range(len(X))])


def _eval_fiber(
    func: Callable,
    k: int,
    grids: list[np.ndarray],
    I_left: np.ndarray,
    J_right: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the fiber at site k for the TT/MT cross algorithm.

    I_left  : (n_left,  k)       integer indices into grids[0..k-1]
    J_right : (n_right, d-k-1)   integer indices into grids[k+1..d-1]
    Returns : (n_left, N_k, n_right)
    """
    d = len(grids)
    N_k = len(grids[k])
    n_left = I_left.shape[0]
    n_right_dims = d - k - 1
    n_right = J_right.shape[0] if J_right.shape[1] > 0 else 1

    fiber = np.zeros((n_left, N_k, n_right))
    x = np.zeros(d)
    for alpha in range(n_left):
        for j in range(k):
            x[j] = grids[j][I_left[alpha, j]]
        for i_k in range(N_k):
            x[k] = grids[k][i_k]
            if J_right.shape[1] == 0:
                fiber[alpha, i_k, 0] = func(x)
            else:
                for beta in range(n_right):
                    for j in range(n_right_dims):
                        x[k + 1 + j] = grids[k + 1 + j][J_right[beta, j]]
                    fiber[alpha, i_k, beta] = func(x)
    return fiber


def _build_tt_cross(
    func: Callable,
    primitives: list[Grid],
    rank: int,
    num_sweeps: int,
    seed: int,
    rcond: float = 1e-8,
    initial_J: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Core TT cross algorithm: alternating maxvol left-to-right / right-to-left sweeps.

    Parameters
    ----------
    initial_J:
        Optional warm-start right multi-indices.  If provided, J[k] is
        initialised from initial_J[k] instead of random integers.  The list
        must have length d; initial_J[d-1] is ignored (always zeros((1,0))).

    Returns
    -------
    cores:
        TT cores G_k of shape (r_{k-1}, N_k, r_k).
    factor_matrices:
        Raw fiber matrices F_k of shape (r_{k-1}, N_k, r_k), one per site.
        factor_matrices[k] = fiber evaluated with the final left/right contexts.
    cross_inverses:
        A_k = pinv(C_sel) of shape (r_k, r_{k+1}), one per internal bond (d-1 total).
        The A^{-1} cross factor from the paper.
    skeleton_grids:
        List of d-1 arrays of shape (r_k, d), one per bond.
        skeleton_grids[k][i] is the i-th full-dimensional pivot at bond k,
        assembled from the final left multi-index I_left[k] (coords 0..k) and
        the right multi-index J[k] (coords k+1..d-1).
    """
    d = len(primitives)
    grids = [p.grid[:, 0] for p in primitives]
    N = [len(g) for g in grids]
    rng = np.random.default_rng(seed)

    # Initialise right multi-indices J[k]: shape (r_k, d-k-1).
    J: list[np.ndarray] = [None] * d  # type: ignore[list-item]
    J[d - 1] = np.zeros((1, 0), dtype=int)
    for k in range(d - 2, -1, -1):
        if initial_J is not None and initial_J[k] is not None and len(initial_J[k]) > 0:
            # Warm start: use provided right multi-indices (clip to valid grid range).
            J_init = np.asarray(initial_J[k], dtype=int)
            n_right_dims = d - k - 1
            for col in range(n_right_dims):
                J_init[:, col] = np.clip(J_init[:, col], 0, N[k + 1 + col] - 1)
            J[k] = J_init
        else:
            n_right_dims = d - k - 1
            # Compute product in Python ints (avoids int64 overflow at high d)
            # and exit early once it exceeds rank.
            _prod = 1
            for _j in range(n_right_dims):
                _prod *= int(N[k + 1 + _j])
                if _prod >= rank:
                    _prod = rank
                    break
            r_init = _prod
            J[k] = np.column_stack(
                [
                    rng.integers(0, N[k + 1 + j], size=r_init)
                    for j in range(n_right_dims)
                ]
            )

    cores: list[np.ndarray | None] = [None] * d
    factor_matrices: list[np.ndarray | None] = [None] * d
    cross_inverses: list[np.ndarray | None] = [None] * (d - 1)
    # Track the left multi-index after processing each site in the final sweep.
    final_I_lefts: list[np.ndarray | None] = [None] * d

    for sweep_idx in range(num_sweeps):
        # ── Left-to-right: build TT cores ─────────────────────────────────
        I_left = np.zeros((1, 0), dtype=int)

        for k in range(d):
            fiber = _eval_fiber(func, k, grids, I_left, J[k])
            n_left = I_left.shape[0]
            n_right = fiber.shape[2]
            C = fiber.reshape(n_left * N[k], n_right)

            if k < d - 1:
                if C.shape[0] <= rank:
                    selected = list(range(C.shape[0]))
                else:
                    selected, _ = maxvol(C)
                r_new = len(selected)

                C_sel = C[selected, :]
                # Core = C @ pinv(C_sel): solve C_sel^T X = C^T.
                G_T, _, _, _ = np.linalg.lstsq(C_sel.T, C.T, rcond=rcond)
                cores[k] = G_T.T.reshape(n_left, N[k], r_new)
                factor_matrices[k] = fiber  # (n_left, N[k], n_right)
                cross_inverses[k] = np.linalg.pinv(
                    C_sel, rcond=rcond
                )  # (n_right, r_new)

                new_I = np.zeros((r_new, k + 1), dtype=int)
                for new_idx, row_flat in enumerate(selected):
                    alpha = row_flat // N[k]
                    i_k = row_flat % N[k]
                    if k > 0:
                        new_I[new_idx, :k] = I_left[alpha]
                    new_I[new_idx, k] = i_k
                I_left = new_I
                final_I_lefts[k] = new_I
            else:
                cores[k] = fiber
                factor_matrices[k] = fiber
                final_I_lefts[k] = I_left

        # ── Right-to-left: update J contexts for the next sweep ────────────
        # Skipped on the last sweep, so J retains the values used above.
        if sweep_idx < num_sweeps - 1:
            J_right_running = np.zeros((1, 0), dtype=int)
            for k in range(d - 1, 0, -1):
                n_left_k = max(1, min(rank, int(np.prod(N[:k]))))
                I_left_k = (
                    np.column_stack(
                        [rng.integers(0, N[j], size=n_left_k) for j in range(k)]
                    )
                    if k > 0
                    else np.zeros((1, 0), dtype=int)
                )

                fiber = _eval_fiber(func, k, grids, I_left_k, J_right_running)
                n_left_f, _, n_right_f = fiber.shape
                C_T = fiber.reshape(n_left_f, N[k] * n_right_f)

                if C_T.shape[1] <= rank:
                    selected_cols = list(range(C_T.shape[1]))
                else:
                    selected_cols, _ = maxvol(C_T.T)
                r_new = len(selected_cols)

                new_J = np.zeros((r_new, d - k), dtype=int)
                for new_idx, col_flat in enumerate(selected_cols):
                    i_k = col_flat // n_right_f
                    beta = col_flat % n_right_f
                    new_J[new_idx, 0] = i_k
                    if J_right_running.shape[1] > 0:
                        new_J[new_idx, 1:] = J_right_running[beta]
                J[k - 1] = new_J
                J_right_running = new_J

    # ── Assemble skeleton grids from final I_left and J ────────────────────
    # After the last L→R sweep J is not modified (R→L is skipped), so J[k]
    # holds the right multi-indices that were used in the final sweep.
    # skeleton_grids[k][i] is the i-th full-dimensional pivot at bond k.
    skeleton_grids: list[np.ndarray] = []
    for k in range(d - 1):
        I_k = final_I_lefts[k]  # (r_left, k+1)  — left coords 0..k
        J_k = J[k]  # (r_right, d-k-1) — right coords k+1..d-1
        r_sk = min(I_k.shape[0], J_k.shape[0])

        S_k = np.zeros((r_sk, d))
        for j in range(k + 1):
            S_k[:, j] = grids[j][I_k[:r_sk, j]]
        for j in range(d - k - 1):
            S_k[:, k + 1 + j] = grids[k + 1 + j][J_k[:r_sk, j]]
        skeleton_grids.append(S_k)

    return cores, factor_matrices, cross_inverses, skeleton_grids  # type: ignore[return-value]


def build_tt_representation(
    func: Callable,
    primitives: list[Grid],
    rank: int,
    num_sweeps: int,
    seed: int,
    rcond: float = 1e-8,
) -> TTRepresentation:
    """
    Build a TT cross approximation via alternating maxvol sweeps.

    Each core is computed as G_k = C @ pinv(C_sel) where C_sel contains the
    maxvol-selected rows of the fiber matrix C — the A^{-1} formulation from
    the paper.  This guarantees exact interpolation at the selected cross rows.

    Parameters
    ----------
    func:
        The function to approximate.
    primitives:
        List of d one-dimensional :class:`Grid` objects.
    rank:
        Maximum TT bond dimension.
    num_sweeps:
        Number of left-to-right + right-to-left alternating sweeps.
    seed:
        RNG seed for initialising the right multi-index sets.
    rcond:
        Condition-number cutoff for the per-core lstsq solve.
    """
    cores, _, _, _ = _build_tt_cross(func, primitives, rank, num_sweeps, seed, rcond)
    return TTRepresentation(cores, primitives)


def build_mt_representation(
    skeleton: Grid,
    primitive_grids: list[Grid],
    function: Callable,
    num_sweeps: int = 0,  # unused; kept for API compatibility
    rcond: float = 1e-8,
    seed: int = 0,  # unused; kept for API compatibility
) -> MTRepresentation:
    """
    Build a matrix-train representation directly from the MTC optimizer skeleton.

    Uses a single left-to-right pass with no TT-cross sweeps and no max-volume
    selection.  The optimizer skeleton fixes both multi-index sets for every site:

        I_{k-1}[α, j] = argmin_i |s^(α)_j - x_j^(i)|   for j = 0..k-1
        J_k[β, j]     = argmin_i |s^(β)_{k+1+j} - x_{k+1+j}^(i)|   for j = 0..d-k-2

    The cross section at bond k is the r×r matrix

        A_k[α, β] = C^(k)[α, idx[α,k], β],   idx[α,k] = argmin_i |s^(α)_k - x_k^(i)|

    and A_k^{-1} = pinv(A_k).  Total cost: r*N evaluations for boundary sites
    (k=0 and k=d-1) and r^2*N for each interior site.

    Parameters
    ----------
    skeleton:
        Final skeleton grid from the MTC optimizer, shape (r, d).
    primitive_grids:
        List of d one-dimensional :class:`Grid` objects.
    function:
        The objective function (raw, without score transform).
    rcond:
        Condition-number cutoff for pinv.
    """
    d = len(primitive_grids)
    grids = [p.grid[:, 0] for p in primitive_grids]
    r = skeleton.num_points()

    # Map skeleton coordinates to nearest grid indices.
    skel_idx = np.zeros((r, d), dtype=int)
    for k in range(d):
        skel_idx[:, k] = _nearest_index(grids[k], skeleton.grid[:, k])

    # Skeleton physical coordinates (grid-snapped).
    skel_pts = np.zeros((r, d))
    for k in range(d):
        skel_pts[:, k] = grids[k][skel_idx[:, k]]

    factor_matrices: list[np.ndarray] = []
    cross_inverses: list[np.ndarray] = []

    for k in range(d):
        # Left context: all r skeleton points' first k coordinates (trivial for k=0).
        I_left = skel_idx[:, :k] if k > 0 else np.zeros((1, 0), dtype=int)
        # Right context: all r skeleton points' last d-k-1 coordinates (trivial for k=d-1).
        J_right = skel_idx[:, k + 1 :] if k < d - 1 else np.zeros((1, 0), dtype=int)

        # fiber shape: (n_left, N_k, n_right)
        # n_left = r if k > 0 else 1,  n_right = r if k < d-1 else 1.
        fiber = _eval_fiber(function, k, grids, I_left, J_right)
        factor_matrices.append(fiber)

        if k < d - 1:
            if k == 0:
                # n_left = 1; select r pivot rows from the (N_k, r) slice fiber[0].
                C_sel = fiber[0, skel_idx[:, k], :]  # (r, r)
            else:
                # For each α, pivot row is (α, skel_idx[α, k]).
                C_sel = fiber[np.arange(r), skel_idx[:, k], :]  # (r, r)
            cross_inverses.append(np.linalg.pinv(C_sel, rcond=rcond))

    skeleton_grids = [skel_pts.copy() for _ in range(d - 1)]
    return MTRepresentation(skeleton_grids, factor_matrices, cross_inverses, primitive_grids)
