"""
Approximation-error benchmark: TRC vs MTC vs TT function representation.

For each benchmark function and each seed we:
  1. Run TRC / MTC optimisation to convergence  (or TT cross directly from f).
  2. Build a representation from the result.
  3. Sample a held-out test set (shared across methods for a given seed).
  4. Evaluate the approximation and the true function on the test set.
  5. Record relative RMSE, mean absolute error, and max absolute error.

Methods
-------
TRC — CP-format cross from a TRC optimiser skeleton (plain or QTT grids).
MTC — single-pass matrix-train cross from an MTC optimiser skeleton
      (plain grids; no max-volume selection, no alternating sweeps).
TT  — TT cross built directly from f via alternating maxvol sweeps
      (always plain grids; QTT mode is not applicable).

Defaults match the optimisation benchmark in the paper:
  d=10, grid_type=plain, N=20, r=8, sweeps=10, functions=F1-F11.

Usage
-----
# All three methods, plain grids
python benchmark_representation.py --methods TRC MTC TT

# Quick smoke-test
python benchmark_representation.py --num_seeds 3 --n_test 200

python benchmark_representation.py [--out_dir DIR] [--num_dimensions D]
       [--grid_type plain|qtt] [--qtt_levels L] [--qtt_base B]
       [--num_grid_points N] [--rank R] [--num_sweeps S] [--num_seeds K]
       [--n_test T] [--methods M1 M2 ...] [--functions F1 F2 ...]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from functions import FUNCTION_REGISTRY
from helpers import make_primitives, _seed_all, _make_ttopt_score_transform

from tq_mtopt.grid import Grid
from tq_mtopt.optimization import (
    Objective,
    TensorRankOptimization,
    MatrixTrainOptimization,
    random_grid_points,
)
from tq_mtopt.qtt import QTTDecoder, make_qtt_objective
from tq_mtopt.representation import (
    build_trc_representation,
    build_tt_representation,
    build_mt_representation,
)

F1_TO_F11 = [
    "Ackley",
    "Alpine1",
    "Brown",
    "Exponential",
    "Griewank",
    "Michalewicz",
    "Qing",
    "Rastrigin",
    "Schaffer",
    "Schwefel",
    "Multiwell",
]
ALL_FUNCTIONS = list(FUNCTION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_test_grid(primitives: List[Grid], n_test: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(primitives)
    return np.stack(
        [
            primitives[k].grid[rng.integers(0, primitives[k].num_points(), n_test), 0]
            for k in range(d)
        ],
        axis=1,
    )


def _eval_true(func: Callable, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([func(x) for x in X], dtype=float)


def _errors(true_vals: np.ndarray, approx_vals: np.ndarray):
    diff = approx_vals - true_vals
    mae = float(np.mean(np.abs(diff)))
    max_ae = float(np.max(np.abs(diff)))
    rms_true = float(np.sqrt(np.mean(true_vals**2))) + 1e-30
    rel_rmse = float(np.sqrt(np.mean(diff**2))) / rms_true
    return rel_rmse, mae, max_ae


def _make_func(
    func_raw: Callable,
    bounds: List[Tuple[float, float]],
    num_dimensions: int,
    *,
    grid_type: str,
    qtt_levels: int,
    qtt_base: int,
) -> Callable:
    if grid_type == "qtt":
        decoder = QTTDecoder(
            num_vars=num_dimensions,
            levels=qtt_levels,
            base=qtt_base,
            bounds=bounds,
        )
        return make_qtt_objective(func_raw, decoder)
    return func_raw


# ---------------------------------------------------------------------------
# Per-method runner
# ---------------------------------------------------------------------------


def run_representation(
    method: str,
    func: Callable,
    primitives: List[Grid],
    rank: int,
    num_sweeps: int,
    seed: int,
    X_test: np.ndarray,
    true_vals: np.ndarray,
) -> dict:
    """Run one method/seed and return representation quality metrics.

    Three representation types:
      TRC — CP-format cross approximation built from a TRC optimizer skeleton.
      TT  — TT cross built directly from f via alternating maxvol sweeps.
      MTC — single-pass matrix-train cross built from an MTC optimizer skeleton:
            the skeleton fixes both multi-index sets at every site (no max-volume
            selection, no sweeps); stores fiber matrices and cross inverses
            (MTRepresentation, not CP format).

    For TRC, ``func`` may be QTT-wrapped and ``primitives`` match the grid type.
    For TT/MTC, ``func`` must be the raw physical function on plain-grid primitives.
    """
    _seed_all(seed)

    if method == "TT":
        rep = build_tt_representation(func, primitives, rank, num_sweeps, seed)
        calls_opt = 0
        # Exact count for _build_tt_cross with N_k >= rank:
        #   each L->R pass:  r*N_0 + r^2*sum(N_1..N_{d-2}) + r*N_{d-1}
        #   each R->L pass:  r*N_{d-1} + r^2*sum(N_1..N_{d-2})   (skipped after last L->R)
        Ns = [p.num_points() for p in primitives]
        inner = rank * rank * sum(Ns[1:-1]) if len(Ns) > 2 else 0
        lr = rank * Ns[0] + inner + rank * Ns[-1]
        rl = rank * Ns[-1] + inner
        calls_rep = num_sweeps * lr + (num_sweeps - 1) * rl
    elif method == "MTC":
        score = _make_ttopt_score_transform(mode="min")
        obj = Objective(func, score)
        grid_start = random_grid_points(primitives, rank, seed)
        model = MatrixTrainOptimization(primitives)
        final_skeleton = model.optimize(grid_start, obj, num_sweeps)
        calls_opt = obj.function_calls
        rep = build_mt_representation(final_skeleton, primitives, func)
        # Single-pass cost: r*N for boundary sites, r^2*N per interior site.
        d = len(primitives)
        Ns = [p.num_points() for p in primitives]
        calls_rep = (rank * Ns[0] + rank * Ns[-1] +
                     rank * rank * sum(Ns[1:-1])) if d > 2 else rank * sum(Ns)
    elif method == "TRC":
        score = _make_ttopt_score_transform(mode="min")
        obj = Objective(func, score)
        grid_start = random_grid_points(primitives, rank, seed)
        model = TensorRankOptimization(primitives)
        final_skeleton = model.optimize(grid_start, obj, num_sweeps)
        calls_opt = obj.function_calls
        rep = build_trc_representation(final_skeleton, primitives, func)
        calls_rep = rank * sum(p.num_points() for p in primitives) + rank
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from TRC, TT, MTC.")

    approx_vals = rep.evaluate_batch(X_test)
    rel_rmse, mae, max_ae = _errors(true_vals, approx_vals)
    return {
        "calls_opt": calls_opt,
        "calls_rep": calls_rep,
        "calls_total": calls_opt + calls_rep,
        "rel_rmse": rel_rmse,
        "mae": mae,
        "max_ae": max_ae,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def benchmark_representation(
    num_dimensions: int,
    num_grid_points: int,
    rank: int,
    num_sweeps: int,
    num_seeds: int,
    n_test: int,
    function_names: List[str],
    methods: List[str],
    *,
    grid_type: str = "plain",
    qtt_levels: int = 20,
    qtt_base: int = 2,
    checkpoint_dir: str | None = None,
) -> pd.DataFrame:
    rows = []
    test_seed_offset = 99999
    chain_methods = {m for m in methods if m in ("TT", "MTC")}
    use_tt = bool(chain_methods)

    for name in tqdm(function_names, desc="functions"):
        func_raw, (lo, hi) = FUNCTION_REGISTRY[name]
        bounds = [(lo, hi)] * num_dimensions

        # Primitives and wrapped function for TRC/MTC.
        primitives = make_primitives(
            bounds,
            num_grid_points,
            grid_type=grid_type,
            qtt_levels=qtt_levels,
            qtt_base=qtt_base,
        )
        func_for_test = _make_func(
            func_raw,
            bounds,
            num_dimensions,
            grid_type=grid_type,
            qtt_levels=qtt_levels,
            qtt_base=qtt_base,
        )

        # TT/MT always need plain-grid primitives and the raw function.
        # Reuse `primitives` when the overall grid_type is already plain.
        if use_tt and grid_type != "plain":
            plain_primitives = make_primitives(
                bounds, num_grid_points, grid_type="plain"
            )
        else:
            plain_primitives = primitives

        for seed in range(num_seeds):
            # Test set for TRC/MTC.
            X_test = _sample_test_grid(primitives, n_test, seed=test_seed_offset + seed)
            true_vals = _eval_true(func_for_test, X_test)

            # Test set for TT (plain grid, raw function).
            # When grid_type=="plain" this is identical to X_test/true_vals above.
            if use_tt and grid_type != "plain":
                X_test_tt = _sample_test_grid(
                    plain_primitives, n_test, seed=test_seed_offset + seed
                )
                true_vals_tt = _eval_true(func_raw, X_test_tt)
            else:
                X_test_tt, true_vals_tt = X_test, true_vals

            for method in methods:
                if method in ("TT", "MTC"):
                    run_func = func_raw
                    run_prim = plain_primitives
                    run_X, run_tv = X_test_tt, true_vals_tt
                else:
                    run_func = _make_func(
                        func_raw,
                        bounds,
                        num_dimensions,
                        grid_type=grid_type,
                        qtt_levels=qtt_levels,
                        qtt_base=qtt_base,
                    )
                    run_prim = primitives
                    run_X, run_tv = X_test, true_vals

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = run_representation(
                            method,
                            run_func,
                            run_prim,
                            rank,
                            num_sweeps,
                            seed,
                            run_X,
                            run_tv,
                        )
                except Exception as e:
                    result = {
                        k: np.nan
                        for k in [
                            "calls_opt",
                            "calls_rep",
                            "calls_total",
                            "rel_rmse",
                            "mae",
                            "max_ae",
                        ]
                    }
                    print(f"  WARNING: {name}/{method}/seed={seed}: {e}")

                effective_grid = "plain" if method in ("TT", "MTC") else grid_type
                rows.append(
                    {
                        "Function": name,
                        "Method": method,
                        "Rank": rank,
                        "Seed": seed,
                        "Dimensions": num_dimensions,
                        "GridType": effective_grid,
                        "QTTLevels": qtt_levels if effective_grid == "qtt" else None,
                        "GridPoints": num_grid_points,
                        "Sweeps": num_sweeps,
                        **result,
                    }
                )

            gc.collect()

        if checkpoint_dir is not None:
            ckpt = os.path.join(checkpoint_dir, "representation_raw_partial.csv")
            pd.DataFrame(rows).to_csv(ckpt, index=False)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def make_summary_table(df: pd.DataFrame, metric: str = "rel_rmse") -> pd.DataFrame:
    agg = df.groupby(["Method", "Function"])[metric].agg(["mean", "std"]).reset_index()
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.2e} ± {r['std']:.2e}", axis=1)
    return agg.pivot(index="Method", columns="Function", values="cell")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Representation quality benchmark (TRC, MTC, TT). "
            "Defaults use plain grids with d=10, r=8, 10 sweeps — matching the "
            "dimension, rank, sweep count, and function suite of the optimisation "
            "benchmark. Plain grids are used so that approximation quality reflects "
            "the function's intrinsic rank in physical space."
        )
    )
    p.add_argument("--num_dimensions", type=int, default=10)
    p.add_argument("--grid_type", choices=["plain", "qtt"], default="plain")
    p.add_argument("--qtt_levels", type=int, default=20)
    p.add_argument("--qtt_base", type=int, default=2)
    p.add_argument("--num_grid_points", type=int, default=20)
    p.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[8],
        help="One or more bond dimensions to sweep over (default: [8])",
    )
    p.add_argument("--num_sweeps", type=int, default=10)
    p.add_argument("--num_seeds", type=int, default=20)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["TRC", "TT", "MTC"],
        help="Methods to benchmark: TRC, TT, MTC (default: all three)",
    )
    p.add_argument(
        "--functions",
        nargs="*",
        default=None,
        help=f"Subset of functions; default is F1-F11: {F1_TO_F11}",
    )
    p.add_argument("--out_dir", type=str, default="representation_results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fns = args.functions or F1_TO_F11

    if args.grid_type == "qtt":
        grid_desc = (
            f"qtt (L={args.qtt_levels}, base={args.qtt_base}, "
            f"{args.qtt_base**args.qtt_levels} pts/var, "
            f"{args.num_dimensions * args.qtt_levels} digit dims)"
        )
    else:
        grid_desc = f"plain (N={args.num_grid_points})"

    print(
        f"d={args.num_dimensions}, grid={grid_desc}, ranks={args.ranks}, "
        f"sweeps={args.num_sweeps}, seeds={args.num_seeds}, "
        f"n_test={args.n_test}, methods={args.methods}"
    )
    print(f"Functions: {fns}")

    all_dfs = []
    for rank in args.ranks:
        print(f"\n── rank={rank} ──────────────────────────────────────")
        df_r = benchmark_representation(
            num_dimensions=args.num_dimensions,
            num_grid_points=args.num_grid_points,
            rank=rank,
            num_sweeps=args.num_sweeps,
            num_seeds=args.num_seeds,
            n_test=args.n_test,
            function_names=fns,
            methods=args.methods,
            grid_type=args.grid_type,
            qtt_levels=args.qtt_levels,
            qtt_base=args.qtt_base,
            checkpoint_dir=args.out_dir,
        )
        all_dfs.append(df_r)
        # Checkpoint after each rank so partial results are safe.
        pd.concat(all_dfs, ignore_index=True).to_csv(
            os.path.join(args.out_dir, "representation_raw.csv"), index=False
        )

    df = pd.concat(all_dfs, ignore_index=True)

    raw_path = os.path.join(args.out_dir, "representation_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results → {raw_path}")

    print("\n=== Mean rel. RMSE by rank (TT/MTC, averaged over functions & seeds) ===")
    mean_tbl = df.groupby(["Method", "Rank"])["rel_rmse"].mean().unstack("Rank")
    print(mean_tbl.to_string(float_format=lambda x: f"{x:.3e}"))
    mean_tbl.to_csv(os.path.join(args.out_dir, "representation_mean_rel_rmse.csv"))

    print("\n=== Mean calls_total ===")
    calls_tbl = (
        df.groupby(["Method", "Function"])["calls_total"].mean().unstack("Function")
    )
    print(calls_tbl.to_string(float_format=lambda x: f"{x:.0f}"))
    calls_tbl.to_csv(os.path.join(args.out_dir, "representation_mean_calls.csv"))


if __name__ == "__main__":
    main()
