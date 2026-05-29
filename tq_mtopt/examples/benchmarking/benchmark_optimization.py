"""Top-level command line interface to run TRC/MTC/TTOpt/... benchmarks and generate plots.

Example:
python benchmark_optimization.py \
--num_dimensions 7 --num_grid_points 5 --ranks 1 2 3 4 5 \
--num_sweeps 6 --seed 42 --num_experiments 10 \
--functions Ackley Alpine1 Brown Exponential Griewank Qing Rastrigin Schaffer \
--methods TRC MTC TTOpt DA DE
"""

from __future__ import annotations
import os
import argparse

from functions import FUNCTION_REGISTRY, F_OPT, get_tests, resolve_f_opt_map
from runner import compare_all, save_best_errors_csv
from plot import make_plots


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Benchmark runner")
    p.add_argument("--num_dimensions", type=int, default=3, help="Problem dimension")
    p.add_argument(
        "--num_grid_points", type=int, default=10, help="Grid points per dimension"
    )
    p.add_argument(
        "--grid_type",
        type=str,
        choices=["plain", "qtt"],
        default="plain",
        help=(
            "Grid parameterization for mtopt methods. "
            "'plain' = standard grids with num_grid_points per dimension; "
            "'qtt' = quantized tensor-train grid (digits) with base and levels."
        ),
    )
    p.add_argument(
        "--qtt_levels",
        type=int,
        default=16,
        help="QTT levels L per physical variable (N = base**L points)",
    )
    p.add_argument(
        "--qtt_base",
        type=int,
        default=2,
        help="QTT base b per digit (typically 2)",
    )
    p.add_argument(
        "--qtt_z",
        type=int,
        default=3,
        help="Group size for z-permuted QTT (used by TRC-z); z=1 recovers var-major order",
    )
    p.add_argument("--ranks", nargs="+", type=int, default=list(range(1, 5)))
    p.add_argument("--num_sweeps", type=int, default=6, help="Number of sweeps")
    p.add_argument("--seed", nargs="+", type=int, default=list(range(10)))
    p.add_argument(
        "--num_experiments", type=int, help="Number of experiments to average over"
    )
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["TRC", "MTC", "TTOpt"],
        help="Optimizers to run (TRC, MTC, TTOpt)",
    )
    p.add_argument(
        "--functions",
        nargs="+",
        help=f"Functions to run (choices: {list(FUNCTION_REGISTRY.keys())})",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        help="Error thresholds for true calls-to-threshold logging (e.g. 1e-2 1e-3 1e-4 1e-5).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results and plots",
    )
    return p.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    tests = get_tests(args.num_dimensions, args.functions)

    # dimension-aware f_opt map (fills Michalewicz for D=5 or D=10)
    f_opt_map = resolve_f_opt_map(F_OPT, args.num_dimensions)

    os.makedirs(args.out_dir, exist_ok=True)

    # Run one function at a time and checkpoint after each so a crash loses
    # at most one function's results.
    all_dfs = []
    for fn_name, fn_callable, fn_bounds in tests:
        results_path = os.path.join(args.out_dir, f"results_{fn_name}.csv")
        if os.path.exists(results_path):
            print(f"Skipping {fn_name} (already saved at {results_path})")
            import pandas as pd

            all_dfs.append(pd.read_csv(results_path))
            continue

        print(f"\n── {fn_name} ─────────────────────────────────────")
        df_fn = compare_all(
            num_dimensions=args.num_dimensions,
            num_grid_points=args.num_grid_points,
            num_experiments=args.num_experiments,
            ranks=args.ranks,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
            tests=[(fn_name, fn_callable, fn_bounds)],
            methods=args.methods,
            f_opt_map=f_opt_map,
            thresholds=tuple(args.thresholds),
            grid_type=args.grid_type,
            qtt_levels=args.qtt_levels,
            qtt_base=args.qtt_base,
            qtt_z=args.qtt_z,
        )
        df_fn.to_csv(results_path, index=False)
        print(f"Saved results -> {results_path}")
        save_best_errors_csv(
            df_fn,
            f_opt_map,
            path=os.path.join(args.out_dir, f"best_errors_{fn_name}.csv"),
            sci_digits=1,
        )
        all_dfs.append(df_fn)

    import pandas as pd

    df_results = pd.concat(all_dfs, ignore_index=True)

    # Plots use the resolved map
    make_plots(df_results, f_opt_map, out_dir=os.path.join(args.out_dir, "plots"))


if __name__ == "__main__":
    main()
