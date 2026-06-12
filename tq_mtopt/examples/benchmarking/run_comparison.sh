#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config
# -------------------------
DIMS=10
L=18
BASE=2
RANK=6
SWEEPS=8
EXPS=20
SEED=42
Z=3

METHODS=(TRC TRC-z MTC TTOpt DE DA)

# the paper suite F1..F11
FUNCS=(Ackley Alpine1 Brown Exponential Griewank Michalewicz Qing Rastrigin Schaffer Schwefel Multiwell)

BENCH_PY="examples/benchmarking/benchmark_optimization.py"

# Output root
OUT_ROOT="bench_qtt_L${L}_b${BASE}_D${DIMS}_r${RANK}_sweeps${SWEEPS}_exps${EXPS}_seed${SEED}_z${Z}"
mkdir -p "${OUT_ROOT}"

# Repo root for PYTHONPATH (so imports like `import mtopt` work even when we cd)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Repo root: ${REPO_ROOT}"
echo "Outputs  : ${REPO_ROOT}/${OUT_ROOT}"
echo

for f in "${FUNCS[@]}"; do
  echo "============================================================"
  echo "Running: ${f}"
  echo "============================================================"

  OUT_DIR="${OUT_ROOT}/${f}"
  mkdir -p "${OUT_DIR}"

  # Run inside per-function directory so plots/logs/results don't collide
  (
    cd "${OUT_DIR}"

    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python -u "${REPO_ROOT}/${BENCH_PY}" \
      --num_dimensions "${DIMS}" \
      --functions "${f}" \
      --methods "${METHODS[@]}" \
      --grid_type qtt \
      --qtt_levels "${L}" \
      --qtt_base "${BASE}" \
      --qtt_z "${Z}" \
      --ranks "${RANK}" \
      --num_sweeps "${SWEEPS}" \
      --num_experiments "${EXPS}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}" \
      | tee "run_${f}.log"
  )

  echo
done

echo "All done. See: ${OUT_ROOT}/<Function>/results_<Function>.csv"
