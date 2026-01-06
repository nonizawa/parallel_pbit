#!/usr/bin/env bash
# Helper script to run pbit_doe.py with common presets (random / TSPLIB / G-set).
# Usage:
#   ./run_pbit_presets.sh random [extra args...]
#   ./run_pbit_presets.sh tsp [extra args...]
#   ./run_pbit_presets.sh gset [extra args...]
#   ./run_pbit_presets.sh gset-auto [extra args...]
#   ./run_pbit_presets.sh gset-sigma-sweep [extra args...]
# Environment overrides:
#   PBIT_PYTHON   - path to Python executable (default: m2max/bin/python or python3)
#   TSP_INSTANCE  - TSPLIB instance (name or path) for the tsp preset (default: berlin52)
#   GSET_INSTANCE - G-set instance (name or path) for the gset preset (default: benchmarks/gset/G14)
#   SIM_TIME_NS   - Simulation duration per run (default: 5000)
#   SAT_INSTANCE  - SATLIB CNF for sat presets (default: uf20-91/uf20-0100.cnf)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PBIT_PYTHON:-$ROOT_DIR/m2max/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

SIM_TIME_NS="${SIM_TIME_NS:-100}"
DT_NS="${DT_NS:-5}"
SAT_INSTANCE="${SAT_INSTANCE:-uf20-91/uf20-0100.cnf}"
PRESET="${1:-help}"
if [[ "$PRESET" != "help" ]]; then
  shift
fi
EXTRA_ARGS=()
EXTRA_ARGS=("$@")

PBIT_SCRIPT="$ROOT_DIR/pbit_doe.py"
if [[ ! -f "$PBIT_SCRIPT" ]]; then
  echo "pbit_doe.py not found at $PBIT_SCRIPT" >&2
  exit 1
fi

run_cmd() {
  local extra_display=""
  local log_file="results/$(date +%Y%m%d)-${PRESET}.log"
  mkdir -p results
  if ((${#EXTRA_ARGS[@]})); then
    extra_display=" ${EXTRA_ARGS[*]}"
    echo ">>> ${PYTHON_BIN} $*${extra_display}"
    "${PYTHON_BIN}" "$@" "${EXTRA_ARGS[@]}" | tee "$log_file"
  else
    echo ">>> ${PYTHON_BIN} $*"
    "${PYTHON_BIN}" "$@" | tee "$log_file"
  fi
}

case "$PRESET" in
  random)
    run_cmd "$PBIT_SCRIPT" \
      --quick \
      --sampler tick \
      --schedule const \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1
    ;;
  tsp)
    TSP_INSTANCE="${TSP_INSTANCE:-burma14}"
    run_cmd "$PBIT_SCRIPT" \
      --problem tsp \
      --tsp-instance "$TSP_INSTANCE" \
      --sampler gillespie \
      --schedule linear \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1 \
      --optimize \
      --optimizer anneal \
      --opt-iters 50 \
      --beta0-range 0.5 2.0 \
      --beta1-range 2.0 6.0
    ;;
  tsp-anneal)
    TSP_INSTANCE="${TSP_INSTANCE:-burma14}"
    run_cmd "$PBIT_SCRIPT" \
      --problem tsp \
      --tsp-instance "$TSP_INSTANCE" \
      --sampler gillespie \
      --schedule linear \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1 \
      --optimize \
      --optimizer anneal \
      --opt-iters 50 \
      --beta0-range 0.1 1.0 \
      --beta1-range 10.0 10000.0
    ;;
  tsp-evo)
    TSP_INSTANCE="${TSP_INSTANCE:-burma14}"
    run_cmd "$PBIT_SCRIPT" \
      --problem tsp \
      --tsp-instance "$TSP_INSTANCE" \
      --sampler gillespie \
      --schedule linear \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1 \
      --optimize \
      --optimizer evo \
      --opt-iters 50 \
      --evo-pop 8 \
      --beta0-range 0.5 2.0 \
      --beta1-range 2.0 6.0
    ;;
  gset-sigma-sweep)
    GSET_LIST=("G1" "G6" "G11" "G14" "G18" "G22" "G34" "G38" "G39" "G47" "G48" "G54" "G55" "G56" "G58")
    TLIST=("0.1" "0.3" "0.5" "0.75" "1" "2" "3" "5" "7" "10")
    today="$(date +%Y%m%d)"
    for inst in "${GSET_LIST[@]}"; do
      inst_path="benchmarks/gset/${inst}"
      OUT_DIR="results/${inst}"
      mkdir -p "$OUT_DIR"
      for tau in "${TLIST[@]}"; do
        log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-tau${tau}.log"
        echo ">>> [inst=${inst} tau=${tau}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
        "${PYTHON_BIN}" "$PBIT_SCRIPT" \
          --problem maxcut \
          --maxcut-instance "$inst_path" \
          --sampler tick \
          --schedule linear \
          --sim-time-ns "$SIM_TIME_NS" \
          --dt-ns "$DT_NS" \
          --repeats 10 \
          --auto-beta-fixed \
          --tau-list "$tau" \
          --bits-list 12 10 8 6 5 4 3 2 1\
          --share-list 1 \
          "$@" | tee "$log_file" &
      done
    done
    wait
    echo ">>> merging tau sweep CSVs into one file ..."
    "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
from pathlib import Path
patterns = [
    "results/*/*-doe-*gillespie_linear_tau*.csv",
    "results/*/*-doe-*tick_linear_tau*.csv",
]
files = []
for pat in patterns:
    files.extend(glob.glob(pat))
files = sorted(set(files))
if not files:
    print("No DOE CSVs found to merge.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        dfs.append(df)
    except Exception as exc:
        print(f"⚠ skip {f}: {exc}")
if not dfs:
    print("No valid CSVs to merge.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = Path("results") / "gset_sigma_sweep_all.csv"
out.to_csv(out_path, index=False)
print(f"✔ merged {len(files)} files into {out_path}")
PY
    ;;
  gset-ekbits)
    GSET_LIST=("G1" "G6" "G11" "G14" "G18" "G22" "G34" "G38" "G39" "G47")
    BITS_LIST=("12" "10" "8" "6" "4" "3" "2" "1")
    SHARE_LIST=("1" "1.1" "1.25" "1.5" "1.75" "2" "3" "4" "5" "8")
    TAU_LIST=("2.5" "5" "7.5")
    today="$(date +%Y%m%d)"
    for inst in "${GSET_LIST[@]}"; do
      inst_path="benchmarks/gset/${inst}"
      OUT_DIR="results/${inst}"
      mkdir -p "$OUT_DIR"
      for share in "${SHARE_LIST[@]}"; do
        log_file="${OUT_DIR}/${today}-${PRESET}-share${share}.log"
        echo ">>> [inst=${inst} share=${share}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
        "${PYTHON_BIN}" "$PBIT_SCRIPT" \
          --problem maxcut \
          --maxcut-instance "$inst_path" \
          --sampler tick \
          --schedule linear \
          --sim-time-ns "$SIM_TIME_NS" \
          --dt-ns "$tau" \
          --repeats 5 \
          --auto-beta-fixed \
          --tau-list "$tau" \
          --bits-list "${BITS_LIST[@]}" \
          --share-list "$share" \
          "$@" | tee "$log_file"
      done
      # Move generated DOE CSVs for this instance into its folder
      for f in results/*-doe-*${inst}*tick_linear_tau${tau}*.csv; do
        [[ -e "$f" ]] || continue
        mv "$f" "$OUT_DIR"/
      done
      echo ">>> merging ek-bits sweep CSVs for ${inst} ..."
      OUT_DIR="$OUT_DIR" "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(glob.glob(str(out_dir / "*-doe-*tick_linear_tau5*.csv")))
if not files:
    print("No DOE CSVs found to merge.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        if "E_k" not in df.columns and {"n_pbit", "dt_ns", "tau_ns", "share"}.issubset(df.columns):
            df["E_k"] = df["n_pbit"] * (df["dt_ns"] / df["tau_ns"]) / df["share"]
        dfs.append(df)
    except Exception as exc:
        print(f"skip {f}: {exc}")
if not dfs:
    print("No valid CSVs to merge.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = out_dir / "gset_ek_bits_all.csv"
out.to_csv(out_path, index=False)
print(f"merged {len(files)} files into {out_path}")

try:
    if "E_k_norm" not in out.columns and {"E_k","n_pbit"}.issubset(out.columns):
        out["E_k_norm"] = out["E_k"] / out["n_pbit"]
    heat = out.pivot_table(index="E_k_norm", columns="bits", values="energy", aggfunc="mean")
    heat = heat.sort_index()
    tau_vals = out["tau_ns"].dropna().unique()
    tau_str = ", ".join(sorted(str(t) for t in tau_vals)) if tau_vals.size else "?"
    plt.figure(figsize=(6,4))
    im = plt.imshow(heat.values, aspect="auto", origin="lower",
                    extent=[heat.columns.min()-0.5, heat.columns.max()+0.5,
                            heat.index.min(), heat.index.max()],
                    cmap="viridis")
    plt.colorbar(im, label="energy (mean)")
    plt.xlabel("bits")
    plt.ylabel("E[k]/n")
    plt.title(f"{out_dir.name} tick sweep (tau={tau_str})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{out_dir.name}_ek_bits_heatmap.png", dpi=200)
    plt.close()
    print(f"heatmap saved to {out_dir / (out_dir.name + '_ek_bits_heatmap.png')}")

    plt.figure(figsize=(6,4))
    for ek_val in sorted(out["E_k_norm"].dropna().unique()):
        subset = out[out["E_k_norm"] == ek_val]
        grp = subset.groupby("bits")["energy"].mean().sort_index()
        plt.plot(grp.index, grp.values, marker="o", label=f"E[k]/n={ek_val:.2f}")
    plt.xlabel("Bit width")
    plt.ylabel("energy (mean)")
    plt.title(f"{out_dir.name} tick sweep lines (tau={tau_str})")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / f"{out_dir.name}_ek_bits_lines.png", dpi=200)
    plt.close()
    print(f"line plot saved to {out_dir / (out_dir.name + '_ek_bits_lines.png')}")

    if "maxcut_cut_value" in out.columns:
        cut_heat = out.pivot_table(index="E_k_norm", columns="bits", values="maxcut_cut_value", aggfunc="mean")
        cut_heat = cut_heat.sort_index()
        plt.figure(figsize=(6,4))
        im = plt.imshow(cut_heat.values, aspect="auto", origin="lower",
                        extent=[cut_heat.columns.min()-0.5, cut_heat.columns.max()+0.5,
                                cut_heat.index.min(), cut_heat.index.max()],
                        cmap="magma")
        plt.colorbar(im, label="maxcut (mean)")
        plt.xlabel("bits")
        plt.ylabel("E[k]/n")
        plt.title(f"{out_dir.name} MaxCut (tau={tau_str})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{out_dir.name}_ek_bits_cut_heatmap.png", dpi=200)
        plt.close()
        print(f"MaxCut heatmap saved to {out_dir / (out_dir.name + '_ek_bits_cut_heatmap.png')}")
except Exception as exc:
    print(f"merge/plot failed: {exc}")
PY
    done
    ;;
  gset-sync-vs-async)
    GSET_LIST=("G1" "G6" "G11" "G14" "G18")
    BITS_LIST=("12" "10" "8" "6" "4" "2")
    SHARE_LIST=("1" "1.25" "1.5" "2" "3")
    TAU_LIST=("2.5" "5" "7.5")
    today="$(date +%Y%m%d)"
    for inst in "${GSET_LIST[@]}"; do
      (
        inst_path="benchmarks/gset/${inst}"
        OUT_DIR="results/${inst}/sync_async"
        mkdir -p "$OUT_DIR"
        for sampler in tick gillespie; do
          log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-${sampler}.log"
          echo ">>> [inst=${inst} sampler=${sampler}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
          "${PYTHON_BIN}" "$PBIT_SCRIPT" \
            --problem maxcut \
            --maxcut-instance "$inst_path" \
            --sampler "$sampler" \
            --schedule linear \
            --sim-time-ns "$SIM_TIME_NS" \
            --dt-ns "$tau" \
            --repeats 10 \
            --auto-beta-fixed \
            --tau-list "$tau" \
            --bits-list "${BITS_LIST[@]}" \
            --share-list "${SHARE_LIST[@]}" \
            --tick-mode "random" \
            --apply-delay-ns 0 \
            "$@" | tee "$log_file"
        done
        # move CSVs for this inst into OUT_DIR
        for f in results/*-doe-*${inst}*${tau}*.csv; do
          [[ -e "$f" ]] || continue
          mv "$f" "$OUT_DIR"/
        done
        echo ">>> merging sync vs async for ${inst} ..."
        OUT_DIR="$OUT_DIR" "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(glob.glob(str(out_dir / "*-doe-*tick*.csv")) + glob.glob(str(out_dir / "*-doe-*gillespie*.csv")))
if not files:
    print("No DOE CSVs found to merge.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        if "E_k" not in df.columns and {"n_pbit","dt_ns","tau_ns","share"}.issubset(df.columns):
            df["E_k"] = df["n_pbit"] * (df["dt_ns"] / df["tau_ns"]) / df["share"]
        if "E_k_norm" not in df.columns and {"E_k","n_pbit"}.issubset(df.columns):
            df["E_k_norm"] = df["E_k"] / df["n_pbit"]
        dfs.append(df)
    except Exception as exc:
        print(f"skip {f}: {exc}")
if not dfs:
    print("No valid CSVs to merge.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = out_dir / "gset_sync_async_all.csv"
out.to_csv(out_path, index=False)
print(f"merged {len(files)} files into {out_path}")

try:
    # energy heatmap per sampler
    for sampler in out["sampler"].unique():
        sub = out[out["sampler"] == sampler].copy()
        if sampler == "gillespie":
            sub["E_k"] = np.nan
            sub["E_k_norm"] = np.nan
        if "E_k_norm" not in sub.columns and {"E_k","n_pbit"}.issubset(sub.columns):
            sub["E_k_norm"] = sub["E_k"] / sub["n_pbit"]
        if sub.empty or "E_k_norm" not in sub.columns:
            continue
        heat = sub.pivot_table(index="E_k_norm", columns="bits", values="energy", aggfunc="mean")
        heat = heat.sort_index()
        plt.figure(figsize=(6,4))
        im = plt.imshow(heat.values, aspect="auto", origin="lower",
                        extent=[heat.columns.min()-0.5, heat.columns.max()+0.5,
                                heat.index.min(), heat.index.max()],
                        cmap="viridis")
        plt.colorbar(im, label="energy (mean)")
        plt.xlabel("bits")
        plt.ylabel("E[k]/n")
        tau_vals = sub["tau_ns"].dropna().unique()
        tau_str = ", ".join(sorted(str(t) for t in tau_vals)) if tau_vals.size else "?"
        plt.title(f"{out_dir.name} {sampler} (tau={tau_str})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{out_dir.name}_{sampler}_heatmap.png", dpi=200)
        plt.close()
        print(f"heatmap saved to {out_dir / (out_dir.name + '_' + sampler + '_heatmap.png')}")

    # MaxCut heatmap per sampler
    if "maxcut_cut_value" in out.columns:
        for sampler in out["sampler"].unique():
            sub = out[out["sampler"] == sampler].copy()
            if sampler == "gillespie":
                sub["E_k"] = np.nan
                sub["E_k_norm"] = np.nan
            if "E_k_norm" not in sub.columns and {"E_k","n_pbit"}.issubset(sub.columns):
                sub["E_k_norm"] = sub["E_k"] / sub["n_pbit"]
            if sub.empty or "E_k_norm" not in sub.columns:
                continue
            cut_heat = sub.pivot_table(index="E_k_norm", columns="bits", values="maxcut_cut_value", aggfunc="mean")
            cut_heat = cut_heat.sort_index()
            plt.figure(figsize=(6,4))
            im = plt.imshow(cut_heat.values, aspect="auto", origin="lower",
                            extent=[cut_heat.columns.min()-0.5, cut_heat.columns.max()+0.5,
                                    cut_heat.index.min(), cut_heat.index.max()],
                            cmap="magma")
            plt.colorbar(im, label="maxcut (mean)")
            plt.xlabel("bits")
            plt.ylabel("E[k]/n")
            tau_vals = sub["tau_ns"].dropna().unique()
            tau_str = ", ".join(sorted(str(t) for t in tau_vals)) if tau_vals.size else "?"
            plt.title(f"{out_dir.name} {sampler} MaxCut (tau={tau_str})")
            plt.tight_layout()
            plt.savefig(out_dir / f"{out_dir.name}_{sampler}_cut_heatmap.png", dpi=200)
            plt.close()
            print(f"MaxCut heatmap saved to {out_dir / (out_dir.name + '_' + sampler + '_cut_heatmap.png')}")
except Exception as exc:
    print(f"merge/plot failed: {exc}")
PY
      ) &
    done
    wait
    ;;
  gset-sync-rr)
    GSET_LIST=("G1" "G6" "G11" "G14" "G18"  "G22" "G34" "G38" "G39" "G47")
    BITS_LIST=("12" "10" "8" "6" "4" "3" "2" "1")
    SHARE_LIST=("1" "1.25" "1.5" "2" "3" "5" "10" "15" "20" "30" "50" "100")
    GILL_SHARE_LIST=("1")
    tau_tick="5"
    TAU_GILL_LIST=("2.5" "5" "7.5" "10" "15")
    today="$(date +%Y%m%d)"
    for inst in "${GSET_LIST[@]}"; do
      (
        inst_path="benchmarks/gset/${inst}"
        OUT_DIR="results/sync_rr/${inst}"
        mkdir -p "$OUT_DIR"
        # gillespie (apply_delay=5, tau sweep)
        for tau_gill in "${TAU_GILL_LIST[@]}"; do
          log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-gillespie-tau${tau_gill}.log"
          echo ">>> [inst=${inst} sampler=gillespie tau=${tau_gill}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
          "${PYTHON_BIN}" "$PBIT_SCRIPT" \
            --problem maxcut \
            --maxcut-instance "$inst_path" \
            --sampler gillespie \
            --schedule linear \
            --sim-time-ns "$SIM_TIME_NS" \
            --dt-ns "$tau_gill" \
            --repeats 10 \
            --auto-beta-fixed \
            --tau-list "$tau_gill" \
            --bits-list "${BITS_LIST[@]}" \
            --share-list "${GILL_SHARE_LIST[@]}" \
            --apply-delay-ns 5 \
            --problem-label "${inst}_gill_tau${tau_gill}" \
            "$@" | tee "$log_file"
          # collect output CSVs (some runs encode tau as "2_5" instead of "2.5")
          tau_pat="${tau_gill//./_}"
          for f in results/*-doe-*${inst}_gill*${tau_gill}*.csv results/*-doe-*${inst}_gill*${tau_pat}*.csv results/*-doe-*${inst}_gillespie*${tau_pat}*.csv; do
            [[ -e "$f" ]] || continue
            mv "$f" "$OUT_DIR"/
          done
        done
        # tick modes: random, block-random, block-random-stride (apply_delay=5, tau=5)
        for mode in random block-random block-random-stride; do
          log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-tick-${mode}.log"
          echo ">>> [inst=${inst} sampler=tick mode=${mode}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
          "${PYTHON_BIN}" "$PBIT_SCRIPT" \
            --problem maxcut \
            --maxcut-instance "$inst_path" \
            --sampler tick \
            --schedule linear \
            --sim-time-ns "$SIM_TIME_NS" \
            --dt-ns "$tau_tick" \
            --repeats 10 \
            --auto-beta-fixed \
            --tau-list "$tau_tick" \
            --bits-list "${BITS_LIST[@]}" \
            --share-list "${SHARE_LIST[@]}" \
            --tick-mode "$mode" \
            --apply-delay-ns 5 \
            --problem-label "${inst}_tick_${mode}" \
            "$@" | tee "$log_file"
          for f in results/*-doe-*${inst}_tick_${mode}*${tau_tick}*.csv; do
            [[ -e "$f" ]] || continue
            mv "$f" "$OUT_DIR"/
          done
        done
        echo ">>> merging sync-rr for ${inst} ..."
        OUT_DIR="$OUT_DIR" "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(glob.glob(str(out_dir / "*-doe-*tick*.csv")) + glob.glob(str(out_dir / "*-doe-*gillespie*.csv")))
if not files:
    print("No DOE CSVs found to merge.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        if "E_k" not in df.columns and {"n_pbit","dt_ns","tau_ns","share"}.issubset(df.columns):
            df["E_k"] = df["n_pbit"] * (df["dt_ns"] / df["tau_ns"]) / df["share"]
        if "E_k_norm" not in df.columns and {"E_k","n_pbit"}.issubset(df.columns):
            df["E_k_norm"] = df["E_k"] / df["n_pbit"]
        # capture tick mode from column (if present) or filename/label
        if "tick" in df["sampler"].unique():
            if "tick_mode" in df.columns and df["tick_mode"].notna().nunique() == 1:
                mode = df["tick_mode"].dropna().iloc[0]
            else:
                mode = "random"
                name = Path(f).name
                label_vals = df["label"].unique() if "label" in df.columns else []
                text = name + " ".join(map(str,label_vals))
                if any(s in text for s in ["block_random_stride", "block-random-stride", "block-stride"]):
                    mode = "block-random-stride"
                elif any(s in text for s in ["block_random", "block-random", "banked"]):
                    mode = "block-random"
                elif any(s in text for s in ["tick_rr_jitter", "tick-rr-jitter", "rr-jitter", "rr_jitter"]):
                    mode = "rr-jitter"
                elif "tick_rr" in text or "tick-rr" in text:
                    mode = "rr"
                elif "tick_random" in text or "tick-random" in text:
                    mode = "random"
            df["tick_mode"] = mode
        dfs.append(df)
    except Exception as exc:
        print(f"skip {f}: {exc}")
if not dfs:
    print("No valid CSVs to merge.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = out_dir / "gset_sync_rr_all.csv"
out.to_csv(out_path, index=False)
print(f"merged {len(files)} files into {out_path}")

try:
    # energy heatmap per sampler/mode
    for sampler in out["sampler"].unique():
        sub_all = out[out["sampler"] == sampler].copy()
        modes = sub_all["tick_mode"].unique() if "tick_mode" in sub_all.columns else [None]
        for mode in modes:
            sub = sub_all if mode is None else sub_all[sub_all["tick_mode"] == mode]
            if "E_k_norm" not in sub.columns and {"E_k","n_pbit"}.issubset(sub.columns):
                sub["E_k_norm"] = sub["E_k"] / sub["n_pbit"]
            if sub.empty or "E_k_norm" not in sub.columns:
                continue
            heat = sub.pivot_table(index="E_k_norm", columns="bits", values="energy", aggfunc="mean")
            heat = heat.sort_index()
            plt.figure(figsize=(6,4))
            im = plt.imshow(heat.values, aspect="auto", origin="lower",
                            extent=[heat.columns.min()-0.5, heat.columns.max()+0.5,
                                    heat.index.min(), heat.index.max()],
                            cmap="viridis")
            plt.colorbar(im, label="energy (mean)")
            plt.xlabel("bits")
            plt.ylabel("E[k]/n")
            tau_vals = sub["tau_ns"].dropna().unique()
            tau_str = ", ".join(sorted(str(t) for t in tau_vals)) if tau_vals.size else "?"
            title_mode = "" if mode is None else f" ({mode})"
            plt.title(f"{out_dir.parent.name} {sampler}{title_mode} (tau={tau_str})")
            plt.tight_layout()
            suffix = sampler if mode is None else f"{sampler}_{mode}"
            plt.savefig(out_dir / f"{out_dir.parent.name}_{suffix}_heatmap.png", dpi=200)
            plt.close()
            print(f"heatmap saved to {out_dir / (out_dir.parent.name + '_' + suffix + '_heatmap.png')}")
except Exception as exc:
    print(f"merge/plot failed: {exc}")
PY
      ) &
    done
    wait
    ;;
  gset-simtime-bits)
    # Sweep sim_time_ns vs bits to test if longer runs can offset low bit-width
    GSET_LIST=("G1" "G6" "G11" "G14" "G18" "G22" "G34" "G38" "G39" "G47")
    BITS_LIST=("1" "2" "3" "4" "6" "8" "10" "12")
    SIM_TIMES=("100" "200" "500" "1000" "2000")
    SHARE_LIST=("1" "1.25" "1.5" "2" "3" "5" "10")
    GILL_SHARE_LIST=("1")
    TAU_LIST=("2.5" "5" "7.5")
    today="$(date +%Y%m%d)"
    for inst in "${GSET_LIST[@]}"; do
      (
        inst_path="benchmarks/gset/${inst}"
        OUT_DIR="results/simtime-bits/${inst}"
        mkdir -p "$OUT_DIR"
        for simt in "${SIM_TIMES[@]}"; do
          # Gillespie: sweep tau
          for tau in "${TAU_LIST[@]}"; do
            tau_tag="${tau//./_}"
            sampler=gillespie
            log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-${sampler}-sim${simt}.log"
            echo ">>> [inst=${inst} sampler=${sampler} sim_time_ns=${simt} tau=${tau}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
            cmd=( "$PYTHON_BIN" "$PBIT_SCRIPT"
              --problem maxcut
              --maxcut-instance "$inst_path"
              --sampler "$sampler"
              --schedule linear
              --sim-time-ns "$simt"
              --dt-ns "$tau"
              --repeats 10
              --auto-beta-fixed
              --tau-list "$tau"
              --bits-list "${BITS_LIST[@]}"
              --share-list "${GILL_SHARE_LIST[@]}"
              --apply-delay-ns 5
              --problem-label "${inst}_simt${simt}_tau${tau}_${sampler}"
              "$@"
            )
            "${cmd[@]}" | tee "$log_file"
            for f in \
              results/*-doe-*${inst}_simt${simt}_tau${tau_tag}_${sampler}*${tau}*.csv \
              results/*-doe-*${inst}_simt${simt}_tau${tau}_${sampler}*${tau}*.csv; do
              [[ -e "$f" ]] || continue
              mv "$f" "$OUT_DIR"/
            done
          done

          # Tick: fixed tau=5 only
          tau=5
          tau_tag="${tau//./_}"
          for tick_mode in random block-random block-random-stride; do
            sampler=tick
            log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-${sampler}-${tick_mode}-sim${simt}.log"
            echo ">>> [inst=${inst} sampler=${sampler} mode=${tick_mode} sim_time_ns=${simt} tau=${tau}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
            cmd=( "$PYTHON_BIN" "$PBIT_SCRIPT"
              --problem maxcut
              --maxcut-instance "$inst_path"
              --sampler "$sampler"
              --schedule linear
              --sim-time-ns "$simt"
              --dt-ns "$tau"
              --repeats 10
              --auto-beta-fixed
              --tau-list "$tau"
              --bits-list "${BITS_LIST[@]}"
              --share-list "${SHARE_LIST[@]}"
              --tick-mode "$tick_mode"
              --apply-delay-ns 0
              --problem-label "${inst}_simt${simt}_tau${tau}_${sampler}_${tick_mode}"
              "$@"
            )
            "${cmd[@]}" | tee "$log_file"
            for f in \
              results/*-doe-*${inst}_simt${simt}_tau${tau_tag}_${sampler}_${tick_mode}*${tau}*.csv \
              results/*-doe-*${inst}_simt${simt}_tau${tau}_${sampler}_${tick_mode}*${tau}*.csv; do
              [[ -e "$f" ]] || continue
              mv "$f" "$OUT_DIR"/
            done
          done
        done
        echo ">>> merging simtime-bits for ${inst} ..."
        OUT_DIR="$OUT_DIR" "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(glob.glob(str(out_dir / "*-doe-*.csv")))
if not files:
    print("No CSVs to merge.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        if "E_k" not in df.columns and {"n_pbit","dt_ns","tau_ns","share"}.issubset(df.columns):
            df["E_k"] = df["n_pbit"] * (df["dt_ns"] / df["tau_ns"]) / df["share"]
        if "E_k_norm" not in df.columns and {"E_k","n_pbit"}.issubset(df.columns):
            df["E_k_norm"] = df["E_k"] / df["n_pbit"]
        dfs.append(df)
    except Exception as exc:
        print(f"skip {f}: {exc}")
if not dfs:
    print("No valid CSVs.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = out_dir / "gset_simtime_bits_all.csv"
out.to_csv(out_path, index=False)
print(f"merged {len(files)} files into {out_path}")
PY
      ) &
    done
    wait
    ;;
  gset-gillespie-tau-delay)
    # Gillespie with fixed apply_delay_ns=5, sweeping tau
    GSET_LIST=("G1" "G6" "G11" "G14" "G18" "G22" "G34" "G38" "G39" "G47")
    BITS_LIST=("1" "2" "3" "4" "6" "8" "10" "12")
    SHARE_LIST=("1")
    TAU_LIST=("2.5" "5" "7.5" "10" "15")
    SIM_TIME_NS="${SIM_TIME_NS:-100}"
    today="$(date +%Y%m%d)"
    all_done=true
    for inst in "${GSET_LIST[@]}"; do
      if [[ ! -f "results/gillespie-tau-delay/${inst}/gillespie_tau_delay_all.csv" ]]; then
        all_done=false
        break
      fi
    done
    if [[ "$all_done" != true ]]; then
      for inst in "${GSET_LIST[@]}"; do
        (
          inst_path="benchmarks/gset/${inst}"
          OUT_DIR="results/gillespie-tau-delay/${inst}"
          mkdir -p "$OUT_DIR"
          log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-gillespie.log"
          echo ">>> [inst=${inst} sampler=gillespie tau sweep apply_delay=5] ${PYTHON_BIN} $PBIT_SCRIPT ..."
          "${PYTHON_BIN}" "$PBIT_SCRIPT" \
            --problem maxcut \
            --maxcut-instance "$inst_path" \
            --sampler gillespie \
            --schedule linear \
            --sim-time-ns "$SIM_TIME_NS" \
            --dt-ns 5 \
            --repeats 10 \
            --auto-beta-fixed \
            --tau-list "${TAU_LIST[@]}" \
            --bits-list "${BITS_LIST[@]}" \
            --share-list "${SHARE_LIST[@]}" \
            --apply-delay-ns 5 \
            --problem-label "${inst}_gill_delay5" \
            "$@" | tee "$log_file"
          for f in results/*-doe-*${inst}_gill_delay5*.csv; do
            [[ -e "$f" ]] || continue
            mv "$f" "$OUT_DIR"/
          done
          echo ">>> merging gillespie tau-delay for ${inst} ..."
          OUT_DIR="$OUT_DIR" "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(glob.glob(str(out_dir / "*-doe-*.csv")))
if not files:
    print("No CSVs to merge.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        if "E_k" not in df.columns and {"n_pbit","dt_ns","tau_ns","share"}.issubset(df.columns):
            df["E_k"] = df["n_pbit"] * (df["dt_ns"] / df["tau_ns"]) / df["share"]
        if "E_k_norm" not in df.columns and {"E_k","n_pbit"}.issubset(df.columns):
            df["E_k_norm"] = df["E_k"] / df["n_pbit"]
        dfs.append(df)
    except Exception as exc:
        print(f"skip {f}: {exc}")
if not dfs:
    print("No valid CSVs.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = out_dir / "gillespie_tau_delay_all.csv"
out.to_csv(out_path, index=False)
print(f"merged {len(files)} files into {out_path}")
PY
        ) &
      done
      wait
    else
      echo ">>> existing gillespie tau-delay results found; skipping simulation."
    fi
    echo ">>> plotting gillespie tau-delay grid ..."
    MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp XDG_CONFIG_HOME=/tmp "${PYTHON_BIN}" - <<'PY'
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

GSET_TARGETS = {
    "G1": 11624, "G6": 2178, "G11": 564, "G34": 1384, "G38": 7688, "G39": 2408,
}
graphs = ["G1","G6","G11","G34","G38","G39"]
root = Path("results/gillespie-tau-delay")
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=False)
bits_order = [1,2,3,4,6,8,10,12]
colors = plt.cm.viridis_r([i/len(bits_order) for i in range(len(bits_order))])
d_ns = 5.0

for ax, graph in zip(axes.flat, graphs):
    f = root / graph / "gillespie_tau_delay_all.csv"
    if not f.exists():
        ax.set_title(f"{graph} (missing)")
        continue
    df = pd.read_csv(f).sort_values("tau_ns")
    target = GSET_TARGETS.get(graph)
    for b, c in zip(bits_order, colors):
        sub = df[df["bits"] == b]
        if sub.empty:
            continue
        x = d_ns / sub["tau_ns"]
        y = sub["maxcut_cut_value"] / target if target else sub["maxcut_cut_value"]
        ax.plot(x, y, "-o", label=f"{b} bits", color=c, markersize=3)
    ax.set_title(graph)
    ax.set_xlabel(r"$d/\tau$")
    ax.set_ylabel("Normalized mean cut value")
    ax.grid(True, alpha=0.3)
axes[0,0].legend(fontsize=8, ncol=2, frameon=False)
plt.tight_layout()
out = Path("latex/figs/cut_vs_tau_grid_gillespie_delay5.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
print(f"saved {out}")
PY
    ;;
  gset-tick-share-sweep)
    # Sweep share for tick (random) to evaluate performance vs. time-multiplexing
    GSET_LIST=("G1" "G6" "G11" "G14" "G18" "G22" "G34" "G38" "G39" "G47")
    BITS_LIST=("1" "2" "3" "4" "6" "8" "10" "12")
    SHARE_LIST=("1" "1.25" "1.5" "2" "3" "5" "10")
    tau="5"
    SIM_TIME_NS="${SIM_TIME_NS:-1000}"
    today="$(date +%Y%m%d)"
    for inst in "${GSET_LIST[@]}"; do
      (
        inst_path="benchmarks/gset/${inst}"
        OUT_DIR="results/tick-share/${inst}"
        mkdir -p "$OUT_DIR"
        for mode in random block-random block-random-stride; do
          log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-tick-${mode}.log"
          echo ">>> [inst=${inst} sampler=tick mode=${mode} share sweep] ${PYTHON_BIN} $PBIT_SCRIPT ..."
          "${PYTHON_BIN}" "$PBIT_SCRIPT" \
            --problem maxcut \
            --maxcut-instance "$inst_path" \
            --sampler tick \
            --tick-mode "$mode" \
            --schedule linear \
            --sim-time-ns "$SIM_TIME_NS" \
            --dt-ns "$tau" \
            --repeats 10 \
            --auto-beta-fixed \
            --tau-list "$tau" \
            --bits-list "${BITS_LIST[@]}" \
            --share-list "${SHARE_LIST[@]}" \
            --apply-delay-ns 0 \
            --problem-label "${inst}_tick_share_${mode}" \
            "$@" | tee "$log_file"
          for f in results/*-doe-*${inst}_tick_share_${mode}*${tau}*.csv; do
            [[ -e "$f" ]] || continue
            mv "$f" "$OUT_DIR"/
          done
        done
        echo ">>> merging tick-share for ${inst} ..."
        OUT_DIR="$OUT_DIR" "${PYTHON_BIN}" - <<'PY'
import glob
import pandas as pd
from pathlib import Path
import os

out_dir = Path(os.environ["OUT_DIR"])
files = sorted(glob.glob(str(out_dir / "*-doe-*.csv")))
if not files:
    print("No DOE CSVs found.")
    raise SystemExit(0)
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name
        if "E_k" not in df.columns and {"n_pbit","dt_ns","tau_ns","share"}.issubset(df.columns):
            df["E_k"] = df["n_pbit"] * (df["dt_ns"] / df["tau_ns"]) / df["share"]
        if "E_k_norm" not in df.columns and {"E_k","n_pbit"}.issubset(df.columns):
            df["E_k_norm"] = df["E_k"] / df["n_pbit"]
        dfs.append(df)
    except Exception as exc:
        print(f"skip {f}: {exc}")
if not dfs:
    print("No valid CSVs.")
    raise SystemExit(0)
out = pd.concat(dfs, ignore_index=True)
out_path = out_dir / "gset_tick_share_all.csv"
out.to_csv(out_path, index=False)
print(f"merged {len(files)} files into {out_path}")
PY
      ) &
    done
    wait
    ;;
  gset-oscillation)
    # Fig.2: synchronous oscillation across six graphs
    GSET_LIST=("G1" "G6" "G11" "G34" "G38" "G39")
    today="$(date +%Y%m%d)"
    tau="5"
    bits="12"
    simt="${SIM_TIME_NS:-1000}"
    share_list=(1 1.25 1.5 2 3)
    for inst in "${GSET_LIST[@]}"; do
      (
        inst_path="benchmarks/gset/${inst}"
        OUT_DIR="results/oscillation/${inst}"
        mkdir -p "$OUT_DIR"
        for share in "${share_list[@]}"; do
          log_file="${OUT_DIR}/${today}-${PRESET}-${inst}-share${share}.log"
          echo ">>> [inst=${inst} tick random share=${share}] ${PYTHON_BIN} $PBIT_SCRIPT ..."
          "${PYTHON_BIN}" "$PBIT_SCRIPT" \
            --problem maxcut \
            --maxcut-instance "$inst_path" \
            --sampler tick \
            --schedule linear \
            --sim-time-ns "$simt" \
            --dt-ns "$tau" \
            --repeats 5 \
            --auto-beta-fixed \
            --tau-list "$tau" \
            --bits-list "$bits" \
            --share-list "$share" \
            --tick-mode random \
            --apply-delay-ns 5 \
            --dump-trace \
            --trace-dir "$OUT_DIR" \
            --problem-label "${inst}_osc_s${share}" \
            "$@" | tee "$log_file"
          share_tag="${share/./_}"
          for f in results/*-doe-*${inst}_osc_s${share_tag}*.csv; do
            [[ -e "$f" ]] || continue
            mv "$f" "$OUT_DIR"/
          done
        done
      ) &
    done
    wait
    echo ">>> plotting oscillation grid ..."
    MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp XDG_CONFIG_HOME=/tmp "${PYTHON_BIN}" "latex/make_oscillation_grid_fig.py" \
      --roots results/oscillation \
      --graphs "${GSET_LIST[@]}" \
      --out "latex/figs/fig2_oscillation.png"
    ;;
  make-landscape)
    # Fig.1: unified performance--cost landscape
    "${PYTHON_BIN}" "latex/make_landscape_fig.py" \
      --inputs results/sync_rr \
      --metric maxcut_cut_value \
      --use-gset-target \
      --normalize-cost \
      --out "latex/figs/fig1_landscape.png"
    ;;
  make-policy-fig)
    # Fig.4: policy comparison (sync modes)
    MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp XDG_CONFIG_HOME=/tmp "${PYTHON_BIN}" "latex/make_policy_fig.py" \
      --inputs results/sync_rr \
      --out "latex/figs/fig4_policy_compare.png"
    ;;
  make-simtime-fig)
    # Fig.5: bit-width vs sim_time trade-off
    MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp XDG_CONFIG_HOME=/tmp "${PYTHON_BIN}" "latex/make_simtime_grid_fig.py" \
      --inputs results/simtime-bits \
      --gill-taus 2.5 5 7.5 \
      --tick-tau 5 \
      --share 3 \
      --out "latex/figs/fig5_simtime_bits_grid.png"
    ;;
  sat)
    run_cmd "$PBIT_SCRIPT" \
      --problem sat \
      --sat-instance "$SAT_INSTANCE" \
      --sampler gillespie \
      --schedule linear \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1 \
      --optimize \
      --opt-iters 50 \
      --sat-clause-penalty 5 \
      --sat-aux-penalty 10 \
      --beta0-range 0.1 1.0 \
      --beta1-range 1.0 6.0
    ;;
  sat-anneal)
    run_cmd "$PBIT_SCRIPT" \
      --problem sat \
      --sat-instance "$SAT_INSTANCE" \
      --sampler gillespie \
      --schedule linear \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1 \
      --optimize \
      --optimizer anneal \
      --opt-iters 50 \
      --sat-clause-penalty 5 \
      --sat-aux-penalty 10 \
      --beta0-range 0.1 1.0 \
      --beta1-range 1.0 6.0
    ;;
  sat-evo)
    run_cmd "$PBIT_SCRIPT" \
      --problem sat \
      --sat-instance "$SAT_INSTANCE" \
      --sampler gillespie \
      --schedule linear \
      --sim-time-ns "$SIM_TIME_NS" \
      --dt-ns "$DT_NS" \
      --repeats 1 \
      --optimize \
      --optimizer evo \
      --opt-iters 50 \
      --evo-pop 8 \
      --sat-clause-penalty 5 \
      --sat-aux-penalty 10 \
      --beta0-range 0.1 1.0 \
      --beta1-range 1.0 6.0
    ;;
  help|*)
    cat <<'EOF'
Usage: ./run_pbit_presets.sh <preset> [extra pbit_doe.py args...]

Presets:
  random       - quick sanity sweep on random Ising instances
  tsp          - TSPLIB run (default instance: berlin52; override via TSP_INSTANCE)
  tsp-anneal   - TSPLIB run with simulated annealing
  tsp-evo      - TSPLIB run with evolutionary optimization
  gset         - G-set Max-Cut run with random-search optimizer (default: benchmarks/gset/G14)
  sat          - SATLIB CNF run with random-search optimizer
  sat-anneal   - SATLIB run with simulated annealing optimizer
  sat-evo      - SATLIB run with an evolutionary optimizer

Examples:
  ./run_pbit_presets.sh random --repeats 5
  TSP_INSTANCE=benchmarks/tsplib/burma14.tsp ./run_pbit_presets.sh tsp --optimize --opt-iters 10
  GSET_INSTANCE=benchmarks/gset/G14 ./run_pbit_presets.sh gset --sampler gillespie
EOF
    ;;
esac
