# pbit DOE runner

Design-of-experiments (DOE) and parameter optimization for p-bit based Ising/QUBO simulations.
The main entrypoint is `pbit_doe.py` and there is a convenience script `run_pbit_presets.sh`.

## Requirements

- Python 3.9+ recommended
- Dependencies:
  - numpy
  - pandas
  - matplotlib
  - certifi (optional, for HTTPS certificate bundle)

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run a small random DOE sweep:

```bash
python3 pbit_doe.py --quick --sampler tick --schedule const --sim-time-ns 100 --dt-ns 5 --repeats 1
```

Or use presets:

```bash
./run_pbit_presets.sh random
./run_pbit_presets.sh tsp
./run_pbit_presets.sh gset-sigma-sweep
```

## Usage

Show full CLI help:

```bash
python3 pbit_doe.py --help
```

The script supports two main modes:

- DOE sweep mode (default): sweeps over `bits`, `share`, `tau` and records metrics.
- Optimization mode (`--optimize`): random search / simulated annealing / evolutionary search over the same space.

## Modes and concepts

### Samplers

- `gillespie`: event-driven asynchronous update (default).
- `tick`: discrete time step updates using `dt-ns`, with selectable update scheduling via `--tick-mode`.

Tick scheduling modes:

- `random`: independent random flips (default)
- `rr`: round-robin selection
- `rr-jitter`: round-robin with random starting offset per tick
- `block-random`: contiguous block selection
- `block-random-stride`: contiguous generator with random stride

### Schedules

- `const`: constant beta
- `exp`: exponential schedule
- `linear`: linear schedule
- `step`: two-level step schedule

The schedule is controlled by `--beta0`, `--beta1`, and `--steps` (when applicable).

### Problems

- `random`: random Ising problem (default)
- `maxcut`: G-set instances (auto-download/cache)

For non-random problems, use the corresponding `--*-instance` flag:

```bash
python3 pbit_doe.py --problem maxcut --maxcut-instance G14
```

### Quantization

By default, input currents are quantized by `--bits-list` and the `--Imin/--Imax` range.
Disable quantization with `--no-quantize`.

### Auto beta helpers

- `--auto-beta-range`: infer reasonable beta ranges from problem statistics (for optimization).
- `--auto-beta-fixed`: set beta0/beta1 using a sigma-based heuristic.

## Output files

- DOE CSVs and summary plots are written under `results/`.
- Energy trace CSVs and plots are written under `energy_logs/` when `--dump-trace` is enabled.
- Benchmarks are cached under `benchmarks/`.

Each DOE run writes a CSV with metadata columns (sampler, schedule, tau, bits, share, etc.).
Optimization runs also write a CSV log and a summary plot.

## Examples

Random problem sweep (custom bits/tau):

```bash
python3 pbit_doe.py --sampler tick --schedule linear --bits-list 4 6 8 --tau-list 0.5 1 5 --repeats 3
```

Tick sampler with round-robin scheduling:

```bash
python3 pbit_doe.py --sampler tick --tick-mode rr --schedule const --sim-time-ns 200 --dt-ns 2
```

Energy trace dump:

```bash
python3 pbit_doe.py --sampler tick --schedule const --dump-trace --trace-dir results/traces
```

Download benchmark only (no run):

```bash
python3 pbit_doe.py --problem maxcut --maxcut-instance G14 --download-only
```

## Preset runner

`run_pbit_presets.sh` wraps common invocations and writes log files into `results/`.
Environment overrides:

- `PBIT_PYTHON`: path to Python executable
- `GSET_INSTANCE`: G-set instance for `gset` presets
- `SIM_TIME_NS`: simulation duration per run
- `DT_NS`: tick time step

## Notes

- Network access is required to download G-set instances.
- Use `--download-only` to fetch and cache a benchmark without running DOE.
- `--optimize` enables random search / simulated annealing / evolutionary optimization.

## License

This project is licensed under the MIT License. See `LICENSE`.
