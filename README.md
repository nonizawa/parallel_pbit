# pbit DOE runner

Design-of-experiments (DOE) and parameter optimization for p-bit based Ising/QUBO simulations.
The main entrypoint is `pbit_doe.py`.

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

## Examples

Random problem sweep (custom bits/tau):

```bash
python3 pbit_doe.py --sampler tick --schedule linear --bits-list 4 6 8 --tau-list 0.5 1 5 --repeats 3
```

G-set MaxCut (auto-download/cache):

```bash
python3 pbit_doe.py --problem maxcut --maxcut-instance G14 --sampler tick --schedule linear --auto-beta-fixed
```

## Outputs

- DOE CSVs and plots are written under `results/`.
- Energy trace CSVs and plots are written under `energy_logs/` when `--dump-trace` is enabled.
- Benchmarks are cached under `benchmarks/`.

## Notes

- Network access is required to download TSPLIB / G-set / SATLIB instances.
- Use `--download-only` to fetch and cache a benchmark without running DOE.
- `--optimize` enables random search / simulated annealing / evolutionary optimization.

## License

This project is licensed under the MIT License. See `LICENSE`.
