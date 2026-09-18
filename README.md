# Risk-Aware GraphOPF

Research code for evaluating learned AC-OPF surrogates under correlated, heavy-tailed renewable uncertainty.

## M0 objective

Before training any GNN, M0 tests whether realistic uncertainty changes operational risk:

1. solve a PGLib AC-OPF case;
2. place renewables reproducibly;
3. generate IID Gaussian, correlated Gaussian, IID Student-t, and correlated Student-t forecast errors with matched marginal variance;
4. propagate errors through AC power flow with generator recourse;
5. measure voltage, thermal, generator, and convergence failures;
6. report joint violation probability with Wilson 95% confidence intervals.

Initial benchmark: `pglib_opf_case57_ieee`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Clone PGLib beside this repository:

```bash
git clone https://github.com/power-grid-lib/pglib-opf.git ../pglib-opf
```

Record the exact PGLib commit used for every reported experiment.

## Smoke test

```bash
pytest
python scripts/validate_case.py --config configs/case57.yaml
python scripts/run_monte_carlo.py --config configs/E01_uncertainty.yaml --n-samples 2000
```

Paper runs should use at least 20,000 Monte Carlo samples.

## Scientific guardrails

- Gaussian and Student-t families must have matched marginal variance.
- Use common random seeds/scenarios for paired comparisons.
- Report PF non-convergence separately from operational-limit violations.
- PGLib cases without coordinates use graph distance; do not call it geographic correlation.
- Populate paper numbers only from saved experiment outputs.
