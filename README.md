# Kestrel

<p align="center">
  <img src="assets/kestrel.png" alt="Nankeen Kestrel" width="200">
  <br>
  <sub><a href="https://commons.wikimedia.org/w/index.php?curid=74707355">Nankeen Kestrel</a> by patrickkavanagh, <a href="https://creativecommons.org/licenses/by/2.0">CC BY 2.0</a></sub>
</p>

[![PyPI version](https://img.shields.io/pypi/v/stokestrel.svg)](https://pypi.org/project/stokestrel/)
[![CI](https://github.com/april-webm/kestrel/actions/workflows/ci.yml/badge.svg)](https://github.com/april-webm/kestrel/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Robust parameter estimation for stochastic differential equations in Python.

## Key Features

- **Analytic MLEs where they exist** — closed-form estimators for linear SDEs (BM, GBM, OU), numerical optimisation only when the model demands it (CIR, Merton)
- **Standard errors on every parameter** — analytical Fisher information or delta-method standard errors, not just point estimates
- **Consistent `fit()` / `sample()` API** — scikit-learn-style interface across all processes
- **Automatic `dt` inference** — pass a `DatetimeIndex` and Kestrel converts to annualised time steps
- **Fully typed and tested** — type annotations throughout, 74 tests covering parameter recovery, edge cases, and sampling moments

## Installation

```bash
pip install stokestrel
```

**Requirements:** Python 3.9+

For development:

```bash
git clone https://github.com/april-webm/kestrel.git
cd kestrel
pip install -e ".[dev]"
```

## Quick Start

```python
from kestrel import OUProcess

model = OUProcess()
model.fit(data, dt=1/252)

print(f"theta = {model.theta_:.4f} +/- {model.theta_se_:.4f}")
print(f"mu    = {model.mu_:.4f} +/- {model.mu_se_:.4f}")
print(f"sigma = {model.sigma_:.4f} +/- {model.sigma_se_:.4f}")

paths = model.sample(n_paths=1000, horizon=252)
paths.plot(title="OU Process — 1000 Simulated Paths")
```

## Supported Processes

| Process | Class | Estimation | Sampling | Status |
|---------|-------|------------|----------|--------|
| Brownian Motion | `BrownianMotion` | Analytic MLE, moments | Exact | Stable |
| Geometric Brownian Motion | `GeometricBrownianMotion` | Analytic MLE | Exact (log-normal) | Stable |
| Ornstein-Uhlenbeck | `OUProcess` | Analytic MLE (= OLS on AR(1)) | Euler-Maruyama | Stable |
| Cox-Ingersoll-Ross | `CIRProcess` | Numerical MLE, least squares | Euler-Maruyama | Stable |
| Merton Jump Diffusion | `MertonProcess` | Numerical MLE | Euler-Maruyama + Poisson | Stable |

## Usage

### Ornstein-Uhlenbeck Process

Mean-reverting process for interest rates, volatility, and spread modelling.

```python
from kestrel import OUProcess

model = OUProcess()
model.fit(data, dt=1/252)  # analytic MLE (closed-form)

simulation = model.sample(n_paths=100, horizon=252)
```

### Geometric Brownian Motion

Log-normal price dynamics (Black-Scholes model).

```python
from kestrel import GeometricBrownianMotion

model = GeometricBrownianMotion()
model.fit(price_data, dt=1/252)

expected = model.expected_price(t=1.0, s0=100)
variance = model.variance_price(t=1.0, s0=100)
```

### Cox-Ingersoll-Ross Process

Non-negative mean-reverting process for interest rate modelling.

```python
from kestrel import CIRProcess

model = CIRProcess()
model.fit(rate_data, dt=1/252, method='mle')

if model.feller_condition_satisfied():
    print("Process guaranteed to remain strictly positive")
```

### Merton Jump Diffusion

GBM with Poisson-distributed jumps for capturing sudden market movements.

```python
from kestrel import MertonProcess

model = MertonProcess()
model.fit(log_returns, dt=1/252)

total_drift = model.expected_return()
total_var = model.total_variance()
```

## API

All processes inherit from `StochasticProcess`:

| Method / Property | Description |
|-------------------|-------------|
| `fit(data, dt, method)` | Estimate parameters from time-series data |
| `sample(n_paths, horizon, dt)` | Simulate Monte Carlo paths (returns `KestrelResult`) |
| `is_fitted` | Whether the model has been fitted |
| `params` | Dictionary of estimated parameters |

Fitted parameters use a trailing-underscore convention:

```python
model.theta_      # point estimate
model.theta_se_   # standard error
```

## Testing

```bash
pytest tests/ -v
```

The test suite includes parameter recovery tests (simulate from known parameters, fit, verify estimates fall within confidence intervals), edge-case tests, pinned-seed regression tests, and sampling moment validation.

## Contributing

Contributions welcome. Please ensure new features include tests and that all existing tests pass.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{stokestrel,
  title = {Stokestrel: Robust Parameter Estimation for SDEs},
  author = {Kidd, April},
  url = {https://github.com/april-webm/kestrel},
  version = {0.1.0},
  year = {2026}
}
```
