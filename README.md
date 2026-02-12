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

- **Analytic MLEs where they exist**: closed-form estimators for linear SDEs (BM, GBM, OU), numerical optimisation only when the model demands it (CIR, Merton)
- **Standard errors on every parameter**: analytical Fisher information or delta-method standard errors, not just point estimates
- **Comprehensive Estimation Diagnostics**: `fit()` returns a `KestrelResult` object with log-likelihood, AIC, BIC, and residual analysis tools (Ljung-Box, normality tests, Q-Q plots)
- **Consistent `fit()` / `sample()` API**: scikit-learn-style interface across all processes
- **Automatic `dt` inference**: pass a `DatetimeIndex` and Kestrel converts to annualised time steps
- **Fully typed and tested**: type annotations throughout, 74 tests covering parameter recovery, edge cases, and sampling moments

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
result = model.fit(data, dt=1/252) # fit() now returns a KestrelResult object

print(f"theta = {result.params['theta']:.4f} +/- {result.param_ses['theta']:.4f}")
print(f"mu    = {result.params['mu']:.4f} +/- {result.param_ses['mu']:.4f}")
print(f"sigma = {result.params['sigma']:.4f} +/- {result.param_ses['sigma']:.4f}")

print(f"Log-Likelihood: {result.log_likelihood:.2f}")
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")

# Perform residual diagnostics
result.residuals.plot(title="Standardized Residuals")
result.ljung_box()
result.normality_test()
result.qq_plot()

# You can still sample from the fitted model instance
paths = model.sample(n_paths=1000, horizon=252)
paths.plot(title="OU Process - 1000 Simulated Paths")
```

## Supported Processes

| Process                   | Class                     | Estimation                                   | Sampling                 | Status |
| ------------------------- | ------------------------- | -------------------------------------------- | ------------------------ | ------ |
| Brownian Motion           | `BrownianMotion`          | Analytic MLE, moments, GMM                   | Exact                    | Stable |
| Geometric Brownian Motion | `GeometricBrownianMotion` | Analytic MLE                                 | Exact (log-normal)       | Stable |
| Ornstein-Uhlenbeck        | `OUProcess`               | Analytic MLE (= OLS on AR(1)), Kalman Filter | Exact                    | Stable |
| Cox-Ingersoll-Ross        | `CIRProcess`              | Numerical MLE, least squares                 | Exact                    | Stable |
| Merton Jump Diffusion     | `MertonProcess`           | EM                                           | Euler-Maruyama + Poisson | Stable |

## Usage

### Ornstein-Uhlenbeck Process

Mean-reverting process for interest rates, volatility, and spread modelling.

```python
from kestrel import OUProcess

model = OUProcess()
ou_result = model.fit(data, dt=1/252)  # analytic MLE (closed-form)
print(f"OU Theta: {ou_result.params['theta']:.4f}")

simulation = model.sample(n_paths=100, horizon=252)
```

### Geometric Brownian Motion

Log-normal price dynamics (Black-Scholes model).

```python
from kestrel import GeometricBrownianMotion

model = GeometricBrownianMotion()
gbm_result = model.fit(price_data, dt=1/252)
print(f"GBM Sigma: {gbm_result.params['sigma']:.4f}")

expected = model.expected_price(t=1.0, s0=100)
variance = model.variance_price(t=1.0, s0=100)
```

### Cox-Ingersoll-Ross Process

Non-negative mean-reverting process for interest rate modelling.

```python
from kestrel import CIRProcess

model = CIRProcess()
cir_result = model.fit(rate_data, dt=1/252, method='mle')
print(f"CIR Kappa: {cir_result.params['kappa']:.4f}")

if model.feller_condition_satisfied(cir_result.params['kappa'], cir_result.params['theta'], cir_result.params['sigma']):
    print("Process guaranteed to remain strictly positive")
```

### Merton Jump Diffusion

GBM with Poisson-distributed jumps for capturing sudden market movements.

```python
from kestrel import MertonProcess

model = MertonProcess()
merton_result = model.fit(log_returns, dt=1/252)
print(f"Merton Lambda: {merton_result.params['lambda_']:.4f}")

total_drift = model.expected_return()
total_var = model.total_variance()
```

## API

All processes inherit from `StochasticProcess`. The `fit` method now returns a `KestrelResult` object containing comprehensive estimation diagnostics.

| Method / Property (on StochasticProcess) | Description                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------------- |
| `fit(data, dt, method, ...)`             | Estimates parameters from time-series data and returns a `KestrelResult` object. |
| `sample(n_paths, horizon, dt)`           | Simulates Monte Carlo paths (returns `KestrelResult` containing paths).          |
| `is_fitted`                              | Whether the model has been fitted.                                               |

The `KestrelResult` object, returned by `fit()` and `sample()`, contains:

| Method / Property (on KestrelResult) | Description                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `paths`                              | `pd.DataFrame` of simulated paths (if from `sample()`).                      |
| `process_name`                       | Name of the fitted process.                                                  |
| `params`                             | Dictionary of estimated parameters (e.g., `result.params['theta']`).         |
| `param_ses`                          | Dictionary of parameter standard errors (e.g., `result.param_ses['theta']`). |
| `log_likelihood`                     | Log-likelihood of the fit.                                                   |
| `aic`                                | Akaike Information Criterion.                                                |
| `bic`                                | Bayesian Information Criterion.                                              |
| `residuals`                          | `pd.Series` of residuals from the fit, for diagnostic analysis.              |
| `n_obs`                              | Number of observations used in the fit.                                      |
| `plot()`                             | Plots simulation paths.                                                      |
| `mean_path()`                        | Calculates the mean across simulation paths.                                 |
| `percentile_paths()`                 | Calculates percentile paths for simulations.                                 |
| `ljung_box(lags)`                    | Performs Ljung-Box test for autocorrelation in residuals.                    |
| `normality_test()`                   | Performs Shapiro-Wilk test for normality on residuals.                       |
| `qq_plot()`                          | Generates a Q-Q plot of the residuals.                                       |


## Testing

```bash
pytest tests/ -v
```

The test suite includes parameter recovery tests (simulate from known parameters, fit, verify estimates fall within confidence intervals), edge-case tests, pinned-seed regression tests, and sampling moment validation. New tests cover the diagnostics and information criteria provided by the `KestrelResult` object.

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