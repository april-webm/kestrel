# Kestrel

<p align="center">
  <img src="assets/kestrel.png" alt="Nankeen Kestrel" width="200">
  <br>
  <sub><a href="https://commons.wikimedia.org/w/index.php?curid=74707355">Nankeen Kestrel</a> by patrickkavanagh, <a href="https://creativecommons.org/licenses/by/2.0">CC BY 2.0</a></sub>
</p>

[![PyPI version](https://img.shields.io/pypi/v/stokestrel.svg)](https://pypi.org/project/stokestrel/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern Python library for stochastic process modelling, parameter estimation, and Monte Carlo simulation.

## Overview

Kestrel provides a unified, scikit-learn-style interface for working with stochastic differential equations (SDEs). The library supports parameter estimation from time-series data and path simulation for a variety of continuous and jump-diffusion processes.

### Supported Processes

| Process | Module | Description |
|---------|--------|-------------|
| Brownian Motion | `kestrel.diffusion` | Standard Wiener process with drift |
| Geometric Brownian Motion | `kestrel.diffusion` | Log-normal price dynamics (Black-Scholes) |
| Ornstein-Uhlenbeck | `kestrel.diffusion` | Mean-reverting Gaussian process |
| Cox-Ingersoll-Ross | `kestrel.diffusion` | Mean-reverting process with state-dependent volatility |
| Merton Jump Diffusion | `kestrel.jump_diffusion` | GBM with Poisson-distributed jumps |

### Key Features

- **Consistent API**: All processes follow `fit()` / `sample()` pattern
- **Multiple Estimation Methods**: MLE, AR(1) regression, least squares
- **Standard Error Reporting**: Parameter uncertainty quantification
- **Flexible Time Handling**: Automatic dt inference from DatetimeIndex
- **Simulation Engine**: Euler-Maruyama discretisation with exact solutions where available

## Installation

**Requirements**: Python 3.9+

```bash
pip install stokestrel
```

For development installation:

```bash
git clone https://github.com/april-webm/kestrel.git
cd kestrel
pip install -e ".[dev]"
```

## Quick Start

```python
from kestrel import OUProcess
import pandas as pd

# Load or generate time-series data
data = pd.Series([...])  # Your observed data

# Fit model parameters
model = OUProcess()
model.fit(data, dt=1/252, method='mle')

# View estimated parameters and standard errors
print(f"Mean reversion speed: {model.theta_:.4f} ± {model.theta_se_:.4f}")
print(f"Long-run mean: {model.mu_:.4f} ± {model.mu_se_:.4f}")
print(f"Volatility: {model.sigma_:.4f} ± {model.sigma_se_:.4f}")

# Simulate future paths
paths = model.sample(n_paths=1000, horizon=252)
paths.plot(title="OU Process Simulation")
```

## Usage Examples

### Ornstein-Uhlenbeck Process

Mean-reverting process commonly used for interest rates and volatility modelling.

```python
from kestrel import OUProcess

model = OUProcess()
model.fit(data, dt=1/252, method='mle')  # or method='ar1'

# Simulate from fitted model
simulation = model.sample(n_paths=100, horizon=50)
```

### Geometric Brownian Motion

Standard model for equity prices.

```python
from kestrel import GeometricBrownianMotion

model = GeometricBrownianMotion()
model.fit(price_data, dt=1/252)

# Expected price and variance at horizon
expected = model.expected_price(t=1.0, s0=100)
variance = model.variance_price(t=1.0, s0=100)
```

### Cox-Ingersoll-Ross Process

Ensures non-negativity; suitable for interest rate modelling.

```python
from kestrel import CIRProcess

model = CIRProcess()
model.fit(rate_data, dt=1/252, method='mle')

# Check Feller condition for strict positivity
if model.feller_condition_satisfied():
    print("Process guaranteed to remain positive")
```

### Merton Jump Diffusion

Captures sudden market movements via Poisson jumps.

```python
from kestrel import MertonProcess

model = MertonProcess()
model.fit(log_returns, dt=1/252)

# Jump-adjusted expected return
total_drift = model.expected_return()
total_var = model.total_variance()
```

## API Reference

### Base Interface

All stochastic processes inherit from `StochasticProcess` and implement:

| Method | Description |
|--------|-------------|
| `fit(data, dt, method)` | Estimate parameters from time-series |
| `sample(n_paths, horizon, dt)` | Generate Monte Carlo paths |
| `is_fitted` | Property indicating fit status |
| `params` | Dictionary of estimated parameters |

### Fitted Attributes

After calling `fit()`, estimated parameters are available as attributes with trailing underscore:

```python
model.theta_      # Estimated parameter value
model.theta_se_   # Standard error of estimate
```

## Dependencies

- numpy
- scipy
- pandas
- matplotlib

## Testing

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome. Please ensure:

1. Code follows existing style conventions
2. New features include appropriate tests
3. Documentation is updated accordingly

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If Kestrel is used in academic research, citation is appreciated:

```bibtex
@software{stokestrel,
  title = {Stokestrel: A Modern Stochastic Modelling Library},
  author = {Kidd, April},
  url = {https://github.com/april-webm/kestrel},
  version = {0.1.0},
  year = {2024}
}
```
