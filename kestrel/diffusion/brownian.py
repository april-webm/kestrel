# kestrel/diffusion/brownian.py
"""Brownian motion and Geometric Brownian Motion implementations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from kestrel.base import StochasticProcess
from kestrel.utils.fisher_information import bm_fisher_information, gbm_fisher_information
from kestrel.utils.kestrel_result import KestrelResult


class BrownianMotion(StochasticProcess):
    """
    Standard Brownian Motion (Wiener Process) with drift.

    SDE: dX_t = mu * dt + sigma * dW_t

    Parameters
    ----------
    mu : float, optional
        Drift coefficient.
    sigma : float, optional
        Volatility (diffusion coefficient).
    """

    mu: Optional[float]
    sigma: Optional[float]

    def __init__(self, mu: Optional[float] = None, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def fit(
        self,
        data: pd.Series,
        dt: Optional[float] = None,
        method: str = 'mle',
        regularization: Optional[float] = None,
    ) -> None:
        """
        Estimates (mu, sigma) from time-series data.

        Parameters
        ----------
        data : pd.Series
            Observed time-series.
        dt : float, optional
            Time step between observations.
        method : str
            Estimation method: 'mle' or 'moments'.
        regularization : float, optional
            L2 penalty strength on drift parameter mu. When set, the estimator
            becomes the MAP estimate under a Gaussian prior. Closed-form Ridge
            solution is used (no optimizer needed).

        Raises
        ------
        ValueError
            If data not pandas Series or unknown method.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt

        if method == 'mle':
            param_ses = self._fit_mle(data, dt, regularization=regularization)
        elif method == 'moments':
            param_ses = self._fit_moments(data, dt)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle' or 'moments'.")

        self._set_params(
            last_data_point=data.iloc[-1],
            mu=self.mu,
            sigma=self.sigma,
            dt=self.dt_,
            param_ses=param_ses
        )

    def _fit_mle(self, data: pd.Series, dt: float, regularization: Optional[float] = None) -> Dict[str, float]:
        """Estimates parameters using Maximum Likelihood."""
        if len(data) < 2:
            raise ValueError("MLE estimation requires at least 2 data points.")

        x = data.values
        dx = np.diff(x)
        n = len(dx)

        # Sigma estimate (unaffected by regularization on mu)
        self.sigma = float(np.sqrt(np.var(dx, ddof=1) / dt))

        # MLE / Ridge estimate for mu
        if regularization is not None and regularization > 0:
            # Penalised MLE: argmin_mu sum((dx_i - mu*dt)^2)/(2*sigma^2*dt) + lambda*mu^2
            # Closed-form: mu = sum(dx) / (n*dt + 2*lambda*sigma^2*dt)
            self.mu = float(np.sum(dx) / (n * dt + 2 * regularization * self.sigma ** 2 * dt))
        else:
            self.mu = float(np.mean(dx) / dt)

        # Standard errors from Fisher Information
        self._fisher_information_, param_ses = bm_fisher_information(self.sigma, dt, n)
        return param_ses

    def _fit_moments(self, data: pd.Series, dt: float) -> Dict[str, float]:
        """Estimates parameters using method of moments."""
        if len(data) < 2:
            raise ValueError("Moments estimation requires at least 2 data points.")

        x = data.values
        dx = np.diff(x)
        n = len(dx)

        # First moment: E[dX] = mu * dt
        self.mu = float(np.mean(dx) / dt)

        # Second moment: Var[dX] = sigma^2 * dt
        self.sigma = float(np.sqrt(np.var(dx, ddof=1) / dt))

        # Standard errors from Fisher Information
        self._fisher_information_, param_ses = bm_fisher_information(self.sigma, dt, n)
        return param_ses

    def sample(self, n_paths: int = 1, horizon: int = 1, dt: Optional[float] = None) -> KestrelResult:
        """
        Simulates future Brownian motion paths.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths.
        horizon : int
            Number of time steps to simulate.
        dt : float, optional
            Simulation time step. Uses fitted dt if None.

        Returns
        -------
        KestrelResult
            Simulation results.
        """
        if not self.is_fitted and (self.mu is None or self.sigma is None):
            raise RuntimeError("Model must be fitted or initialised with parameters before sampling.")

        if dt is None:
            dt = self._dt_ if self.is_fitted and hasattr(self, '_dt_') else 1.0

        mu = self.mu_ if self.is_fitted else self.mu
        sigma = self.sigma_ if self.is_fitted else self.sigma

        if any(p is None for p in [mu, sigma]):
            raise RuntimeError("Parameters (mu, sigma) must be set or estimated to sample.")

        paths = np.zeros((horizon + 1, n_paths))
        if self.is_fitted and hasattr(self, '_last_data_point'):
            initial_val = self._last_data_point
        else:
            initial_val = 0.0

        paths[0, :] = initial_val

        for t in range(horizon):
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_paths)
            paths[t + 1, :] = paths[t, :] + mu * dt + sigma * dW

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion (GBM).

    Standard model for stock prices.
    SDE: dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Equivalent to: d(log S_t) = (mu - 0.5*sigma^2) dt + sigma dW_t

    Parameters
    ----------
    mu : float, optional
        Drift (expected return).
    sigma : float, optional
        Volatility.
    """

    mu: Optional[float]
    sigma: Optional[float]

    def __init__(self, mu: Optional[float] = None, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def fit(
        self,
        data: pd.Series,
        dt: Optional[float] = None,
        method: str = 'mle',
        regularization: Optional[float] = None,
    ) -> None:
        """
        Estimates (mu, sigma) from price time-series.

        Parameters
        ----------
        data : pd.Series
            Price time-series (must be strictly positive).
        dt : float, optional
            Time step between observations.
        method : str
            Estimation method: 'mle' only currently supported.
        regularization : float, optional
            L2 penalty strength on drift parameter mu. Closed-form Ridge
            solution in log-return space.

        Raises
        ------
        ValueError
            If data not pandas Series, contains non-positive values, or unknown method.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if (data <= 0).any():
            raise ValueError("GBM requires strictly positive price data.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt

        if method == 'mle':
            param_ses = self._fit_mle(data, dt, regularization=regularization)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle'.")

        self._set_params(
            last_data_point=data.iloc[-1],
            mu=self.mu,
            sigma=self.sigma,
            dt=self.dt_,
            param_ses=param_ses
        )

    def _fit_mle(self, data: pd.Series, dt: float, regularization: Optional[float] = None) -> Dict[str, float]:
        """Estimates parameters using Maximum Likelihood on log-returns."""
        if len(data) < 2:
            raise ValueError("MLE estimation requires at least 2 data points.")

        prices = data.values
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)

        # MLE for log-returns: r_t = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
        mean_r = np.mean(log_returns)
        var_r = np.var(log_returns, ddof=1)

        # Estimate sigma from variance (unaffected by regularization)
        self.sigma = float(np.sqrt(var_r / dt))

        # Estimate mu
        if regularization is not None and regularization > 0:
            # Ridge on alpha = mu - 0.5*sigma^2 in log-return space
            alpha_penalized = float(np.sum(log_returns) / (n * dt + 2 * regularization * self.sigma ** 2 * dt))
            self.mu = float(alpha_penalized + 0.5 * self.sigma ** 2)
        else:
            self.mu = float(mean_r / dt + 0.5 * self.sigma ** 2)

        # Standard errors from Fisher Information
        self._fisher_information_, param_ses = gbm_fisher_information(self.mu, self.sigma, dt, n)
        return param_ses

    def sample(self, n_paths: int = 1, horizon: int = 1, dt: Optional[float] = None) -> KestrelResult:
        """
        Simulates future GBM price paths.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths.
        horizon : int
            Number of time steps to simulate.
        dt : float, optional
            Simulation time step. Uses fitted dt if None.

        Returns
        -------
        KestrelResult
            Simulation results (all paths strictly positive).
        """
        if not self.is_fitted and (self.mu is None or self.sigma is None):
            raise RuntimeError("Model must be fitted or initialised with parameters before sampling.")

        if dt is None:
            dt = self._dt_ if self.is_fitted and hasattr(self, '_dt_') else 1.0

        mu = self.mu_ if self.is_fitted else self.mu
        sigma = self.sigma_ if self.is_fitted else self.sigma

        if any(p is None for p in [mu, sigma]):
            raise RuntimeError("Parameters (mu, sigma) must be set or estimated to sample.")

        paths = np.zeros((horizon + 1, n_paths))
        if self.is_fitted and hasattr(self, '_last_data_point'):
            initial_val = self._last_data_point
        else:
            initial_val = 1.0  # Default to 1.0 for prices

        paths[0, :] = initial_val

        # Simulate using exact solution: S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        for t in range(horizon):
            Z = np.random.normal(loc=0.0, scale=1.0, size=n_paths)
            paths[t + 1, :] = paths[t, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)

    def expected_price(self, t: float, s0: Optional[float] = None) -> float:
        """
        Computes expected price at time t.

        E[S_t] = S_0 * exp(mu * t)

        Parameters
        ----------
        t : float
            Time horizon.
        s0 : float, optional
            Initial price. Uses last fitted value if None.

        Returns
        -------
        float
            Expected price at time t.
        """
        mu = self.mu_ if self.is_fitted else self.mu
        if mu is None:
            raise RuntimeError("Parameters must be set or estimated first.")

        if s0 is None:
            if self.is_fitted and hasattr(self, '_last_data_point'):
                s0 = self._last_data_point
            else:
                raise ValueError("Initial price s0 must be provided.")

        return float(s0 * np.exp(mu * t))

    def variance_price(self, t: float, s0: Optional[float] = None) -> float:
        """
        Computes variance of price at time t.

        Var[S_t] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)

        Parameters
        ----------
        t : float
            Time horizon.
        s0 : float, optional
            Initial price. Uses last fitted value if None.

        Returns
        -------
        float
            Variance of price at time t.
        """
        mu = self.mu_ if self.is_fitted else self.mu
        sigma = self.sigma_ if self.is_fitted else self.sigma
        if any(p is None for p in [mu, sigma]):
            raise RuntimeError("Parameters must be set or estimated first.")

        if s0 is None:
            if self.is_fitted and hasattr(self, '_last_data_point'):
                s0 = self._last_data_point
            else:
                raise ValueError("Initial price s0 must be provided.")

        return float((s0 ** 2) * np.exp(2 * mu * t) * (np.exp(sigma ** 2 * t) - 1))
