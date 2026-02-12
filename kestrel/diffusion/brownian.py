# kestrel/diffusion/brownian.py
"""Brownian motion and Geometric Brownian Motion implementations."""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize as sp_minimize # Import for GMM

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
    ) -> KestrelResult:
        """
        Estimates (mu, sigma) from time-series data.

        Parameters
        ----------
        data : pd.Series
            Observed time-series.
        dt : float, optional
            Time step between observations.
        method : str
            Estimation method: 'mle', 'moments', or 'gmm'.
        regularization : float, optional
            L2 penalty strength on drift parameter mu. When set, the estimator
            becomes the MAP estimate under a Gaussian prior. Closed-form Ridge
            solution is used (no optimizer needed).

        Raises
        ------
        ValueError
            If data not pandas Series or unknown method.

        Returns
        -------
        KestrelResult
            An object containing estimation results, diagnostics, and potentially simulated paths.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt
        n_obs = len(data) - 1

        if method == 'mle':
            mu, sigma, param_ses = self._fit_mle(data, dt, regularization=regularization)
        elif method == 'moments':
            mu, sigma, param_ses = self._fit_moments(data, dt)
        elif method == 'gmm':
            mu, sigma, param_ses = self._fit_gmm(data, dt)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle', 'moments', or 'gmm'.")

        self.mu = mu  # Store on self for sample method if not passed explicitly
        self.sigma = sigma # Store on self for sample method if not passed explicitly

        log_likelihood, residuals = self._calculate_bm_log_likelihood_and_residuals(data, dt, mu, sigma)

        self._post_fit_setup(
            last_data_point=data.iloc[-1],
            dt=self.dt_,
            freq=None # No frequency inference for BM
        )

        return KestrelResult(
            process_name="BrownianMotion",
            params={'mu': mu, 'sigma': sigma},
            param_ses=param_ses,
            log_likelihood=log_likelihood,
            residuals=residuals,
            n_obs=n_obs,
        )

    def _fit_mle(self, data: pd.Series, dt: float, regularization: Optional[float] = None) -> Tuple[float, float, Dict[str, float]]:
        """Estimates parameters using Maximum Likelihood."""
        if len(data) < 2:
            raise ValueError("MLE estimation requires at least 2 data points.")

        x = data.values
        dx = np.diff(x)
        n = len(dx)

        # Sigma estimate (unaffected by regularization on mu)
        sigma = float(np.sqrt(np.var(dx, ddof=1) / dt))

        # MLE / Ridge estimate for mu
        if regularization is not None and regularization > 0:
            # Penalised MLE: argmin_mu sum((dx_i - mu*dt)^2)/(2*sigma^2*dt) + lambda*mu^2
            # Closed-form: mu = sum(dx) / (n*dt + 2*lambda*sigma^2*dt)
            mu = float(np.sum(dx) / (n * dt + 2 * regularization * sigma ** 2 * dt))
        else:
            mu = float(np.mean(dx) / dt)

        # Standard errors from Fisher Information
        _, param_ses = bm_fisher_information(sigma, dt, n)
        return mu, sigma, param_ses

    def _fit_moments(self, data: pd.Series, dt: float) -> Tuple[float, float, Dict[str, float]]:
        """Estimates parameters using method of moments."""
        if len(data) < 2:
            raise ValueError("Moments estimation requires at least 2 data points.")

        x = data.values
        dx = np.diff(x)
        n = len(dx)

        # First moment: E[dX] = mu * dt
        mu = float(np.mean(dx) / dt)

        # Second moment: Var[dX] = sigma^2 * dt
        sigma = float(np.sqrt(np.var(dx, ddof=1) / dt))

        # Standard errors from Fisher Information
        _, param_ses = bm_fisher_information(sigma, dt, n)
        return mu, sigma, param_ses

    def _fit_gmm(self, data: pd.Series, dt: float) -> Tuple[float, float, Dict[str, float]]:
        """
        Estimates Brownian Motion parameters using Generalized Method of Moments (GMM).
        Uses first two raw moments of increments.
        """
        if len(data) < 2:
            raise ValueError("GMM estimation requires at least 2 data points.")

        dx = np.diff(data.values)
        n = len(dx)

        # Initial parameter estimates from method of moments
        mu0, sigma0, _ = self._fit_moments(data, dt)
        initial_params = [mu0, sigma0]
        bounds = [(None, None), (1e-6, None)] # sigma > 0

        # Minimize the GMM objective function
        result = sp_minimize(
            self._gmm_objective_function,
            initial_params,
            args=(dx, dt),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )

        mu, sigma = np.nan, np.nan
        if result.success:
            mu, sigma = float(result.x[0]), float(result.x[1])
        else:
            mu, sigma = initial_params # Fallback to initial estimates

        # Standard errors from Fisher Information (for comparability, though GMM has its own SEs)
        _, param_ses = bm_fisher_information(sigma, dt, n)
        return mu, sigma, param_ses

    @staticmethod
    def _gmm_objective_function(params: np.ndarray, dx: np.ndarray, dt: float) -> float:
        """
        GMM objective function for Brownian Motion.
        Moments: E[dX] = mu*dt, E[(dX)^2] = (mu*dt)^2 + sigma^2*dt
        """
        mu, sigma = params

        if sigma <= 0: # Constraint check
            return np.inf

        # Sample moments
        sample_moment1 = np.mean(dx)
        sample_moment2 = np.mean(dx**2)

        # Theoretical moments
        theoretical_moment1 = mu * dt
        theoretical_moment2 = (mu * dt)**2 + sigma**2 * dt

        # Moment conditions (g(params, data) = 0)
        g1 = sample_moment1 - theoretical_moment1
        g2 = sample_moment2 - theoretical_moment2
        
        # Simple identity weighting matrix for now (minimizes sum of squares)
        # For a full GMM, this would be optimal weighting matrix W = S_hat_inv
        objective = g1**2 + g2**2
        return objective


    def _calculate_bm_log_likelihood_and_residuals(self, data: pd.Series, dt: float, mu: float, sigma: float) -> Tuple[float, np.ndarray]:
        """Calculates log-likelihood and residuals for Brownian Motion."""
        x = data.values
        dx = np.diff(x)
        n = len(dx)

        if sigma <= 0:
            # Fallback for log_likelihood calculation in case sigma is problematic
            warnings.warn("Sigma is non-positive, log-likelihood and residuals might be inaccurate.", RuntimeWarning)
            return -np.inf, np.full(n, np.nan)

        # Log-likelihood
        loc = mu * dt
        scale = sigma * np.sqrt(dt)
        log_likelihood = np.sum(norm.logpdf(dx, loc=loc, scale=scale))

        # Residuals
        residuals = (dx - loc) / scale
        return log_likelihood, residuals

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
    ) -> KestrelResult:
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

        Returns
        -------
        KestrelResult
            An object containing estimation results, diagnostics, and potentially simulated paths.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if (data <= 0).any():
            raise ValueError("GBM requires strictly positive price data.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt
        n_obs = len(data) - 1

        if method == 'mle':
            mu, sigma, param_ses = self._fit_mle(data, dt, regularization=regularization)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle'.")

        self.mu = mu # Store on self for sample method if not passed explicitly
        self.sigma = sigma # Store on self for sample method if not passed explicitly

        log_likelihood, residuals = self._calculate_gbm_log_likelihood_and_residuals(data, dt, mu, sigma)

        self._post_fit_setup(
            last_data_point=data.iloc[-1],
            dt=self.dt_,
            freq=None # No frequency inference for GBM
        )

        return KestrelResult(
            process_name="GeometricBrownianMotion",
            params={'mu': mu, 'sigma': sigma},
            param_ses=param_ses,
            log_likelihood=log_likelihood,
            residuals=residuals,
            n_obs=n_obs,
        )

    def _fit_mle(self, data: pd.Series, dt: float, regularization: Optional[float] = None) -> Tuple[float, float, Dict[str, float]]:
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
        sigma = float(np.sqrt(var_r / dt))

        # Estimate mu
        if regularization is not None and regularization > 0:
            # Ridge on alpha = mu - 0.5*sigma^2 in log-return space
            alpha_penalized = float(np.sum(log_returns) / (n * dt + 2 * regularization * sigma ** 2 * dt))
            mu = float(alpha_penalized + 0.5 * sigma ** 2)
        else:
            mu = float(mean_r / dt + 0.5 * sigma ** 2)

        # Standard errors from Fisher Information
        _, param_ses = gbm_fisher_information(mu, sigma, dt, n)
        return mu, sigma, param_ses

    def _calculate_gbm_log_likelihood_and_residuals(self, data: pd.Series, dt: float, mu: float, sigma: float) -> Tuple[float, np.ndarray]:
        """Calculates log-likelihood and residuals for Geometric Brownian Motion."""
        prices = data.values
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)

        if sigma <= 0:
            # Fallback for log_likelihood calculation in case sigma is problematic
            warnings.warn("Sigma is non-positive, log-likelihood and residuals might be inaccurate.", RuntimeWarning)
            return -np.inf, np.full(n, np.nan)

        # For GBM, we use the log-returns which are normally distributed
        alpha = mu - 0.5 * sigma ** 2
        loc = alpha * dt
        scale = sigma * np.sqrt(dt)
        log_likelihood = np.sum(norm.logpdf(log_returns, loc=loc, scale=scale))

        # Residuals are standardized log-returns
        residuals = (log_returns - loc) / scale
        return log_likelihood, residuals

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