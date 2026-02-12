# kestrel/diffusion/ou.py
"""Ornstein-Uhlenbeck process implementation."""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

from kestrel.base import StochasticProcess
from kestrel.utils.fisher_information import ou_fisher_information
from kestrel.utils.kestrel_result import KestrelResult
from kestrel.utils.warnings import BiasWarning, ConvergenceWarning


class OUProcess(StochasticProcess):
    """
    Ornstein-Uhlenbeck (OU) Process.

    Models mean-reverting stochastic dynamics via SDE:
    dX_t = theta * (mu - X_t) dt + sigma dW_t

    Parameters:
        theta (float, optional): Mean reversion speed.
        mu (float, optional): Long-term mean.
        sigma (float, optional): Volatility.
    """

    theta: Optional[float]
    mu: Optional[float]
    sigma: Optional[float]

    def __init__(self, theta: Optional[float] = None, mu: Optional[float] = None, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def fit(
        self,
        data: pd.Series,
        dt: Optional[float] = None,
        method: str = 'mle',
        freq: Optional[str] = None,
        bias_correction: bool = True,
        regularization: Optional[float] = None,
    ) -> None:
        """
        Estimates (theta, mu, sigma) parameters from time-series data.

        The analytic MLE for the OU process is equivalent to OLS on the AR(1)
        representation. Both 'mle' and 'ar1' methods use this closed-form solution.

        Args:
            data (pd.Series): Time-series data for fitting.
            dt (float, optional): Time step between observations.
                                   If None, inferred from data; else defaults to 1.0.
            method (str): Estimation method: 'mle' or 'ar1' (both use analytic MLE).
            freq (str, optional): Data frequency if `data` has `DatetimeIndex`.
                                  e.g., 'B' (business day), 'D' (calendar day).
                                  Converts `dt` to annual basis. Inferred if None.
            bias_correction (bool): If True (default), applies delete-1 jackknife
                                     bias correction to the mean-reversion speed estimate.
                                     Corrects the Hurwicz bias that causes systematic
                                     overestimation of theta in finite samples.
                                     Skipped when regularization is active.
            regularization (float, optional): L2 penalty strength on theta and mu.
                                              When set, uses penalised MLE via scipy
                                              instead of analytic solution. Bias correction
                                              is skipped when regularization is active.
        Raises:
            ValueError: Input data not pandas Series or unknown estimation method.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt

        if regularization is not None and regularization > 0:
            self._fit_penalized_mle(data, dt, freq, regularization)
        elif method in ('mle', 'ar1'):
            self._fit_analytic(data, dt, freq, bias_correction=bias_correction)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle' or 'ar1'.")

    def _fit_analytic(self, data: pd.Series, dt: float, freq: Optional[str] = None, bias_correction: bool = True) -> None:
        """
        Estimates OU parameters using the analytic MLE (equivalent to AR(1) regression).

        The OU transition density is Gaussian:
            X_{t+dt} | X_t ~ N(X_t * exp(-theta*dt) + mu*(1-exp(-theta*dt)),
                              sigma^2*(1-exp(-2*theta*dt))/(2*theta))

        This is equivalent to AR(1): X_{t+1} = c + phi*X_t + epsilon
        where phi = exp(-theta*dt), c = mu*(1-phi).
        OLS gives the exact MLE.
        """
        if len(data) < 2:
            raise ValueError("Estimation requires at least 2 data points.")

        x_t = data.iloc[:-1].values
        x_t_plus_dt = data.iloc[1:].values
        n = len(x_t)

        # OLS regression: x_{t+dt} = c + phi * x_t + epsilon
        X_reg = np.vstack([np.ones(n), x_t]).T
        beta_hat, ss_residuals, rank, s = np.linalg.lstsq(X_reg, x_t_plus_dt, rcond=None)
        c, phi = beta_hat[0], beta_hat[1]

        # Compute regression standard errors
        epsilon_t = x_t_plus_dt - (c + phi * x_t)
        sigma_epsilon_sq = float(np.var(epsilon_t))
        if len(ss_residuals) > 0 and (n - rank) > 0:
            sigma_epsilon_sq_ols = float(ss_residuals[0] / (n - rank))
        else:
            sigma_epsilon_sq_ols = sigma_epsilon_sq

        try:
            cov_beta = np.linalg.inv(X_reg.T @ X_reg) * sigma_epsilon_sq_ols
            se_c = float(np.sqrt(cov_beta[0, 0]))
            se_phi = float(np.sqrt(cov_beta[1, 1]))
            cov_c_phi = float(cov_beta[0, 1])
        except np.linalg.LinAlgError:
            warnings.warn(
                "Could not invert (X_reg.T @ X_reg) for standard errors.",
                ConvergenceWarning,
                stacklevel=2,
            )
            se_c, se_phi, cov_c_phi = np.nan, np.nan, np.nan

        param_ses: Dict[str, float] = {}

        # Compute data-derived fallback sigma
        dx = np.diff(data.values)
        fallback_sigma = float(np.sqrt(np.var(dx, ddof=1) / dt)) if len(dx) > 1 else 0.1

        # Map AR(1) coefficients to OU parameters
        if phi >= 1.0 - 1e-6:
            warnings.warn(
                f"AR(1) coefficient phi={phi:.6f} >= 1. The data appears "
                f"non-stationary; the OU mean-reversion model is not appropriate. "
                f"Parameters are set to fallback values; standard errors are unavailable.",
                ConvergenceWarning,
                stacklevel=2,
            )
            self.theta = 1e-6
            self.mu = float(np.mean(data))
            self.sigma = fallback_sigma
            self._non_stationary_ = True
            param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        elif phi <= 0:
            warnings.warn(
                f"AR(1) coefficient phi={phi:.6f} <= 0. This implies negative "
                f"autocorrelation inconsistent with a standard OU process. "
                f"Parameters are set to fallback values; standard errors are unavailable.",
                ConvergenceWarning,
                stacklevel=2,
            )
            self.theta = 1e-6
            self.mu = float(np.mean(data))
            self.sigma = fallback_sigma
            self._non_stationary_ = True
            param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        else:
            self._non_stationary_ = False
            self.theta = float(-np.log(phi) / dt)
            self.mu = float(c / (1 - phi))

            # Sigma from residual variance
            if self.theta > 0 and (1 - np.exp(-2 * self.theta * dt)) > 0:
                sigma_sq_ou = (sigma_epsilon_sq * 2 * self.theta) / (1 - np.exp(-2 * self.theta * dt))
                self.sigma = float(np.sqrt(sigma_sq_ou))
            else:
                self.sigma = float(np.sqrt(sigma_epsilon_sq / dt))

            if self.sigma <= 0:
                self.sigma = 0.1

            # Apply jackknife bias correction if requested
            if bias_correction:
                self.theta, self.mu, self.sigma = self._jackknife_bias_correction(
                    x_t, x_t_plus_dt, dt, self.theta, self.mu, self.sigma
                )
                self._bias_corrected_ = True
            else:
                self._bias_corrected_ = False

            # Standard errors from analytical Fisher Information (at final parameter values)
            self._fisher_information_, param_ses = ou_fisher_information(
                self.theta, self.mu, self.sigma, dt, n, x_t
            )

        self._set_params(
            last_data_point=data.iloc[-1],
            theta=self.theta,
            mu=self.mu,
            sigma=self.sigma,
            dt=self.dt_,
            freq=freq,
            param_ses=param_ses
        )

    def _fit_penalized_mle(
        self, data: pd.Series, dt: float, freq: Optional[str], regularization: float
    ) -> None:
        """
        Penalised MLE for OU process via L-BFGS-B.

        Objective: NLL + lambda * (theta^2 + mu^2)

        Uses the analytic solution as initial values. Standard errors are
        computed from the numerical inverse Hessian (these are penalised/posterior
        SEs, not frequentist SEs).
        """
        from scipy.optimize import minimize as sp_minimize

        if len(data) < 2:
            raise ValueError("Estimation requires at least 2 data points.")

        x_t = data.iloc[:-1].values
        x_tp = data.iloc[1:].values
        n = len(x_t)

        # Get analytic solution for initial values
        X_reg = np.vstack([np.ones(n), x_t]).T
        beta_hat, _, _, _ = np.linalg.lstsq(X_reg, x_tp, rcond=None)
        c0, phi0 = beta_hat[0], beta_hat[1]

        if 0 < phi0 < 1.0 - 1e-6:
            theta0 = -np.log(phi0) / dt
            mu0 = c0 / (1 - phi0)
            eps0 = x_tp - (c0 + phi0 * x_t)
            sig_eps_sq0 = float(np.var(eps0))
            if theta0 > 0 and (1 - np.exp(-2 * theta0 * dt)) > 0:
                sigma0 = np.sqrt(sig_eps_sq0 * 2 * theta0 / (1 - np.exp(-2 * theta0 * dt)))
            else:
                sigma0 = np.sqrt(sig_eps_sq0 / dt)
        else:
            theta0, mu0, sigma0 = 1.0, float(np.mean(data)), float(np.std(np.diff(data.values)) / np.sqrt(dt))

        def penalized_nll(params: np.ndarray) -> float:
            theta, mu, sigma = params
            if theta <= 0 or sigma <= 0:
                return np.inf
            phi = np.exp(-theta * dt)
            c = mu * (1 - phi)
            sigma_eps_sq = sigma ** 2 * (1 - phi ** 2) / (2 * theta)
            if sigma_eps_sq <= 0:
                return np.inf
            residuals = x_tp - (c + phi * x_t)
            nll = 0.5 * n * np.log(2 * np.pi * sigma_eps_sq) + np.sum(residuals ** 2) / (2 * sigma_eps_sq)
            penalty = regularization * (theta ** 2 + mu ** 2)
            return nll + penalty

        result = sp_minimize(
            penalized_nll,
            [theta0, mu0, sigma0],
            bounds=[(1e-6, None), (None, None), (1e-6, None)],
            method='L-BFGS-B',
        )

        if result.success:
            self.theta, self.mu, self.sigma = float(result.x[0]), float(result.x[1]), float(result.x[2])
        else:
            self.theta, self.mu, self.sigma = float(theta0), float(mu0), float(sigma0)

        self._bias_corrected_ = False
        self._non_stationary_ = False

        # SEs from numerical inverse Hessian
        param_ses: Dict[str, float] = {}
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if callable(getattr(result.hess_inv, 'todense', None)):
                cov_matrix = np.asarray(result.hess_inv.todense())
            else:
                cov_matrix = np.asarray(result.hess_inv)
            if cov_matrix.shape == (3, 3):
                param_ses = {
                    'theta': float(np.sqrt(max(0, cov_matrix[0, 0]))),
                    'mu': float(np.sqrt(max(0, cov_matrix[1, 1]))),
                    'sigma': float(np.sqrt(max(0, cov_matrix[2, 2]))),
                }
            else:
                param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        else:
            param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}

        self._set_params(
            last_data_point=data.iloc[-1],
            theta=self.theta,
            mu=self.mu,
            sigma=self.sigma,
            dt=self.dt_,
            freq=freq,
            param_ses=param_ses,
        )

    def _jackknife_bias_correction(
        self,
        x_t: np.ndarray,
        x_tp: np.ndarray,
        dt: float,
        theta_full: float,
        mu_full: float,
        sigma_full: float,
    ) -> tuple:
        """
        Delete-1 jackknife bias correction for OU parameters.

        Corrects the Hurwicz bias that causes systematic overestimation of
        theta (mean-reversion speed) in finite samples. The jackknife estimate is:
            theta_jk = n * theta_full - (n-1) * mean(theta_loo)

        For time series, removes each consecutive transition (x_t, x_{t+dt})
        to maintain the Markov structure.

        Complexity: O(n^2) — each of n leave-one-out samples requires a 2x2 OLS
        solve. Fast enough for n < 50,000.

        Returns:
            (theta_corrected, mu_corrected, sigma_corrected)
        """
        n = len(x_t)

        theta_loo = np.empty(n)
        mu_loo = np.empty(n)
        sigma_loo = np.empty(n)
        valid = np.ones(n, dtype=bool)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            x_t_i = x_t[mask]
            x_tp_i = x_tp[mask]

            X_reg_i = np.vstack([np.ones(n - 1), x_t_i]).T
            beta_hat_i, _, _, _ = np.linalg.lstsq(X_reg_i, x_tp_i, rcond=None)
            c_i, phi_i = beta_hat_i[0], beta_hat_i[1]

            if 0 < phi_i < 1.0 - 1e-6:
                theta_i = -np.log(phi_i) / dt
                mu_i = c_i / (1 - phi_i)
                eps_i = x_tp_i - (c_i + phi_i * x_t_i)
                sigma_eps_sq_i = float(np.var(eps_i))
                if theta_i > 0 and (1 - np.exp(-2 * theta_i * dt)) > 0:
                    sigma_sq_i = sigma_eps_sq_i * 2 * theta_i / (1 - np.exp(-2 * theta_i * dt))
                    sigma_i = np.sqrt(sigma_sq_i)
                else:
                    sigma_i = np.sqrt(sigma_eps_sq_i / dt)
                theta_loo[i] = theta_i
                mu_loo[i] = mu_i
                sigma_loo[i] = sigma_i
            else:
                valid[i] = False

        n_valid = valid.sum()
        if n_valid < n * 0.5:
            warnings.warn(
                "Jackknife bias correction failed: too many leave-one-out samples "
                "were non-stationary. Returning uncorrected estimates.",
                BiasWarning,
                stacklevel=3,
            )
            return theta_full, mu_full, sigma_full

        theta_jk = float(n * theta_full - (n - 1) * np.mean(theta_loo[valid]))
        mu_jk = float(n * mu_full - (n - 1) * np.mean(mu_loo[valid]))
        sigma_jk = float(n * sigma_full - (n - 1) * np.mean(sigma_loo[valid]))

        # Ensure physical constraints
        theta_jk = max(1e-8, theta_jk)
        sigma_jk = max(1e-8, sigma_jk)

        return theta_jk, mu_jk, sigma_jk

    def sample(self, n_paths: int = 1, horizon: int = 1, dt: Optional[float] = None) -> KestrelResult:
        """
        Simulates future paths using Euler-Maruyama method.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths to generate.
        horizon : int
            Number of future time steps to simulate.
        dt : float, optional
            Simulation time step. Uses fitted dt if None.

        Returns
        -------
        KestrelResult
            Simulation results with plotting and analysis methods.

        Raises
        ------
        RuntimeError
            If model not fitted and parameters not provided at initialisation.
        """
        if not self.is_fitted and (self.theta is None or self.mu is None or self.sigma is None):
            raise RuntimeError("Model must be fitted or initialised with parameters before sampling.")

        if dt is None:
            dt = self._dt_ if self.is_fitted and hasattr(self, '_dt_') else 1.0

        theta = self.theta_ if self.is_fitted else self.theta
        mu = self.mu_ if self.is_fitted else self.mu
        sigma = self.sigma_ if self.is_fitted else self.sigma

        if any(p is None for p in [theta, mu, sigma]):
            raise RuntimeError("OU parameters (theta, mu, sigma) must be set or estimated to sample.")

        paths = np.zeros((horizon + 1, n_paths))
        if self.is_fitted and hasattr(self, '_last_data_point'):
            initial_val = self._last_data_point
            paths[0, :] = initial_val
        else:
            initial_val = mu
            paths[0, :] = initial_val

        # Euler-Maruyama simulation
        for t in range(horizon):
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_paths)
            paths[t + 1, :] = paths[t, :] + theta * (mu - paths[t, :]) * dt + sigma * dW

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)
