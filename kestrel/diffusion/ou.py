# kestrel/diffusion/ou.py
"""Ornstein-Uhlenbeck process implementation."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize as sp_minimize

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
    ) -> KestrelResult:
        """
        Estimates (theta, mu, sigma) parameters from time-series data.

        The analytic MLE for the OU process is equivalent to OLS on the AR(1)
        representation. Both 'mle' and 'ar1' methods use this closed-form solution.

        Args:
            data (pd.Series): Time-series data for fitting.
            dt (float, optional): Time step between observations.
                                   If None, inferred from data; else defaults to 1.0.
            method (str): Estimation method: 'mle', 'ar1', or 'kalman'.
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

        Returns:
            KestrelResult: An object containing estimation results, diagnostics, and potentially simulated paths.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt # Store dt on self for sample method if not passed explicitly
        n_obs = len(data) - 1

        if regularization is not None and regularization > 0:
            theta, mu, sigma, param_ses, log_likelihood, residuals = self._fit_penalized_mle(data, dt, regularization)
        elif method in ('mle', 'ar1'):
            theta, mu, sigma, param_ses, log_likelihood, residuals = self._fit_analytic(data, dt, bias_correction=bias_correction)
        elif method == 'kalman':
            theta, mu, sigma, param_ses, log_likelihood, residuals = self._fit_kalman(data, dt)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle', 'ar1', or 'kalman'.")

        self.theta = theta # Store on self for sample method if not passed explicitly
        self.mu = mu
        self.sigma = sigma

        self._post_fit_setup(
            last_data_point=data.iloc[-1],
            dt=self.dt_,
            freq=freq
        )

        return KestrelResult(
            process_name="OUProcess",
            params={'theta': theta, 'mu': mu, 'sigma': sigma},
            param_ses=param_ses,
            log_likelihood=log_likelihood,
            residuals=residuals,
            n_obs=n_obs,
        )

    def _fit_analytic(self, data: pd.Series, dt: float, bias_correction: bool = True) -> Tuple[float, float, float, Dict[str, float], float, np.ndarray]:
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

        param_ses: Dict[str, float] = {}
        theta, mu, sigma = np.nan, np.nan, np.nan # Initialize

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
            theta = 1e-6
            mu = float(np.mean(data))
            sigma = fallback_sigma
            # self._non_stationary_ = True # No longer set on self
            param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        elif phi <= 0:
            warnings.warn(
                f"AR(1) coefficient phi={phi:.6f} <= 0. This implies negative "
                f"autocorrelation inconsistent with a standard OU process. "
                f"Parameters are set to fallback values; standard errors are unavailable.",
                ConvergenceWarning,
                stacklevel=2,
            )
            theta = 1e-6
            mu = float(np.mean(data))
            sigma = fallback_sigma
            # self._non_stationary_ = True # No longer set on self
            param_ses = {'theta': np.nan, 'mu': np.nan, 'sigma': np.nan}
        else:
            # self._non_stationary_ = False # No longer set on self
            theta = float(-np.log(phi) / dt)
            mu = float(c / (1 - phi))

            # Sigma from residual variance
            if theta > 0 and (1 - np.exp(-2 * theta * dt)) > 0:
                sigma_sq_ou = (sigma_epsilon_sq * 2 * theta) / (1 - np.exp(-2 * theta * dt))
                sigma = float(np.sqrt(sigma_sq_ou))
            else:
                sigma = float(np.sqrt(sigma_epsilon_sq / dt))

            if sigma <= 0:
                sigma = 0.1

            # Apply jackknife bias correction if requested
            if bias_correction:
                theta, mu, sigma = self._jackknife_bias_correction(
                    x_t, x_t_plus_dt, dt, theta, mu, sigma
                )
                # self._bias_corrected_ = True # No longer set on self
            else:
                pass
                # self._bias_corrected_ = False # No longer set on self

            # Standard errors from analytical Fisher Information (at final parameter values)
            _, param_ses = ou_fisher_information(
                theta, mu, sigma, dt, n, x_t
            )
        
        # Calculate log-likelihood and residuals based on final parameters
        log_likelihood, residuals = self._calculate_ou_log_likelihood_and_residuals(data, dt, theta, mu, sigma)

        return theta, mu, sigma, param_ses, log_likelihood, residuals

    def _fit_penalized_mle(
        self, data: pd.Series, dt: float, regularization: float
    ) -> Tuple[float, float, float, Dict[str, float], float, np.ndarray]:
        """
        Penalised MLE for OU process via L-BFGS-B.

        Objective: NLL + lambda * (theta^2 + mu^2)

        Uses the analytic solution as initial values. Standard errors are
        computed from the numerical inverse Hessian (these are penalised/posterior
        SEs, not frequentist SEs).
        """
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
            phi_val = np.exp(-theta * dt)
            c_val = mu * (1 - phi_val)
            sigma_eps_sq_val = sigma ** 2 * (1 - phi_val ** 2) / (2 * theta)
            if sigma_eps_sq_val <= 0:
                return np.inf
            residuals_val = x_tp - (c_val + phi_val * x_t)
            nll = 0.5 * n * np.log(2 * np.pi * sigma_eps_sq_val) + np.sum(residuals_val ** 2) / (2 * sigma_eps_sq_val)
            penalty = regularization * (theta ** 2 + mu ** 2)
            return nll + penalty

        result = sp_minimize(
            penalized_nll,
            [theta0, mu0, sigma0],
            bounds=[(1e-6, None), (None, None), (1e-6, None)],
            method='L-BFGS-B',
        )

        theta, mu, sigma = np.nan, np.nan, np.nan # Initialize
        if result.success:
            theta, mu, sigma = float(result.x[0]), float(result.x[1]), float(result.x[2])
        else:
            theta, mu, sigma = float(theta0), float(mu0), float(sigma0)

        # self._bias_corrected_ = False # No longer set on self
        # self._non_stationary_ = False # No longer set on self

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

        log_likelihood, residuals = self._calculate_ou_log_likelihood_and_residuals(data, dt, theta, mu, sigma)

        return theta, mu, sigma, param_ses, log_likelihood, residuals

    def _fit_kalman(self, data: pd.Series, dt: float) -> Tuple[float, float, float, Dict[str, float], float, np.ndarray]:
        """
        Estimates OU parameters using Kalman Filter-based MLE.
        The Kalman Filter is used to calculate the likelihood of the observations,
        which is then maximized using scipy.optimize.minimize.
        Assumes direct observation of the state.
        """
        if len(data) < 2:
            raise ValueError("Kalman filter estimation requires at least 2 data points.")

        x = data.values # Observations
        n = len(x)

        # Initial parameter estimates (e.g., from OLS/analytic MLE)
        x_t = data.iloc[:-1].values
        x_tp = data.iloc[1:].values
        n_ols = len(x_t)
        X_reg = np.vstack([np.ones(n_ols), x_t]).T
        beta_hat, _, _, _ = np.linalg.lstsq(X_reg, x_tp, rcond=None)
        c0, phi0 = beta_hat[0], beta_hat[1]
        
        theta0, mu0, sigma0 = 1.0, float(np.mean(data)), float(np.std(np.diff(data.values)) / np.sqrt(dt))

        if 0 < phi0 < 1.0 - 1e-6:
            theta0 = -np.log(phi0) / dt
            mu0 = c0 / (1 - phi0)
            eps0 = x_tp - (c0 + phi0 * x_t)
            sig_eps_sq0 = float(np.var(eps0))
            if theta0 > 0 and (1 - np.exp(-2 * theta0 * dt)) > 0:
                sigma0 = np.sqrt(sig_eps_sq0 * 2 * theta0 / (1 - np.exp(-2 * theta0 * dt)))
            else:
                sigma0 = np.sqrt(sig_eps_sq0 / dt)
        
        initial_params = [theta0, mu0, sigma0]
        bounds = [(1e-6, None), (None, None), (1e-6, None)] # theta, sigma > 0

        # Minimize negative log-likelihood from Kalman Filter
        result = sp_minimize(
            self._kalman_filter_log_likelihood,
            initial_params,
            args=(x, dt),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        theta, mu, sigma = np.nan, np.nan, np.nan
        if result.success:
            theta, mu, sigma = float(result.x[0]), float(result.x[1]), float(result.x[2])
        else:
            # Fallback to initial estimates if optimization fails
            theta, mu, sigma = initial_params

        # Recalculate likelihood and get residuals with optimal params
        # Note: For Kalman filter, residuals are actually innovation residuals, not
        # the simple conditional Gaussian residuals. For simplicity, we reuse
        # _calculate_ou_log_likelihood_and_residuals here, but this is a simplification.
        log_likelihood, residuals = self._calculate_ou_log_likelihood_and_residuals(data, dt, theta, mu, sigma) 
        
        # Standard errors from numerical inverse Hessian of the optimizer result
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


        return theta, mu, sigma, param_ses, log_likelihood, residuals

    @staticmethod
    def _kalman_filter_log_likelihood(params: np.ndarray, y: np.ndarray, dt: float) -> float:
        """
        Calculates the negative log-likelihood of observations `y` using the Kalman Filter
        for an OU process with parameters `params = [theta, mu, sigma]`.
        Assumes direct observation of the state, i.e., Y_t = X_t (no observation noise).

        State equation: X_t+1 = F * X_t + U + W_t, with W_t ~ N(0, Q)
        Observation equation: Y_t = H * X_t + V_t, with V_t ~ N(0, R) (here R=0)
        """
        theta, mu, sigma = params
        
        if theta <= 0 or sigma <= 0:
            return np.inf

        n_obs = len(y)
        
        # Transition matrix (scalar for OU)
        F = np.exp(-theta * dt)
        
        # Control input (deterministic part)
        U = mu * (1 - F)
        
        # Process noise covariance (Q)
        Q = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        
        if Q <= 0: # Ensure Q is positive
            return np.inf

        # Observation matrix (Y_t = X_t)
        H = 1.0
        # Observation noise covariance (R) - assume zero for now (direct observation)
        R = 1e-12 # A very small number to represent practically zero observation noise, avoid division by zero

        # Initialize Kalman Filter
        x_hat = y[0] # Initial state estimate
        P = Q        # Initial state covariance (assume steady state)

        log_likelihood = 0.0
        
        for i in range(1, n_obs):
            # Prediction Step
            x_pred = F * x_hat + U
            P_pred = F * P * F + Q

            # Update Step
            # Innovation (measurement residual)
            innovation = y[i] - H * x_pred
            
            # Innovation covariance
            S = H * P_pred * H + R
            
            if S <= 0: # Ensure innovation covariance is positive
                return np.inf

            # Kalman Gain
            K = P_pred * H / S

            # Update state estimate
            x_hat = x_pred + K * innovation

            # Update state covariance
            P = (1 - K * H) * P_pred
            
            # Add log-likelihood contribution
            log_likelihood += -0.5 * np.log(2 * np.pi * S) - 0.5 * (innovation ** 2 / S)

        return -log_likelihood # Return negative log-likelihood for minimization

    def _calculate_ou_log_likelihood_and_residuals(self, data: pd.Series, dt: float, theta: float, mu: float, sigma: float) -> Tuple[float, np.ndarray]:
        """Calculates log-likelihood and residuals for OU Process."""
        x_t = data.iloc[:-1].values
        x_t_plus_dt = data.iloc[1:].values
        n = len(x_t)

        if theta <= 0 or sigma <= 0:
            warnings.warn("Theta or Sigma is non-positive, log-likelihood and residuals might be inaccurate.", RuntimeWarning)
            return -np.inf, np.full(n, np.nan)

        # Conditional mean and variance for X_{t+dt} | X_t
        exp_minus_theta_dt = np.exp(-theta * dt)
        conditional_mean = x_t * exp_minus_theta_dt + mu * (1 - exp_minus_theta_dt)
        conditional_variance = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)

        if conditional_variance <= 0:
            warnings.warn("Conditional variance is non-positive, log-likelihood and residuals might be inaccurate.", RuntimeWarning)
            return -np.inf, np.full(n, np.nan)

        log_likelihood = np.sum(stats.norm.logpdf(x_t_plus_dt, loc=conditional_mean, scale=np.sqrt(conditional_variance)))
        residuals = (x_t_plus_dt - conditional_mean) / np.sqrt(conditional_variance)

        return log_likelihood, residuals

    def _jackknife_bias_correction(
        self,
        x_t: np.ndarray,
        x_tp: np.ndarray,
        dt: float,
        theta_full: float,
        mu_full: float,
        sigma_full: float,
    ) -> Tuple[float, float, float]:
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
        Simulates future paths using the exact transition density.

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
        
        if theta <= 0:
            warnings.warn(f"Theta ({theta}) must be positive for exact OU sampling formula. Falling back to Euler-Maruyama.", RuntimeWarning)
            # Fallback to Euler-Maruyama if theta is non-positive
            paths = np.zeros((horizon + 1, n_paths))
            if self.is_fitted and hasattr(self, '_last_data_point'):
                initial_val = self._last_data_point
                paths[0, :] = initial_val
            else:
                initial_val = mu
                paths[0, :] = initial_val

            for t in range(horizon):
                dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_paths)
                paths[t + 1, :] = paths[t, :] + theta * (mu - paths[t, :]) * dt + sigma * dW
            return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)

        paths = np.zeros((horizon + 1, n_paths))
        if self.is_fitted and hasattr(self, '_last_data_point'):
            initial_val = self._last_data_point
            paths[0, :] = initial_val
        else:
            initial_val = mu # Start at mean if not fitted
            paths[0, :] = initial_val

        # Exact simulation using transition density
        # X_{t+dt} = X_t * e^{-theta*dt} + mu * (1 - e^{-theta*dt}) + sigma_t * epsilon
        # where sigma_t = sigma * sqrt((1 - e^{-2*theta*dt}) / (2*theta))
        exp_minus_theta_dt = np.exp(-theta * dt)
        
        # This is the conditional standard deviation of X_{t+dt} given X_t
        sigma_sd_term = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))

        for t in range(horizon):
            epsilon = np.random.normal(loc=0.0, scale=1.0, size=n_paths)
            paths[t + 1, :] = (paths[t, :] * exp_minus_theta_dt +
                               mu * (1 - exp_minus_theta_dt) +
                               sigma_sd_term * epsilon)

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)