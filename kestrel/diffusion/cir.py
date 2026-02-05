# kestrel/diffusion/cir.py
"""Cox-Ingersoll-Ross process implementation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from kestrel.base import StochasticProcess
from kestrel.utils.kestrel_result import KestrelResult


class CIRProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross (CIR) Process.

    Models mean-reverting dynamics with state-dependent volatility.
    SDE: dX_t = kappa * (theta - X_t) dt + sigma * sqrt(X_t) dW_t

    Feller condition (2*kappa*theta > sigma^2) ensures strict positivity.

    Parameters
    ----------
    kappa : float, optional
        Rate of mean reversion.
    theta : float, optional
        Long-term mean level.
    sigma : float, optional
        Volatility coefficient.
    """

    def __init__(self, kappa: float = None, theta: float = None, sigma: float = None):
        super().__init__()
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def fit(self, data: pd.Series, dt: float = None, method: str = 'mle'):
        """
        Estimates (kappa, theta, sigma) from time-series data.

        Parameters
        ----------
        data : pd.Series
            Observed time-series (must be positive for CIR).
        dt : float, optional
            Time step between observations. Inferred from DatetimeIndex if None.
        method : str
            Estimation method: 'mle' or 'lsq'.

        Raises
        ------
        ValueError
            If data not pandas Series, contains non-positive values, or unknown method.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        # CIR requires strictly positive data
        if (data <= 0).any():
            raise ValueError("CIR process requires strictly positive data.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt

        if method == 'mle':
            param_ses = self._fit_mle(data, dt)
        elif method == 'lsq':
            param_ses = self._fit_lsq(data, dt)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle' or 'lsq'.")

        self._set_params(
            last_data_point=data.iloc[-1],
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            dt=self.dt_,
            param_ses=param_ses
        )

    def _infer_dt(self, data: pd.Series) -> float:
        """Infers dt from DatetimeIndex or defaults to 1.0."""
        if isinstance(data.index, pd.DatetimeIndex):
            if len(data.index) < 2:
                return 1.0

            inferred_timedelta = data.index[1] - data.index[0]
            current_freq = pd.infer_freq(data.index)
            if current_freq is None:
                current_freq = 'B'

            if current_freq in ['B', 'C', 'D']:
                dt = inferred_timedelta / pd.Timedelta(days=252.0)
            elif current_freq.startswith('W'):
                dt = inferred_timedelta / pd.Timedelta(weeks=52)
            elif current_freq in ['M', 'MS', 'BM', 'BMS']:
                dt = inferred_timedelta / pd.Timedelta(days=365 / 12)
            elif current_freq in ['Q', 'QS', 'BQ', 'BQS']:
                dt = inferred_timedelta / pd.Timedelta(days=365 / 4)
            elif current_freq in ['A', 'AS', 'BA', 'BAS', 'Y', 'YS', 'BY', 'BYS']:
                dt = inferred_timedelta / pd.Timedelta(days=365)
            else:
                dt = inferred_timedelta.total_seconds() / (365 * 24 * 3600)

            return max(dt, 1e-10)
        return 1.0

    def _fit_mle(self, data: pd.Series, dt: float) -> dict:
        """
        Estimates CIR parameters using Maximum Likelihood.

        Uses Gaussian approximation to conditional distribution for tractability.
        """
        if len(data) < 2:
            raise ValueError("MLE estimation requires at least 2 data points.")

        x = data.values
        n = len(x)

        # Initial estimates via LSQ for starting values
        x_t = x[:-1]
        x_next = x[1:]
        dx = x_next - x_t

        # Regression: dx/sqrt(x_t) = (kappa*theta)/sqrt(x_t) - kappa*sqrt(x_t) + noise
        # Rewrite as: dx = kappa*theta*dt - kappa*x_t*dt + sigma*sqrt(x_t)*dW
        # Simple moment-based initial guess
        mean_x = np.mean(x)
        var_dx = np.var(dx)

        kappa_0 = max(0.1, -np.cov(dx, x_t)[0, 1] / (np.var(x_t) * dt))
        theta_0 = mean_x
        sigma_0 = max(0.01, np.sqrt(var_dx / (mean_x * dt)))

        initial_params = [kappa_0, theta_0, sigma_0]
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]

        result = minimize(
            self._neg_log_likelihood,
            initial_params,
            args=(x, dt),
            bounds=bounds,
            method='L-BFGS-B'
        )

        if result.success:
            self.kappa, self.theta, self.sigma = result.x
        else:
            # Fallback to LSQ estimates
            self._fit_lsq_internal(data, dt)

        # Compute standard errors from inverse Hessian
        param_ses = self._compute_standard_errors(result, ['kappa', 'theta', 'sigma'])
        return param_ses

    def _neg_log_likelihood(self, params, x, dt):
        """
        Negative log-likelihood for CIR using Gaussian approximation.

        Conditional distribution X_{t+dt} | X_t approximated as Gaussian
        with mean and variance from Euler discretisation.
        """
        kappa, theta, sigma = params

        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return np.inf

        n = len(x)
        ll = 0.0

        for i in range(1, n):
            x_prev = x[i - 1]
            x_curr = x[i]

            # Conditional mean: E[X_{t+dt} | X_t] = X_t * exp(-kappa*dt) + theta*(1 - exp(-kappa*dt))
            exp_kdt = np.exp(-kappa * dt)
            mean_cond = x_prev * exp_kdt + theta * (1 - exp_kdt)

            # Conditional variance (Euler approximation)
            var_cond = (sigma ** 2 * x_prev * (1 - exp_kdt ** 2)) / (2 * kappa)

            if var_cond <= 1e-12:
                return np.inf

            # Gaussian log-likelihood contribution
            ll += -0.5 * np.log(2 * np.pi * var_cond)
            ll += -0.5 * ((x_curr - mean_cond) ** 2 / var_cond)

        return -ll

    def _fit_lsq(self, data: pd.Series, dt: float) -> dict:
        """
        Estimates CIR parameters using Least Squares regression.

        Transforms CIR dynamics for linear regression.
        """
        if len(data) < 2:
            raise ValueError("LSQ estimation requires at least 2 data points.")

        param_ses = self._fit_lsq_internal(data, dt)
        return param_ses

    def _fit_lsq_internal(self, data: pd.Series, dt: float) -> dict:
        """Internal LSQ implementation."""
        x = data.values
        n = len(x) - 1

        x_t = x[:-1]
        x_next = x[1:]
        dx = x_next - x_t

        # Regression: dx = alpha + beta * x_t + epsilon
        # where alpha = kappa * theta * dt, beta = -kappa * dt
        X_reg = np.vstack([np.ones(n), x_t]).T
        beta_hat, residuals, rank, s = np.linalg.lstsq(X_reg, dx, rcond=None)

        alpha, beta = beta_hat[0], beta_hat[1]

        # Map to CIR parameters
        self.kappa = max(1e-6, -beta / dt)
        self.theta = alpha / (self.kappa * dt) if self.kappa > 1e-6 else np.mean(x)

        # Estimate sigma from residual variance
        epsilon = dx - (alpha + beta * x_t)
        sigma_sq_eps = np.var(epsilon)

        # For CIR: Var(epsilon) approx sigma^2 * E[X_t] * dt
        mean_x = np.mean(x_t)
        sigma_sq = sigma_sq_eps / (mean_x * dt) if mean_x > 0 else sigma_sq_eps / dt
        self.sigma = max(1e-6, np.sqrt(sigma_sq))

        # Compute standard errors
        try:
            cov_beta = np.linalg.inv(X_reg.T @ X_reg) * sigma_sq_eps
            se_alpha = np.sqrt(cov_beta[0, 0])
            se_beta = np.sqrt(cov_beta[1, 1])

            # Delta method for kappa and theta
            se_kappa = np.abs(-1 / dt) * se_beta
            se_theta = se_alpha / (self.kappa * dt) if self.kappa > 1e-6 else np.nan
            se_sigma = self.sigma / np.sqrt(2 * n)
        except np.linalg.LinAlgError:
            se_kappa, se_theta, se_sigma = np.nan, np.nan, np.nan

        return {'kappa': se_kappa, 'theta': se_theta, 'sigma': se_sigma}

    def _compute_standard_errors(self, result, param_names: list) -> dict:
        """Computes standard errors from optimisation result."""
        param_ses = {}
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if callable(getattr(result.hess_inv, 'todense', None)):
                cov_matrix = result.hess_inv.todense()
            else:
                cov_matrix = result.hess_inv

            if cov_matrix.shape == (len(param_names), len(param_names)):
                for i, name in enumerate(param_names):
                    param_ses[name] = np.sqrt(max(0, cov_matrix[i, i]))
            else:
                for name in param_names:
                    param_ses[name] = np.nan
        else:
            for name in param_names:
                param_ses[name] = np.nan

        return param_ses

    def sample(self, n_paths: int = 1, horizon: int = 1, dt: float = None) -> KestrelResult:
        """
        Simulates future paths using Euler-Maruyama method.

        Uses reflection at zero to maintain positivity.

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
            Simulation results (all paths non-negative).
        """
        if not self.is_fitted and any(p is None for p in [self.kappa, self.theta, self.sigma]):
            raise RuntimeError("Model must be fitted or initialised with parameters before sampling.")

        if dt is None:
            dt = self._dt_ if self.is_fitted and hasattr(self, '_dt_') else 1.0

        kappa = self.kappa_ if self.is_fitted else self.kappa
        theta = self.theta_ if self.is_fitted else self.theta
        sigma = self.sigma_ if self.is_fitted else self.sigma

        if any(p is None for p in [kappa, theta, sigma]):
            raise RuntimeError("CIR parameters must be set or estimated to sample.")

        paths = np.zeros((horizon + 1, n_paths))
        if self.is_fitted and hasattr(self, '_last_data_point'):
            initial_val = self._last_data_point
        else:
            initial_val = theta

        paths[0, :] = initial_val

        for t in range(horizon):
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_paths)
            sqrt_Xt = np.sqrt(np.maximum(0, paths[t, :]))
            paths[t + 1, :] = paths[t, :] + kappa * (theta - paths[t, :]) * dt + sigma * sqrt_Xt * dW
            # Reflection scheme for positivity
            paths[t + 1, :] = np.abs(paths[t + 1, :])

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)

    def feller_condition_satisfied(self) -> bool:
        """
        Checks if Feller condition (2*kappa*theta > sigma^2) is satisfied.

        Returns True if process is guaranteed to stay strictly positive.
        """
        kappa = self.kappa_ if self.is_fitted else self.kappa
        theta = self.theta_ if self.is_fitted else self.theta
        sigma = self.sigma_ if self.is_fitted else self.sigma

        if any(p is None for p in [kappa, theta, sigma]):
            return False

        return 2 * kappa * theta > sigma ** 2
