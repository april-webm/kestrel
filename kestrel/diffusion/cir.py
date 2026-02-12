# kestrel/diffusion/cir.py
"""Cox-Ingersoll-Ross process implementation."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from kestrel.base import StochasticProcess
from kestrel.utils.kestrel_result import KestrelResult
from kestrel.utils.warnings import FellerConditionWarning


class CIRProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross (CIR) Process.

    Models mean-reverting dynamics with state-dependent volatility.
    SDE: dX_t = kappa * (theta - X_t) dt + sigma * sqrt(X_t) dW_t

    Feller condition (2*kappa*theta >= sigma^2) ensures strict positivity.

    Parameters
    ----------
    kappa : float, optional
        Rate of mean reversion.
    theta : float, optional
        Long-term mean level.
    sigma : float, optional
        Volatility coefficient.
    """

    kappa: Optional[float]
    theta: Optional[float]
    sigma: Optional[float]

    def __init__(self, kappa: Optional[float] = None, theta: Optional[float] = None, sigma: Optional[float] = None) -> None:
        super().__init__()
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def fit(
        self,
        data: pd.Series,
        dt: Optional[float] = None,
        method: str = 'mle',
        bias_correction: bool = False,
        feller_constraint: bool = False,
        regularization: Optional[float] = None,
    ) -> KestrelResult:
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
        bias_correction : bool
            If True, corrects finite-sample bias in kappa. For 'mle', applies
            analytical first-order correction kappa * n/(n+3). For 'lsq',
            applies delete-1 jackknife. Default False (more expensive than OU).
        feller_constraint : bool
            If True, constrain the optimizer to respect the Feller condition
            (2*kappa*theta > sigma^2). For 'mle', uses constrained optimization;
            for 'lsq', applies post-hoc projection.
        regularization : float, optional
            L2 penalty strength on kappa and theta. Added to the negative
            log-likelihood during MLE optimisation.

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

        # CIR requires strictly positive data
        if (data <= 0).any():
            raise ValueError("CIR process requires strictly positive data.")

        if dt is None:
            dt = self._infer_dt(data)

        self.dt_ = dt
        n_obs = len(data) - 1

        if method == 'mle':
            kappa, theta, sigma, param_ses, log_likelihood, residuals = self._fit_mle(
                data, dt, feller_constraint=feller_constraint, regularization=regularization
            )
            if bias_correction:
                # Analytical first-order correction for MLE
                n = len(data) - 1
                kappa = float(kappa * n / (n + 3))
                # self._bias_corrected_ = True # No longer set on self
            else:
                pass
                # self._bias_corrected_ = False # No longer set on self
        elif method == 'lsq':
            kappa, theta, sigma, param_ses, log_likelihood, residuals = self._fit_lsq(
                data, dt, feller_constraint=feller_constraint
            )
            if bias_correction:
                kappa = self._apply_lsq_jackknife_bias_correction(data, dt, kappa, theta, sigma)
                # self._bias_corrected_ = True # No longer set on self
            else:
                pass
                # self._bias_corrected_ = False # No longer set on self
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle' or 'lsq'.")

        self.kappa = kappa # Store on self for sample method if not passed explicitly
        self.theta = theta
        self.sigma = sigma

        self._post_fit_setup(
            last_data_point=data.iloc[-1],
            dt=self.dt_,
            freq=None
        )

        # Check Feller condition and warn if violated
        feller_satisfied = self.feller_condition_satisfied(kappa, theta, sigma)
        if not feller_satisfied:
            feller_lhs = 2 * kappa * theta
            feller_rhs = sigma ** 2
            warnings.warn(
                f"Fitted CIR parameters violate the Feller condition: "
                f"2*kappa*theta={feller_lhs:.6f} <= sigma^2={feller_rhs:.6f}. "
                f"The process can reach zero, which may invalidate downstream uses "
                f"(e.g., interest rate modelling).",
                FellerConditionWarning,
                stacklevel=2,
            )

        return KestrelResult(
            process_name="CIRProcess",
            params={'kappa': kappa, 'theta': theta, 'sigma': sigma},
            param_ses=param_ses,
            log_likelihood=log_likelihood,
            residuals=residuals,
            n_obs=n_obs,
        )

    def _fit_mle(
        self, data: pd.Series, dt: float, feller_constraint: bool = False, regularization: Optional[float] = None
    ) -> Tuple[float, float, float, Dict[str, float], float, np.ndarray]:
        """
        Estimates CIR parameters using Maximum Likelihood.

        Uses Gaussian approximation to conditional distribution for tractability.
        When feller_constraint=True, uses SLSQP with an inequality constraint
        enforcing 2*kappa*theta > sigma^2.
        """
        if len(data) < 2:
            raise ValueError("MLE estimation requires at least 2 data points.")

        x = data.values
        n = len(x)

        # Initial estimates via LSQ for starting values
        x_t = x[:-1]
        x_next = x[1:]
        dx = x_next - x_t

        mean_x = np.mean(x)
        var_dx = np.var(dx)

        kappa_0 = max(0.1, -np.cov(dx, x_t)[0, 1] / (np.var(x_t) * dt))
        theta_0 = mean_x
        sigma_0 = max(0.01, np.sqrt(var_dx / (mean_x * dt)))

        initial_params = [kappa_0, theta_0, sigma_0]
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]

        reg = regularization if regularization is not None else 0.0

        if feller_constraint:
            constraints = [{
                'type': 'ineq',
                'fun': lambda p: 2 * p[0] * p[1] - p[2] ** 2 - 1e-8,
            }]
            result = minimize(
                self._neg_log_likelihood_and_residuals,
                initial_params,
                args=(x, dt, reg, True), # Pass True to return only NLL
                bounds=bounds,
                method='SLSQP',
                constraints=constraints,
            )
        else:
            result = minimize(
                self._neg_log_likelihood_and_residuals,
                initial_params,
                args=(x, dt, reg, True), # Pass True to return only NLL
                bounds=bounds,
                method='L-BFGS-B',
            )
        
        kappa, theta, sigma = np.nan, np.nan, np.nan
        if result.success:
            kappa, theta, sigma = result.x
        else:
            # Fallback to LSQ estimates
            kappa, theta, sigma, _, _, _ = self._fit_lsq_internal(data, dt)


        # Re-evaluate NLL and get residuals with optimal params
        nll, residuals = self._neg_log_likelihood_and_residuals(
            [kappa, theta, sigma], x, dt, reg, False # Pass False to return NLL and residuals
        )
        log_likelihood = -nll

        # Compute standard errors from inverse Hessian
        param_ses = self._compute_standard_errors(result, ['kappa', 'theta', 'sigma'])
        return kappa, theta, sigma, param_ses, log_likelihood, residuals

    def _neg_log_likelihood_and_residuals(self, params: List[float], x: np.ndarray, dt: float, reg: float = 0.0, only_nll: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        Negative log-likelihood for CIR using Gaussian approximation.

        Conditional distribution X_{t+dt} | X_t approximated as Gaussian
        with mean and variance from Euler discretisation.

        When reg > 0, adds L2 penalty: reg * (kappa^2 + theta^2).
        """
        kappa, theta, sigma = params

        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return np.inf, None

        n = len(x)
        ll = 0.0
        residuals_list = []

        for i in range(1, n):
            x_prev = x[i - 1]
            x_curr = x[i]

            # Conditional mean: E[X_{t+dt} | X_t] = X_t * exp(-kappa*dt) + theta*(1 - exp(-kappa*dt))
            exp_kdt = np.exp(-kappa * dt)
            mean_cond = x_prev * exp_kdt + theta * (1 - exp_kdt)

            # Conditional variance (Euler approximation)
            var_cond = (sigma ** 2 * x_prev * (1 - exp_kdt ** 2)) / (2 * kappa)

            if var_cond <= 1e-12:
                # If variance is too small, assume log-likelihood is -inf for this point
                ll += -1e10 # A large negative number to effectively make NLL inf
                if not only_nll:
                    residuals_list.append(np.nan)
                continue

            # Gaussian log-likelihood contribution
            log_pdf_val = stats.norm.logpdf(x_curr, loc=mean_cond, scale=np.sqrt(var_cond))
            if np.isinf(log_pdf_val): # Handle cases where logpdf returns -inf
                ll += -1e10
            else:
                ll += log_pdf_val

            if not only_nll:
                residuals_list.append((x_curr - mean_cond) / np.sqrt(var_cond))

        nll = -ll
        if reg > 0:
            nll += reg * (kappa ** 2 + theta ** 2)
        
        if only_nll:
            return nll, None
        else:
            return nll, np.array(residuals_list)

    def _fit_lsq(self, data: pd.Series, dt: float, feller_constraint: bool = False) -> Tuple[float, float, float, Dict[str, float], float, np.ndarray]:
        """
        Estimates CIR parameters using Least Squares regression.

        Transforms CIR dynamics for linear regression. When feller_constraint=True,
        applies post-hoc projection to ensure 2*kappa*theta > sigma^2.
        """
        if len(data) < 2:
            raise ValueError("LSQ estimation requires at least 2 data points.")

        kappa, theta, sigma, param_ses, log_likelihood, residuals = self._fit_lsq_internal(data, dt)

        # Post-hoc Feller projection: scale sigma down if needed
        if feller_constraint and not self.feller_condition_satisfied(kappa, theta, sigma):
            max_sigma = np.sqrt(2 * kappa * theta - 1e-8)
            if max_sigma > 0:
                sigma = float(max_sigma)
            else:
                # kappa*theta too small; cannot satisfy Feller
                sigma = 1e-6
            # Recalculate log-likelihood and residuals with adjusted sigma
            nll, residuals = self._neg_log_likelihood_and_residuals(
                [kappa, theta, sigma], data.values, dt, 0.0, False
            )
            log_likelihood = -nll

        return kappa, theta, sigma, param_ses, log_likelihood, residuals

    def _fit_lsq_internal(self, data: pd.Series, dt: float) -> Tuple[float, float, float, Dict[str, float], float, np.ndarray]:
        """Internal LSQ implementation."""
        x = data.values
        n = len(x) - 1

        x_t = x[:-1]
        x_next = x[1:]
        dx = x_next - x_t

        # Regression: dx = alpha + beta * x_t + epsilon
        # where alpha = kappa * theta * dt, beta = -kappa * dt
        X_reg = np.vstack([np.ones(n), x_t]).T
        beta_hat, residuals_ols, rank, s = np.linalg.lstsq(X_reg, dx, rcond=None)

        alpha, beta = beta_hat[0], beta_hat[1]

        # Map to CIR parameters
        kappa = max(1e-6, -beta / dt)
        theta = alpha / (kappa * dt) if kappa > 1e-6 else float(np.mean(x))

        # Estimate sigma from residual variance
        epsilon = dx - (alpha + beta * x_t)
        sigma_sq_eps = float(np.var(epsilon))

        # For CIR: Var(epsilon) approx sigma^2 * E[X_t] * dt
        mean_x = float(np.mean(x_t))
        sigma_sq = sigma_sq_eps / (mean_x * dt) if mean_x > 0 else sigma_sq_eps / dt
        sigma = max(1e-6, np.sqrt(sigma_sq))

        # Compute standard errors
        param_ses: Dict[str, float] = {}
        try:
            cov_beta = np.linalg.inv(X_reg.T @ X_reg) * sigma_sq_eps
            se_alpha = float(np.sqrt(cov_beta[0, 0]))
            se_beta = float(np.sqrt(cov_beta[1, 1]))

            # Delta method for kappa and theta
            se_kappa = float(np.abs(-1 / dt) * se_beta)
            se_theta = float(se_alpha / (kappa * dt)) if kappa > 1e-6 else np.nan
            se_sigma = float(sigma / np.sqrt(2 * n))
            param_ses = {'kappa': se_kappa, 'theta': se_theta, 'sigma': se_sigma}
        except np.linalg.LinAlgError:
            warnings.warn("Could not invert (X_reg.T @ X_reg) for standard errors in LSQ.", RuntimeWarning)
            param_ses = {'kappa': np.nan, 'theta': np.nan, 'sigma': np.nan}
        
        # Calculate log-likelihood and residuals based on final parameters
        nll, residuals = self._neg_log_likelihood_and_residuals(
            [kappa, theta, sigma], data.values, dt, 0.0, False
        )
        log_likelihood = -nll

        return kappa, theta, sigma, param_ses, log_likelihood, residuals

    def _compute_standard_errors(self, result: Any, param_names: List[str]) -> Dict[str, float]:
        """Computes standard errors from optimisation result."""
        param_ses: Dict[str, float] = {}
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            if callable(getattr(result.hess_inv, 'todense', None)):
                cov_matrix = result.hess_inv.todense()
            else:
                cov_matrix = result.hess_inv

            if cov_matrix.shape == (len(param_names), len(param_names)):
                for i, name in enumerate(param_names):
                    param_ses[name] = float(np.sqrt(max(0, cov_matrix[i, i])))
            else:
                for name in param_names:
                    param_ses[name] = np.nan
        else:
            for name in param_names:
                param_ses[name] = np.nan

        return param_ses

    def _apply_lsq_jackknife_bias_correction(self, data: pd.Series, dt: float, kappa_full: float, theta_full: float, sigma_full: float) -> float:
        """
        Delete-1 jackknife bias correction for CIR LSQ estimates.

        Removes each transition, refits LSQ, and applies the standard
        jackknife formula: param_jk = n*param_full - (n-1)*mean(param_loo).
        """
        x = data.values
        n = len(x) - 1
        x_t = x[:-1]
        dx = x[1:] - x_t

        kappa_loo = np.empty(n)
        valid = np.ones(n, dtype=bool)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            x_t_i = x_t[mask]
            dx_i = dx[mask]

            X_reg_i = np.vstack([np.ones(n - 1), x_t_i]).T
            beta_hat_i, _, _, _ = np.linalg.lstsq(X_reg_i, dx_i, rcond=None)
            alpha_i, beta_i = beta_hat_i[0], beta_hat_i[1]

            kappa_i = max(1e-6, -beta_i / dt)
            if kappa_i > 1e-6:
                kappa_loo[i] = kappa_i
            else:
                valid[i] = False

        n_valid = valid.sum()
        if n_valid < n * 0.5:
            return kappa_full

        kappa_jk = float(max(1e-6, n * kappa_full - (n - 1) * np.mean(kappa_loo[valid])))
        return kappa_jk

    def sample(self, n_paths: int = 1, horizon: int = 1, dt: Optional[float] = None) -> KestrelResult:
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

    def feller_condition_satisfied(self, kappa: Optional[float] = None, theta: Optional[float] = None, sigma: Optional[float] = None) -> bool:
        """
        Checks if Feller condition (2*kappa*theta > sigma^2) is satisfied.

        Returns True if process is guaranteed to stay strictly positive.
        """
        kappa = self.kappa_ if kappa is None and self.is_fitted else kappa
        theta = self.theta_ if theta is None and self.is_fitted else theta
        sigma = self.sigma_ if sigma is None and self.is_fitted else sigma

        if any(p is None for p in [kappa, theta, sigma]):
            return False

        return 2 * kappa * theta > sigma ** 2