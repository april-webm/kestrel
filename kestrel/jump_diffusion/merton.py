# kestrel/jump_diffusion/merton.py
"""Merton Jump Diffusion process implementation."""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Any, Dict, List, Optional, Tuple

from kestrel.base import StochasticProcess
from kestrel.utils.kestrel_result import KestrelResult


class MertonProcess(StochasticProcess):
    """
    Merton Jump Diffusion Process.

    Combines Brownian motion with Poisson-distributed jumps.
    For log-returns: r_t = (mu - 0.5*sigma^2 - lambda*k) dt + sigma dW_t + J_t dN_t

    Where J_t ~ N(jump_mu, jump_sigma^2) and N_t is Poisson(lambda).

    Parameters
    ----------
    mu : float, optional
        Drift of continuous component.
    sigma : float, optional
        Volatility of continuous component.
    lambda_ : float, optional
        Jump intensity (expected jumps per unit time).
    jump_mu : float, optional
        Mean of jump size distribution.
    jump_sigma : float, optional
        Standard deviation of jump size distribution.
    """

    def __init__(self, mu: float = None, sigma: float = None,
                 lambda_: float = None, jump_mu: float = None, jump_sigma: float = None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.lambda_ = lambda_
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

    def fit(self, data: pd.Series, dt: float = None, method: str = 'em', regularization: float = None) -> KestrelResult:
        """
        Estimates parameters from log-return time-series data using EM algorithm.

        Parameters
        ----------
        data : pd.Series
            Log-returns (or price levels, which are converted to log-returns).
        dt : float, optional
            Time step between observations.
        method : str
            Estimation method: 'em' only currently supported.
        regularization : float, optional
            L2 penalty strength on drift parameter mu.

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
        n_obs = len(data)

        if method == 'em':
            mu, sigma, lambda_, jump_mu, jump_sigma, param_ses, log_likelihood, residuals = self._fit_em(data, dt, regularization=regularization)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'em'.")

        self.mu = mu # Store on self for sample method if not passed explicitly
        self.sigma = sigma
        self.lambda_ = lambda_
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

        self._post_fit_setup(
            last_data_point=data.iloc[-1],
            dt=self.dt_,
            freq=None
        )

        return KestrelResult(
            process_name="MertonProcess",
            params={
                'mu': mu,
                'sigma': sigma,
                'lambda_': lambda_,
                'jump_mu': jump_mu,
                'jump_sigma': jump_sigma
            },
            param_ses=param_ses,
            log_likelihood=log_likelihood,
            residuals=residuals,
            n_obs=n_obs,
        )

    def _fit_em(self, data: pd.Series, dt: float, regularization: float = None, max_iter: int = 100, tol: float = 1e-4) -> Tuple[float, float, float, float, float, Dict[str, float], float, np.ndarray]:
        """
        Estimates Merton parameters using the Expectation-Maximization (EM) algorithm.
        """
        if len(data) < 10:
            raise ValueError("EM estimation requires at least 10 data points.")

        returns = data.values
        n = len(returns)

        # Initial parameter estimates from moments (same as previous MLE)
        mean_r = np.mean(returns)
        var_r = np.var(returns)
        skew_r = self._skewness(returns)
        kurt_r = self._kurtosis(returns)

        excess_kurt = max(0, kurt_r - 3)

        if excess_kurt > 0.5:
            lambda_ = min(2.0, max(0.1, excess_kurt / 2))
            jump_sigma = np.sqrt(var_r * 0.3)
            sigma = np.sqrt(max(0.01, var_r * 0.7 / dt))
        else:
            lambda_ = 0.1
            jump_sigma = np.sqrt(var_r * 0.1)
            sigma = np.sqrt(max(0.01, var_r * 0.9 / dt))

        mu = mean_r / dt
        jump_mu = 0.0

        current_params = [mu, sigma, lambda_, jump_mu, jump_sigma]
        prev_log_likelihood = -np.inf # Initialize with a very small log-likelihood

        reg = regularization if regularization is not None else 0.0

        for i in range(max_iter):
            # E-step: Compute posterior probabilities P(k|r_j) for each observation r_j and jump count k
            log_likelihood, posterior_probs_matrix = self._e_step(returns, dt, current_params, reg)

            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

            # M-step: Update parameters based on posterior probabilities
            current_params = self._m_step(returns, dt, posterior_probs_matrix, current_params, reg)
            
            # Ensure positive constraints
            current_params[1] = max(1e-6, current_params[1]) # sigma
            current_params[2] = max(1e-6, current_params[2]) # lambda_
            current_params[4] = max(1e-6, current_params[4]) # jump_sigma

        mu, sigma, lambda_, jump_mu, jump_sigma = current_params

        # Compute standard errors (using numerical Hessian from an MLE pass if needed, or approximations)
        # For simplicity, we'll re-run a minimize to get Hessian for SEs.
        # This is not strictly part of EM but a practical way to get SEs.
        bounds = [
            (None, None),      # mu
            (1e-6, None),      # sigma
            (1e-6, 10.0),      # lambda_
            (None, None),      # jump_mu
            (1e-6, None)       # jump_sigma
        ]
        result_for_ses = minimize(
            self._neg_log_likelihood_vectorized, # Use vectorized version
            current_params,
            args=(returns, dt, reg, True), # only_nll=True
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 500}
        )
        param_ses = self._compute_standard_errors(
            result_for_ses,
            ['mu', 'sigma', 'lambda_', 'jump_mu', 'jump_sigma']
        )
        
        # Final log-likelihood and residuals after convergence
        final_log_likelihood, residuals = self._neg_log_likelihood_vectorized(
            current_params, returns, dt, reg, False # only_nll=False
        )
        final_log_likelihood = -final_log_likelihood

        return mu, sigma, lambda_, jump_mu, jump_sigma, param_ses, final_log_likelihood, residuals

    def _e_step(self, returns: np.ndarray, dt: float, params: List[float], reg: float) -> Tuple[float, np.ndarray]:
        """
        E-step: Computes the posterior probabilities P(k|r_j) for each observation r_j
        and possible jump count k.
        """
        mu, sigma, lambda_, jump_mu, jump_sigma = params
        n = len(returns)

        max_jumps = max(10, int(3 * lambda_ * dt + 5))
        
        # Pre-calculate Poisson probabilities
        poisson_probs = np.array([self._poisson_pmf(k, lambda_ * dt) for k in range(max_jumps + 1)])
        
        # Matrix to store conditional densities: P(r_j | k)
        conditional_densities = np.zeros((n, max_jumps + 1))
        
        for k in range(max_jumps + 1):
            mean_k = (mu - 0.5 * sigma ** 2) * dt + k * jump_mu
            var_k = sigma ** 2 * dt + k * jump_sigma ** 2

            # Handle cases where var_k might be problematic
            if var_k <= 1e-12:
                # If variance is too small, density is effectively zero for this k
                conditional_densities[:, k] = 1e-300 # A very small number
            else:
                conditional_densities[:, k] = norm.pdf(returns, loc=mean_k, scale=np.sqrt(var_k))

        # Joint densities: P(r_j, k) = P(r_j | k) * P(k)
        joint_densities = conditional_densities * poisson_probs[np.newaxis, :]
        
        # Marginal density: P(r_j) = sum_k P(r_j, k)
        marginal_densities = np.sum(joint_densities, axis=1)
        
        # Ensure no division by zero for marginal_densities, clamp at a small value
        marginal_densities[marginal_densities <= 1e-300] = 1e-300

        # Posterior probabilities: P(k | r_j) = P(r_j, k) / P(r_j)
        posterior_probs_matrix = joint_densities / marginal_densities[:, np.newaxis]
        
        # Calculate log-likelihood for convergence check
        log_likelihood = np.sum(np.log(marginal_densities))
        if reg > 0:
            log_likelihood -= reg * mu ** 2 # Subtract penalty to get effective likelihood

        return log_likelihood, posterior_probs_matrix

    def _m_step(self, returns: np.ndarray, dt: float, posterior_probs_matrix: np.ndarray, current_params: List[float], reg: float) -> List[float]:
        """
        M-step: Updates Merton parameters based on posterior probabilities.
        """
        mu_prev, sigma_prev, lambda_prev, jump_mu_prev, jump_sigma_prev = current_params
        n = len(returns)
        max_jumps = posterior_probs_matrix.shape[1] - 1

        # 1. Update lambda (jump intensity)
        # E[N_T] = lambda * T, where T = n * dt (approx for many observations)
        # Sum of expected jump counts: sum_j sum_k k * P(k|r_j)
        expected_total_jumps = np.sum(posterior_probs_matrix * np.arange(max_jumps + 1)[np.newaxis, :])
        lambda_ = expected_total_jumps / (n * dt)
        
        # 2. Update jump_mu and jump_sigma (jump size distribution)
        # These updates require sums weighted by P(k|r_j) and k.
        # This part is more complex as it's a weighted sum of returns-minus-diffusion part
        
        # We need a weighted average of (r_j - (mu - 0.5*sigma^2)*dt) and (r_j - (mu - 0.5*sigma^2)*dt - k*jump_mu)
        # This is where the M-step becomes a bit intricate without a full derivation here.
        # For simplicity for now, we'll make a pragmatic update.
        
        # Calculate "de-diffused" returns: returns_prime_j = r_j - (mu_prev - 0.5*sigma_prev^2)*dt
        # This needs to be correctly specified. The original term is mu * dt - 0.5 * sigma^2 * dt
        # So we should consider returns_j - (mu_diff * dt) to isolate jump effect
        
        mu_diff_term = (mu_prev - 0.5 * sigma_prev ** 2) * dt
        returns_minus_continuous = returns - mu_diff_term

        # Expected number of jumps * jump_mu for each observation, given current params
        expected_k = np.sum(posterior_probs_matrix * np.arange(max_jumps + 1)[np.newaxis, :], axis=1)

        # Update jump_mu: E[sum(J_i)] / E[k] (for observations where k > 0)
        # Numerator: sum_j sum_k P(k|r_j) * (r_j - continuous_drift_contribution) / k (if k > 0) * k
        weighted_jump_sums = np.sum(
            posterior_probs_matrix[:, 1:] * returns_minus_continuous[:, np.newaxis] / np.arange(1, max_jumps + 1)[np.newaxis, :],
            axis=1
        )
        # This simplification needs to be refined.

        # Simplified update for jump_mu based on expectation over jump sizes
        # E[J_k] = k * jump_mu. We are trying to find jump_mu.
        
        # Let's try a more robust approach for M-step, as it requires re-optimizing with weights.
        # The M-step generally involves solving weighted MLE problems for each component.
        # This can be done by constructing new "effective" data for each component.

        # For simplicity in this iteration, we'll try a weighted average for jump_mu and jump_sigma_sq
        # based on the expected jump counts and "de-diffused" returns.
        
        # Weighted sum of (returns_minus_continuous - k*jump_mu_prev) for k jumps
        # Effective jump sizes for updating jump_mu
        effective_jump_sizes_sum = np.sum(
            posterior_probs_matrix[:, 1:] * (returns_minus_continuous[:, np.newaxis] - np.arange(1, max_jumps + 1)[np.newaxis, :] * jump_mu_prev),
            axis=1
        )
        
        # Total expected number of jumps
        total_expected_k = np.sum(posterior_probs_matrix[:, 1:] * np.arange(1, max_jumps + 1)[np.newaxis, :])

        if total_expected_k > 1e-12:
            jump_mu = np.sum(effective_jump_sizes_sum) / total_expected_k + jump_mu_prev
        else:
            jump_mu = jump_mu_prev

        # Update jump_sigma (simplified from variance of effective jump sizes)
        # Weighted sum of (effective_jump_size - jump_mu)^2
        # This will require further thought to make rigorous for multiple jumps per observation.
        # For now, let's keep previous jump_sigma if the formula gets too complex.

        # A simpler M-step: treat the EM as a sequence of weighted MLEs for parameters.
        # Let's focus on mu, sigma, lambda first, and keep jump_mu, jump_sigma simple.

        # Let's simplify M-step for mu, sigma, lambda, jump_mu, jump_sigma.
        # From Cont & Tankov (2004) "Financial Modelling with Jump Processes" Chapter 10, section 10.3, EM for Merton.
        # The M-step updates are:
        # lambda_new = (1/T) * sum_j sum_k k * P(k|r_j)
        # jump_mu_new = (sum_j sum_k k * P(k|r_j) * r_j_prime_k) / (sum_j sum_k k * P(k|r_j))
        # where r_j_prime_k = r_j - (mu_prev - 0.5 * sigma_prev^2) * dt
        # and sigma_diff_new^2 = (sum_j sum_k P(k|r_j) * (r_j_prime_k - k*jump_mu_new)^2) / (n*dt)
        # The above is for Merton's original process where S_t = S_0 * exp(X_t)
        # For log-returns: r_t = (mu - 0.5*sigma^2)*dt + sigma*dW_t + J_t dN_t

        # Re-derive M-step for log-returns version.
        # E-step provides w_{jk} = P(k_j | r_j; current_params)
        # Then, we need to estimate mu, sigma, lambda, jump_mu, jump_sigma from `returns`
        # and `w_{jk}`.

        # For lambda:
        lambda_ = np.sum(posterior_probs_matrix * np.arange(max_jumps + 1)[np.newaxis, :]) / (n * dt)

        # For mu, sigma, jump_mu, jump_sigma: these require solving weighted equations.
        # This is where the complexity comes. For now, we will use simplified updates as originally prototyped.

        # Simplified update for mu and sigma (from diffusion part after removing expected jumps)
        # r_j_adjusted = r_j - sum_k P(k|r_j) * k * jump_mu_prev
        expected_jump_for_each_obs = np.sum(posterior_probs_matrix * np.arange(max_jumps + 1)[np.newaxis, :] * jump_mu_prev, axis=1)
        diffusion_returns = returns - expected_jump_for_each_obs

        mu_new = np.mean(diffusion_returns) / dt + 0.5 * sigma_prev ** 2
        sigma_new = np.sqrt(max(1e-6, np.var(diffusion_returns) / dt))

        # Simplified update for jump_mu and jump_sigma
        # Using a weighted average of `returns_j - diffusion_drift_part`
        # for `k > 0` and then taking weighted mean/variance of these.
        
        # Calculate conditional means and variances for Gaussian component given k jumps (same as E-step)
        mean_k_diffusion = (mu_prev - 0.5 * sigma_prev ** 2) * dt
        
        weighted_jump_returns = np.zeros_like(returns)
        weighted_jump_returns_sq = np.zeros_like(returns)
        total_weight_for_jump_stats = np.zeros_like(returns)

        # Iterate over k (jump counts)
        for k_val in range(1, max_jumps + 1):
            weights_k = posterior_probs_matrix[:, k_val]
            
            # contribution to jump_mu: (r_j - mean_k_diffusion)
            weighted_jump_returns += weights_k * (returns - mean_k_diffusion) / k_val
            weighted_jump_returns_sq += weights_k * ((returns - mean_k_diffusion) / k_val)**2
            total_weight_for_jump_stats += weights_k

        jump_mu_new = np.sum(weighted_jump_returns) / np.sum(total_weight_for_jump_stats) if np.sum(total_weight_for_jump_stats) > 1e-12 else jump_mu_prev

        # For jump_sigma, need to consider overall jump variance
        # A more rigorous derivation of M-step for jump_sigma would involve sum over (r_j - (mu_diff_new + k*jump_mu_new))^2
        # For current purpose, we will approximate sigma_new from the diffusion_returns,
        # and rely on the initial guess for jump_sigma if M-step is complex.

        jump_sigma_new_sq_num = 0.0
        jump_sigma_new_sq_den = 0.0

        for k_val in range(1, max_jumps + 1):
            weights_k = posterior_probs_matrix[:, k_val]
            
            diff_term = returns - (mean_k_diffusion + k_val * jump_mu_new)
            jump_sigma_new_sq_num += np.sum(weights_k * diff_term**2)
            jump_sigma_new_sq_den += np.sum(weights_k * k_val)
        
        jump_sigma_new = np.sqrt(max(1e-6, jump_sigma_new_sq_num / (dt * jump_sigma_new_sq_den) if jump_sigma_new_sq_den > 1e-12 else jump_sigma_prev**2))


        # Apply regularization to mu if active
        if reg > 0:
            mu_new = mu_new / (1 + 2 * reg * dt) # Simplified Ridge-like update
            
        return [mu_new, sigma_new, lambda_, jump_mu_new, jump_sigma_new]


    def _neg_log_likelihood_vectorized(self, params: List[float], returns: np.ndarray, dt: float, reg: float = 0.0, only_nll: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        Negative log-likelihood for Merton model (vectorized).

        Density is mixture over Poisson jump counts:
        f(r) = sum_{k=0}^{K} P(N=k) * phi(r; mu_k, sigma_k^2)

        where mu_k = (mu - 0.5*sigma^2)*dt + k*jump_mu
              sigma_k^2 = sigma^2*dt + k*jump_sigma^2

        When reg > 0, adds L2 penalty: reg * mu^2.
        """
        mu, sigma, lambda_, jump_mu, jump_sigma = params

        if sigma <= 0 or lambda_ < 0 or jump_sigma <= 0:
            return np.inf, None

        n = len(returns)

        # Truncate Poisson sum at reasonable number of jumps
        max_jumps = max(10, int(3 * lambda_ * dt + 5))
        k_values = np.arange(max_jumps + 1)

        # Pre-calculate Poisson probabilities for all k
        poisson_probs = np.exp(-lambda_ * dt) * ((lambda_ * dt) ** k_values) / np.array([math.factorial(int(x)) for x in k_values])
        poisson_probs[poisson_probs < 1e-300] = 1e-300 # Clamp small probabilities

        # Pre-calculate conditional means and variances for all k
        mean_k = (mu - 0.5 * sigma ** 2) * dt + k_values * jump_mu
        var_k = sigma ** 2 * dt + k_values * jump_sigma ** 2

        # Handle problematic variances by setting corresponding densities to a very small number
        problematic_vars = var_k <= 1e-12
        var_k[problematic_vars] = 1.0 # Use a dummy value to avoid NaN/Inf in sqrt/division
        
        # Calculate Gaussian densities for all returns and all k
        # conditional_densities[j, k] = norm.pdf(returns[j], loc=mean_k[k], scale=np.sqrt(var_k[k]))
        # This will be a matrix of shape (n, max_jumps + 1)
        conditional_densities = norm.pdf(returns[:, np.newaxis], loc=mean_k[np.newaxis, :], scale=np.sqrt(var_k)[np.newaxis, :])
        conditional_densities[:, problematic_vars] = 1e-300 # Set problematic densities to very small

        # Total density for each return (sum over k)
        # density_for_each_return[j] = sum_k (conditional_densities[j, k] * poisson_probs[k])
        density_for_each_return = np.sum(conditional_densities * poisson_probs[np.newaxis, :], axis=1)
        
        # Handle very small densities
        density_for_each_return[density_for_each_return <= 1e-300] = 1e-300

        ll = np.sum(np.log(density_for_each_return))

        nll = -ll
        if reg > 0:
            nll += reg * mu ** 2
        
        if only_nll:
            return nll, None
        else:
            # Residuals are defined as log-likelihood contribution for each observation
            residuals = np.log(density_for_each_return)
            return nll, residuals

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Poisson probability mass function."""
        # Using math.exp for single value, which is fine within a loop.
        # This function is now primarily used in _e_step where it's called in a loop for k_values.
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return np.exp(-lam) * (lam ** k) / math.factorial(k)

    def _skewness(self, x: np.ndarray) -> float:
        """Computes sample skewness."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std == 0:
            return 0.0
        return np.sum((x - mean) ** 3) / (n * std ** 3)

    def _kurtosis(self, x: np.ndarray) -> float:
        """Computes sample kurtosis (not excess)."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        if std == 0:
            return 3.0
        return np.sum((x - mean) ** 4) / (n * std ** 4)

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
        Simulates future paths of Merton Jump Diffusion.

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
            Simulation results containing log-return paths.
        """
        if not self.is_fitted and any(p is None for p in
                                       [self.mu, self.sigma, self.lambda_, self.jump_mu, self.jump_sigma]):
            raise RuntimeError("Model must be fitted or initialised with parameters before sampling.")

        if dt is None:
            dt = self._dt_ if self.is_fitted and hasattr(self, '_dt_') else 1.0

        mu = self.mu if self.is_fitted else self.mu
        sigma = self.sigma if self.is_fitted else self.sigma
        lambda_ = self.lambda_ if self.is_fitted else self.lambda_
        jump_mu = self.jump_mu if self.is_fitted else self.jump_mu
        jump_sigma = self.jump_sigma if self.is_fitted else self.jump_sigma

        if any(p is None for p in [mu, sigma, lambda_, jump_mu, jump_sigma]):
            raise RuntimeError("Merton parameters must be set or estimated to sample.")

        paths = np.zeros((horizon + 1, n_paths))
        if self.is_fitted and hasattr(self, '_last_data_point'):
            initial_val = self._last_data_point
        else:
            initial_val = 0.0

        paths[0, :] = initial_val

        for t in range(horizon):
            # Continuous component (Brownian motion)
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_paths)
            continuous_step = (mu - 0.5 * sigma ** 2) * dt + sigma * dW

            # Jump component (Poisson process with normal jump sizes)
            num_jumps = np.random.poisson(lambda_ * dt, size=n_paths)
            jump_sizes = np.zeros(n_paths)

            for i in range(n_paths):
                if num_jumps[i] > 0:
                    # Sum of normal jumps
                    individual_jumps = np.random.normal(
                        loc=jump_mu,
                        scale=jump_sigma,
                        size=num_jumps[i]
                    )
                    jump_sizes[i] = np.sum(individual_jumps)

            paths[t + 1, :] = paths[t, :] + continuous_step + jump_sizes

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)

    def expected_return(self) -> float:
        """
        Computes expected return per unit time.

        E[r] = mu - 0.5*sigma^2 + lambda*jump_mu
        """
        mu = self.mu if self.is_fitted else self.mu
        sigma = self.sigma if self.is_fitted else self.sigma
        lambda_ = self.lambda_ if self.is_fitted else self.lambda_
        jump_mu = self.jump_mu if self.is_fitted else self.jump_mu

        if any(p is None for p in [mu, sigma, lambda_, jump_mu]):
            raise RuntimeError("Parameters must be set or estimated first.")

        return mu - 0.5 * sigma ** 2 + lambda_ * jump_mu

    def total_variance(self) -> float:
        """
        Computes total variance per unit time.

        Var[r] = sigma^2 + lambda*(jump_mu^2 + jump_sigma^2)
        """
        sigma = self.sigma if self.is_fitted else self.sigma
        lambda_ = self.lambda_ if self.is_fitted else self.lambda_
        jump_mu = self.jump_mu if self.is_fitted else self.jump_mu
        jump_sigma = self.jump_sigma if self.is_fitted else self.jump_sigma

        if any(p is None for p in [sigma, lambda_, jump_mu, jump_sigma]):
            raise RuntimeError("Parameters must be set or estimated first.")

        return sigma ** 2 + lambda_ * (jump_mu ** 2 + jump_sigma ** 2)