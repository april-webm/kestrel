# kestrel/jump_diffusion/merton.py
"""Merton Jump Diffusion process implementation."""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
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

    def fit(self, data: pd.Series, dt: float = None, method: str = 'mle', regularization: float = None):
        """
        Estimates parameters from log-return time-series data.

        Parameters
        ----------
        data : pd.Series
            Log-returns (or price levels, which are converted to log-returns).
        dt : float, optional
            Time step between observations.
        method : str
            Estimation method: 'mle' only currently supported.
        regularization : float, optional
            L2 penalty strength on drift parameter mu.

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
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle'.")

        self._set_params(
            last_data_point=data.iloc[-1],
            mu=self.mu,
            sigma=self.sigma,
            lambda_=self.lambda_,
            jump_mu=self.jump_mu,
            jump_sigma=self.jump_sigma,
            dt=self.dt_,
            param_ses=param_ses
        )

    def _fit_mle(self, data: pd.Series, dt: float, regularization: float = None) -> dict:
        """
        Estimates Merton parameters using Maximum Likelihood.

        Uses mixture density approach where observation density is weighted
        sum over possible jump counts.
        """
        if len(data) < 10:
            raise ValueError("MLE estimation requires at least 10 data points.")

        returns = data.values
        n = len(returns)

        # Initial parameter estimates from moments
        mean_r = np.mean(returns)
        var_r = np.var(returns)
        skew_r = self._skewness(returns)
        kurt_r = self._kurtosis(returns)

        # Method of moments initial guess
        # Excess kurtosis suggests jumps; higher kurtosis = more/larger jumps
        excess_kurt = max(0, kurt_r - 3)

        if excess_kurt > 0.5:
            # Evidence of jumps
            lambda_0 = min(2.0, max(0.1, excess_kurt / 2))
            jump_sigma_0 = np.sqrt(var_r * 0.3)
            sigma_0 = np.sqrt(max(0.01, var_r * 0.7 / dt))
        else:
            # Weak evidence of jumps
            lambda_0 = 0.1
            jump_sigma_0 = np.sqrt(var_r * 0.1)
            sigma_0 = np.sqrt(max(0.01, var_r * 0.9 / dt))

        mu_0 = mean_r / dt
        jump_mu_0 = 0.0  # Symmetric jumps initially

        initial_params = [mu_0, sigma_0, lambda_0, jump_mu_0, jump_sigma_0]
        bounds = [
            (None, None),      # mu
            (1e-6, None),      # sigma
            (1e-6, 10.0),      # lambda_
            (None, None),      # jump_mu
            (1e-6, None)       # jump_sigma
        ]

        reg = regularization if regularization is not None else 0.0

        result = minimize(
            self._neg_log_likelihood,
            initial_params,
            args=(returns, dt, reg),
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 500}
        )

        if result.success:
            self.mu, self.sigma, self.lambda_, self.jump_mu, self.jump_sigma = result.x
        else:
            # Fallback to moment estimates
            self.mu = mu_0
            self.sigma = sigma_0
            self.lambda_ = lambda_0
            self.jump_mu = jump_mu_0
            self.jump_sigma = jump_sigma_0

        # Ensure sigma and jump_sigma are positive
        self.sigma = max(1e-6, self.sigma)
        self.lambda_ = max(1e-6, self.lambda_)
        self.jump_sigma = max(1e-6, self.jump_sigma)

        param_ses = self._compute_standard_errors(
            result,
            ['mu', 'sigma', 'lambda_', 'jump_mu', 'jump_sigma']
        )
        return param_ses

    def _neg_log_likelihood(self, params, returns, dt, reg=0.0):
        """
        Negative log-likelihood for Merton model.

        Density is mixture over Poisson jump counts:
        f(r) = sum_{k=0}^{K} P(N=k) * phi(r; mu_k, sigma_k^2)

        where mu_k = (mu - 0.5*sigma^2)*dt + k*jump_mu
              sigma_k^2 = sigma^2*dt + k*jump_sigma^2

        When reg > 0, adds L2 penalty: reg * mu^2.
        """
        mu, sigma, lambda_, jump_mu, jump_sigma = params

        if sigma <= 0 or lambda_ < 0 or jump_sigma <= 0:
            return np.inf

        n = len(returns)
        ll = 0.0

        # Truncate Poisson sum at reasonable number of jumps
        max_jumps = max(10, int(3 * lambda_ * dt + 5))

        for r in returns:
            density = 0.0

            for k in range(max_jumps + 1):
                # Poisson probability of k jumps
                poisson_prob = self._poisson_pmf(k, lambda_ * dt)

                # Conditional mean and variance given k jumps
                mean_k = (mu - 0.5 * sigma ** 2) * dt + k * jump_mu
                var_k = sigma ** 2 * dt + k * jump_sigma ** 2

                if var_k <= 0:
                    continue

                # Gaussian density contribution
                density += poisson_prob * norm.pdf(r, loc=mean_k, scale=np.sqrt(var_k))

            if density <= 1e-300:
                ll += -700  # Log of very small number
            else:
                ll += np.log(density)

        nll = -ll
        if reg > 0:
            nll += reg * mu ** 2
        return nll

    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Poisson probability mass function."""
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

        mu = self.mu_ if self.is_fitted else self.mu
        sigma = self.sigma_ if self.is_fitted else self.sigma
        lambda_ = self.lambda_ if self.is_fitted else self.lambda_
        jump_mu = self.jump_mu_ if self.is_fitted else self.jump_mu
        jump_sigma = self.jump_sigma_ if self.is_fitted else self.jump_sigma

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
        mu = self.mu_ if self.is_fitted else self.mu
        sigma = self.sigma_ if self.is_fitted else self.sigma
        lambda_ = self.lambda_ if self.is_fitted else self.lambda_
        jump_mu = self.jump_mu_ if self.is_fitted else self.jump_mu

        if any(p is None for p in [mu, sigma, lambda_, jump_mu]):
            raise RuntimeError("Parameters must be set or estimated first.")

        return mu - 0.5 * sigma ** 2 + lambda_ * jump_mu

    def total_variance(self) -> float:
        """
        Computes total variance per unit time.

        Var[r] = sigma^2 + lambda*(jump_mu^2 + jump_sigma^2)
        """
        sigma = self.sigma_ if self.is_fitted else self.sigma
        lambda_ = self.lambda_ if self.is_fitted else self.lambda_
        jump_mu = self.jump_mu_ if self.is_fitted else self.jump_mu
        jump_sigma = self.jump_sigma_ if self.is_fitted else self.jump_sigma

        if any(p is None for p in [sigma, lambda_, jump_mu, jump_sigma]):
            raise RuntimeError("Parameters must be set or estimated first.")

        return sigma ** 2 + lambda_ * (jump_mu ** 2 + jump_sigma ** 2)
