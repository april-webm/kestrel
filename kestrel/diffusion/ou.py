# kestrel/diffusion/ou.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from kestrel.base import StochasticProcess
from kestrel.utils.kestrel_result import KestrelResult

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
    def __init__(self, theta: float = None, mu: float = None, sigma: float = None):
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def fit(self, data: pd.Series, dt: float = None, method: str = 'mle', freq: str = None):
        """
        Estimates (theta, mu, sigma) parameters from time-series data.

        Args:
            data (pd.Series): Time-series data for fitting.
            dt (float, optional): Time step between observations.
                                   If None, inferred from data; else defaults to 1.0.
            method (str): Estimation method: 'mle' (Exact Maximum Likelihood) or
                          'ar1' (AR(1) regression).
            freq (str, optional): Data frequency if `data` has `DatetimeIndex`.
                                  e.g., 'B' (business day), 'D' (calendar day).
                                  Converts `dt` to annual basis. Inferred if None.
        Raises:
            ValueError: Input data not pandas Series or unknown estimation method.
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series.")

        if dt is None:
            if isinstance(data.index, pd.DatetimeIndex):
                # Need at least two points for time difference from index
                if len(data.index) < 2: 
                    dt = 1.0
                    print("dt not provided, DatetimeIndex has less than 2 points. Defaulting to 1.0.")
                else:
                    inferred_timedelta = data.index[1] - data.index[0]
                    # Determine frequency to convert timedelta to annual dt
                    current_freq = freq # Use provided freq
                    if current_freq is None:
                        if data.index.freq:
                            current_freq = data.index.freq
                            print(f"Using frequency from DatetimeIndex.freq: {current_freq}")
                        else:
                            inferred_freq = pd.infer_freq(data.index)
                            if inferred_freq:
                                current_freq = inferred_freq
                                print(f"Inferred frequency from DatetimeIndex: {current_freq}")
                            else:
                                print("Could not infer frequency from DatetimeIndex. Defaulting to business day ('B') for dt conversion.")
                                current_freq = 'B' # Default if inference fails

                    # Convert timedelta to numerical dt based on current_freq
                    if current_freq in ['B', 'C', 'D']: # Business day, Custom Business Day, Calendar Day
                        dt = inferred_timedelta / pd.Timedelta(days=252.0)
                    elif current_freq in ['W', 'W-SUN', 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT']: # Weekly
                        dt = inferred_timedelta / pd.Timedelta(weeks=52)
                    elif current_freq in ['M', 'MS', 'BM', 'BMS']: # Monthly
                        dt = inferred_timedelta / pd.Timedelta(days=365/12)
                    elif current_freq in ['Q', 'QS', 'BQ', 'BQS']: # Quarterly
                        dt = inferred_timedelta / pd.Timedelta(days=365/4)
                    elif current_freq in ['A', 'AS', 'BA', 'BAS', 'Y', 'YS', 'BY', 'BYS']: # Annual
                        dt = inferred_timedelta / pd.Timedelta(days=365)
                    else: # Fallback for other frequencies, or if conversion ambiguous
                        dt = inferred_timedelta.total_seconds() / (365 * 24 * 3600)
                        print(f"Using total seconds for dt conversion for frequency '{current_freq}'. Consider explicit dt.")

                    if dt == 0: # Handle zero time_diffs
                        dt = 1.0
                    print(f"Inferred dt from DatetimeIndex: {dt} (annualized based on '{current_freq}' frequency)")
            else: # Not DatetimeIndex
                dt = 1.0
                print("dt not provided; cannot infer from index. Defaulting to 1.0.")
        
        self.dt_ = dt # Store dt used for fitting

        if method == 'mle':
            self._fit_mle(data, dt, freq)
        elif method == 'ar1':
            self._fit_ar1(data, dt, freq)
        else:
            raise ValueError(f"Unknown estimation method: {method}. Choose 'mle' or 'ar1'.")


    def _fit_mle(self, data: pd.Series, dt: float, freq: str = None):
        """
        Estimates OU parameters using Exact Maximum Likelihood Estimation.
        Method based on transition density.
        """
        if len(data) < 2:
            raise ValueError("MLE estimation requires at least 2 data points.")
        x = data.values
        n = len(x)
        
        # Initial parameter guess
        # Based on approximate moment matching / AR(1) regression for start values
        dx = np.diff(x)

        # Approximate AR(1) parameters for initial guess
        # Regression: dx = a + b*x + epsilon
        X_reg = np.vstack([np.ones(n - 1), x[:-1]]).T
        beta_hat = np.linalg.lstsq(X_reg, dx, rcond=None)[0]
        a, b = beta_hat[0], beta_hat[1]

        # Map AR(1) to OU for initial guess
        theta_0 = -b / dt
        mu_0 = -a / b if b != 0 else np.mean(x) # Fallback if b near zero
        
        # Estimate sigma from residuals
        residuals = dx - (a + b * x[:-1])
        sigma_sq_0 = np.var(residuals) / dt # Rough initial sigma^2
        sigma_0 = np.sqrt(sigma_sq_0) if sigma_sq_0 > 0 else 0.1

        # Ensure theta_0 positive
        theta_0 = max(0.01, theta_0)

        initial_params = [theta_0, mu_0, sigma_0]

        # Bounds for parameters: theta > 0, sigma > 0
        bounds = [(1e-6, None), (None, None), (1e-6, None)]

        # Use L-BFGS-B method as it supports bounds and can return approximation of inverse Hessian
        result = minimize(self._log_likelihood_ou, initial_params, args=(x, dt), bounds=bounds, method='L-BFGS-B')

        if result.success:
            self.theta, self.mu, self.sigma = result.x
            
            param_ses = {}
            # Calculate standard errors from the inverse Hessian (covariance matrix approximation)
            # The inverse Hessian is returned as result.hess_inv for 'L-BFGS-B'
            if hasattr(result, 'hess_inv') and result.hess_inv is not None:
                # Convert hess_inv (LinearOperator) to a dense matrix if it's not already
                # hess_inv can be either a dense array or a LinearOperator
                if callable(getattr(result.hess_inv, 'todense', None)):
                    cov_matrix = result.hess_inv.todense()
                else: # Assume it's already a dense matrix or array
                    cov_matrix = result.hess_inv
                
                # Check if cov_matrix is a square matrix of expected size
                if cov_matrix.shape == (len(initial_params), len(initial_params)):
                    param_ses = {
                        'theta': np.sqrt(cov_matrix[0, 0]),
                        'mu': np.sqrt(cov_matrix[1, 1]),
                        'sigma': np.sqrt(cov_matrix[2, 2]),
                    }
                else:
                    print("Warning: Hessian inverse shape mismatch, cannot calculate standard errors.")
            else:
                print("Warning: Could not retrieve Hessian inverse for standard error calculation.")
        else:
            raise RuntimeError(f"MLE optimization failed: {result.message}")
        
        # Store parameters and standard errors
        self._set_params(last_data_point=data.iloc[-1], theta=self.theta, mu=self.mu, sigma=self.sigma, 
                         dt=self.dt_, freq=freq, param_ses=param_ses)

    def _log_likelihood_ou(self, params, x, dt):
        """
        Negative log-likelihood function for OU process (Exact MLE).
        """
        theta, mu, sigma = params
        n = len(x)

        if theta <= 0 or sigma <= 0:
            return np.inf # Penalise invalid parameters

        # Pre-calculate terms
        exp_theta_dt = np.exp(-theta * dt)
        one_minus_exp_theta_dt = 1 - exp_theta_dt
        
        # Variance of conditional distribution X_t | X_{t-1}
        variance_conditional = sigma**2 * (1 - exp_theta_dt**2) / (2 * theta)

        # Numerical stability check for variance_conditional
        if variance_conditional <= 1e-12: # Small positive threshold
            return np.inf

        log_variance_conditional = np.log(variance_conditional)
        log_2pi = np.log(2 * np.pi)

        ll = 0.0
        for i in range(1, n):
            mean_conditional = x[i-1] * exp_theta_dt + mu * one_minus_exp_theta_dt
            
            # Normal PDF log-likelihood part
            ll += -0.5 * (log_variance_conditional + log_2pi)
            ll += -0.5 * ((x[i] - mean_conditional)**2 / variance_conditional)
        
        return -ll # Minimise negative log-likelihood

    def _fit_ar1(self, data: pd.Series, dt: float, freq: str = None):
        """
        Estimates OU parameters using AR(1) regression.
        Maps coefficients back to continuous-time parameters.
        """
        if len(data) < 2:
            raise ValueError("AR(1) regression requires at least 2 data points.")
        x_t = data.iloc[:-1].values
        x_t_plus_dt = data.iloc[1:].values
        n = len(x_t)

        # Linear regression: x_{t+dt} = c + phi * x_t + epsilon
        X_reg = np.vstack([np.ones(n), x_t]).T
        
        # Robust LSQ
        beta_hat, ss_residuals, rank, s = np.linalg.lstsq(X_reg, x_t_plus_dt, rcond=None)
        
        c, phi = beta_hat[0], beta_hat[1]

        # Calculate standard errors for c and phi
        # Residual variance: sigma_epsilon_sq
        sigma_epsilon_sq = ss_residuals[0] / (n - rank) if (n - rank) > 0 else np.var(x_t_plus_dt)
        
        # Covariance matrix for beta_hat (c, phi)
        # (X_reg^T X_reg)^-1 * sigma_epsilon_sq
        try:
            cov_beta = np.linalg.inv(X_reg.T @ X_reg) * sigma_epsilon_sq
            se_c = np.sqrt(cov_beta[0, 0])
            se_phi = np.sqrt(cov_beta[1, 1])
            cov_c_phi = cov_beta[0, 1]
        except np.linalg.LinAlgError:
            print("Warning: Could not invert (X_reg.T @ X_reg) for AR(1) standard errors.")
            se_c, se_phi, cov_c_phi = np.nan, np.nan, np.nan

        param_ses = {}

        # Map AR(1) coefficients to OU parameters
        if phi >= 1.0 - 1e-6: # Check for stationarity / near-unit root
            print("Warning: AR(1) coefficient phi >= 1.0. OU process may not be stationary / mean-reverting. Setting theta small positive.")
            self.theta = 1e-6 # Set small positive theta
            self.mu = np.mean(data) # Long-term mean is data mean
            self.sigma = 0.1 # Default sigma
            # Assign NaNs for SE as parameters are defaulted
            param_ses['theta'] = np.nan
            param_ses['mu'] = np.nan
            param_ses['sigma'] = np.nan
        else:
            # Ensure theta positive. If phi > 1, log(phi) > 0, theta negative.
            # If phi negative, log(phi) complex. Assume phi between 0 and 1 for stationary OU.
            if phi <= 0:
                print("Warning: Inferred AR(1) coefficient phi non-positive. Setting theta small positive, mu to data mean.")
                self.theta = 1e-6
                self.mu = np.mean(data)
                self.sigma = 0.1 # Default sigma
                param_ses['theta'] = np.nan
                param_ses['mu'] = np.nan
                param_ses['sigma'] = np.nan
            else:
                self.theta = -np.log(phi) / dt
                self.mu = c / (1 - phi)
                
                # Estimate sigma from residuals (same as before)
                epsilon_t = x_t_plus_dt - (c + phi * x_t)
                sigma_epsilon_sq = np.var(epsilon_t)
                
                if self.theta > 0 and (1 - np.exp(-2 * self.theta * dt)) > 0:
                    sigma_sq_ou = (sigma_epsilon_sq * 2 * self.theta) / (1 - np.exp(-2 * self.theta * dt))
                    self.sigma = np.sqrt(sigma_sq_ou)
                else:
                    self.sigma = np.sqrt(sigma_epsilon_sq / dt) # Fallback if theta near zero
                
                if self.sigma <= 0:
                    self.sigma = 0.1 # Ensure sigma positive

                # Calculate standard errors for OU parameters using Delta Method approximations
                # d(theta)/d(phi) = -1 / (phi * dt)
                # d(mu)/d(c) = 1 / (1 - phi)
                # d(mu)/d(phi) = c / (1 - phi)^2

                d_theta_d_phi = -1 / (phi * dt)
                se_theta = np.abs(d_theta_d_phi) * se_phi if not np.isnan(se_phi) else np.nan
                
                d_mu_d_c = 1 / (1 - phi)
                d_mu_d_phi = c / ((1 - phi)**2)
                
                # Variance of mu: Var(mu) = (d(mu)/dc)^2 Var(c) + (d(mu)/dphi)^2 Var(phi) + 2 (d(mu)/dc)(d(mu)/dphi)Cov(c,phi)
                # Assuming Var(c) = se_c^2, Var(phi) = se_phi^2
                if not np.isnan(se_c) and not np.isnan(se_phi) and not np.isnan(cov_c_phi):
                    se_mu_sq = (d_mu_d_c**2 * se_c**2) + (d_mu_d_phi**2 * se_phi**2) + (2 * d_mu_d_c * d_mu_d_phi * cov_c_phi)
                    se_mu = np.sqrt(se_mu_sq) if se_mu_sq >= 0 else np.nan
                else:
                    se_mu = np.nan

                # A simple approximation for sigma SE (might need more rigorous derivation)
                se_sigma = self.sigma * np.sqrt(sigma_epsilon_sq / (2 * n * dt * self.theta)) if self.theta > 0 else np.nan


                param_ses = {
                    'theta': se_theta,
                    'mu': se_mu,
                    'sigma': se_sigma
                }

        # Store parameters and standard errors
        self._set_params(last_data_point=data.iloc[-1], theta=self.theta, mu=self.mu, sigma=self.sigma,
                         dt=self.dt_, freq=freq, param_ses=param_ses)

    def sample(self, n_paths: int = 1, horizon: int = 1, dt: float = None) -> KestrelResult:
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
            initial_val = mu  # Start at long-term mean if not fitted
            paths[0, :] = initial_val

        # Euler-Maruyama simulation
        for t in range(horizon):
            dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_paths)
            paths[t + 1, :] = paths[t, :] + theta * (mu - paths[t, :]) * dt + sigma * dW

        return KestrelResult(pd.DataFrame(paths), initial_value=initial_val)