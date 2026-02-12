# tests/test_ou_process.py
import pytest
import pandas as pd
import numpy as np
from kestrel.diffusion.ou import OUProcess
from kestrel.utils.kestrel_result import KestrelResult
import warnings

# --- Test Data Generation ---
@pytest.fixture
def sample_ou_data():
    """Generates sample Ornstein-Uhlenbeck process data for testing."""
    np.random.seed(42) # for reproducibility
    
    # Define OU parameters
    theta = 0.5  # Mean reversion speed
    mu = 10.0    # Long-term mean
    sigma = 1.0  # Volatility
    dt = 1/252   # Daily time step (e.g., trading days in a year)
    
    n_steps = 1000
    x0 = 12.0    # Starting value
    
    data = [x0]
    for _ in range(n_steps - 1):
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        xt = data[-1] + theta * (mu - data[-1]) * dt + sigma * dW
        data.append(xt)
    
    return pd.Series(data)

@pytest.fixture
def sample_ou_datetime_data():
    """Generates sample OU data with a DatetimeIndex."""
    np.random.seed(43)
    theta, mu, sigma, dt = 0.5, 10.0, 1.0, 1/252
    n_steps = 200
    x0 = 12.0
    
    data = [x0]
    for _ in range(n_steps - 1):
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))
        xt = data[-1] + theta * (mu - data[-1]) * dt + sigma * dW
        data.append(xt)
            
    dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='B') # Business day frequency
    return pd.Series(data, index=dates)

# --- OUProcess Initialization Tests ---
def test_ou_process_init_default():
    """Test default initialization of OUProcess."""
    ou = OUProcess()
    assert ou.theta is None
    assert ou.mu is None
    assert ou.sigma is None
    assert not ou.is_fitted

def test_ou_process_init_with_params():
    """Test initialization of OUProcess with parameters."""
    ou = OUProcess(theta=0.1, mu=5.0, sigma=0.5)
    assert ou.theta == 0.1
    assert ou.mu == 5.0
    assert ou.sigma == 0.5
    assert not ou.is_fitted

# --- Fit Method Tests (MLE) ---
def test_ou_process_fit_mle(sample_ou_data):
    """Test MLE fitting of OUProcess."""
    ou = OUProcess()
    result = ou.fit(sample_ou_data, dt=1/252, method='mle')
    
    assert ou.is_fitted
    assert result.params['theta'] is not None
    assert result.params['mu'] is not None
    assert result.params['sigma'] is not None
    assert ou._dt_ == pytest.approx(1/252) # Check internal dt is set
    assert result.param_ses['theta'] is not None # Check for standard error
    assert result.param_ses['mu'] is not None    # Check for standard error
    assert result.param_ses['sigma'] is not None # Check for standard error
    
    # Check if estimated parameters are reasonably close to true parameters (from sample_ou_data generation)
    # The actual values will vary due to stochastic nature, so we check for plausible ranges
    assert 0.1 < result.params['theta'] < 2.5 # True theta was 0.5
    assert 8.0 < result.params['mu'] < 12.0   # True mu was 10.0
    assert 0.5 < result.params['sigma'] < 1.5 # True sigma was 1.0
    assert result.param_ses['theta'] > 0 # Standard errors should be positive
    assert result.param_ses['mu'] > 0
    assert result.param_ses['sigma'] > 0
    
    # Check new KestrelResult fields
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(sample_ou_data) - 1

    # Check that the parameters are also stored on the model instance for sampling
    assert ou.theta == result.params['theta']
    assert ou.mu == result.params['mu']
    assert ou.sigma == result.params['sigma']

def test_ou_process_fit_mle_datetime_index(sample_ou_datetime_data):
    """Test MLE fitting with DatetimeIndex and inferred dt."""
    ou = OUProcess()
    result = ou.fit(sample_ou_datetime_data, method='mle', freq='B') # dt should be inferred
    
    assert ou.is_fitted
    assert result.params['theta'] is not None
    assert result.params['mu'] is not None
    assert result.params['sigma'] is not None
    assert ou._dt_ == pytest.approx(1/252, rel=0.1) # Check inferred dt is close
    assert result.param_ses['theta'] is not None # Check for standard error
    assert result.param_ses['mu'] is not None    # Check for standard error
    assert result.param_ses['sigma'] is not None # Check for standard error
    assert result.param_ses['theta'] > 0
    assert result.param_ses['mu'] > 0
    assert result.param_ses['sigma'] > 0

# --- Fit Method Tests (AR1) ---
def test_ou_process_fit_ar1(sample_ou_data):
    """Test AR(1) fitting of OUProcess."""
    ou = OUProcess()
    result = ou.fit(sample_ou_data, dt=1/252, method='ar1')
    
    assert ou.is_fitted
    assert result.params['theta'] is not None
    assert result.params['mu'] is not None
    assert result.params['sigma'] is not None
    assert ou._dt_ == pytest.approx(1/252) # Check internal dt is set
    assert result.param_ses['theta'] is not None # Check for standard error
    assert result.param_ses['mu'] is not None    # Check for standard error
    assert result.param_ses['sigma'] is not None # Check for standard error
    
    # Check if estimated parameters are reasonably close to true parameters
    assert 0.1 < result.params['theta'] < 2.5 # True theta was 0.5
    assert 8.0 < result.params['mu'] < 12.0   # True mu was 10.0
    assert 0.5 < result.params['sigma'] < 1.5 # True sigma was 1.0
    assert result.param_ses['theta'] > 0
    assert result.param_ses['mu'] > 0
    assert result.param_ses['sigma'] > 0

    # Check new KestrelResult fields
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(sample_ou_data) - 1

    # Check that the parameters are also stored on the model instance for sampling
    assert ou.theta == result.params['theta']
    assert ou.mu == result.params['mu']
    assert ou.sigma == result.params['sigma']

def test_ou_process_fit_ar1_datetime_index(sample_ou_datetime_data):
    """Test AR(1) fitting with DatetimeIndex and inferred dt."""
    ou = OUProcess()
    result = ou.fit(sample_ou_datetime_data, method='ar1', freq='B') # dt should be inferred
    
    assert ou.is_fitted
    assert result.params['theta'] is not None
    assert result.params['mu'] is not None
    assert result.params['sigma'] is not None
    assert ou._dt_ == pytest.approx(1/252, rel=0.1) # Check inferred dt is close
    assert result.param_ses['theta'] is not None # Check for standard error
    assert result.param_ses['mu'] is not None    # Check for standard error
    assert result.param_ses['sigma'] is not None # Check for standard error
    assert result.param_ses['theta'] > 0
    assert result.param_ses['mu'] > 0
    assert result.param_ses['sigma'] > 0

# --- Kalman Filter Test ---
def test_ou_process_fit_kalman(sample_ou_data):
    """Test Kalman Filter fitting of OUProcess."""
    # Using parameters from sample_ou_data generation: theta=0.5, mu=10.0, sigma=1.0
    theta_true, mu_true, sigma_true, dt = 0.5, 10.0, 1.0, 1/252
    
    ou = OUProcess()
    result = ou.fit(sample_ou_data, dt=dt, method='kalman')

    assert ou.is_fitted
    assert result.params['theta'] is not None
    assert result.params['mu'] is not None
    assert result.params['sigma'] is not None
    assert result.param_ses['theta'] is not None
    assert result.param_ses['mu'] is not None
    assert result.param_ses['sigma'] is not None

    # Check if estimated parameters are reasonably close to true parameters
    assert np.isclose(result.params['theta'], theta_true, rtol=0.2, atol=0.2)
    assert np.isclose(result.params['mu'], mu_true, rtol=0.1, atol=1.0)
    assert np.isclose(result.params['sigma'], sigma_true, rtol=0.2, atol=0.2)
    
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.residuals is not None
    assert len(result.residuals) == len(sample_ou_data) - 1

    assert ou.theta == result.params['theta']
    assert ou.mu == result.params['mu']
    assert ou.sigma == result.params['sigma']


# --- Sample Method Tests ---
def test_ou_process_sample_after_fit(sample_ou_data):
    """Test sampling after fitting the model."""
    ou = OUProcess()
    ou.fit(sample_ou_data, dt=1/252) # Fit method updates ou.theta, ou.mu, ou.sigma
    
    n_paths = 5
    horizon = 10
    sim_paths = ou.sample(n_paths=n_paths, horizon=horizon)
    
    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)
    
    # Check if simulation starts from the last fitted data point
    # We now store _last_data_point on the instance for this purpose
    assert np.allclose(sim_paths.paths.iloc[0, :], ou._last_data_point)

def test_ou_process_sample_with_initial_params():
    """Test sampling with parameters provided at initialization."""
    theta_true, mu_true, sigma_true = 0.5, 10.0, 1.0
    ou = OUProcess(theta=theta_true, mu=mu_true, sigma=sigma_true)
    
    n_paths = 5
    horizon = 10
    sim_paths = ou.sample(n_paths=n_paths, horizon=horizon, dt=1/252)
    
    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)
    
    # When not fitted, simulation should start around the long-term mean (mu)
    assert np.allclose(sim_paths.paths.iloc[0, :], mu_true, atol=2.0) # Allow some deviation for random start

def test_ou_process_sample_unfitted_no_params_raises_error():
    """Test sampling without fitting or initial params raises error."""
    ou = OUProcess()
    with pytest.raises(RuntimeError):
        ou.sample(n_paths=1, horizon=1)

def test_ou_process_sample_dt_override(sample_ou_data):
    """Test if sample method respects an overridden dt."""
    ou = OUProcess()
    ou.fit(sample_ou_data, dt=1/252) # Fit method updates ou.theta, ou.mu, ou.sigma
    
    # Override dt for sampling
    override_dt = 1/12 # Monthly dt
    sim_paths = ou.sample(n_paths=1, horizon=10, dt=override_dt)
    
    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (10 + 1, 1)

def test_ou_exact_sampling_moments():
    """Test that exact sampling produces correct terminal mean and variance."""
    np.random.seed(500)
    theta, mu, sigma, dt = 1.0, 10.0, 2.0, 1/12  # Monthly dt
    x0 = 5.0
    horizon = 12 # Simulate for 1 year
    n_paths = 50000

    ou = OUProcess(theta=theta, mu=mu, sigma=sigma)
    sim_result = ou.sample(n_paths=n_paths, horizon=horizon, dt=dt)

    terminal_values = sim_result.paths.iloc[-1, :]
    
    T = horizon * dt
    # Theoretical Expected value: E[X_T] = X_0 * e^{-theta*T} + mu * (1 - e^{-theta*T})
    expected_mean = x0 * np.exp(-theta * T) + mu * (1 - np.exp(-theta * T))
    # Theoretical Variance: Var[X_T] = sigma^2 * (1 - e^{-2*theta*T}) / (2*theta)
    expected_variance = sigma**2 * (1 - np.exp(-2 * theta * T)) / (2 * theta)
    
    assert np.isclose(terminal_values.mean(), expected_mean, rtol=0.05, atol=0.1)
    assert np.isclose(terminal_values.var(), expected_variance, rtol=0.1, atol=0.2)

def test_ou_exact_sampling_fallback_on_negative_theta():
    """Test that sampling falls back to Euler-Maruyama if theta <= 0."""
    np.random.seed(501)
    theta, mu, sigma, dt = -0.1, 10.0, 1.0, 1/12 # Negative theta
    x0 = 5.0
    horizon = 12
    n_paths = 10

    ou = OUProcess(theta=theta, mu=mu, sigma=sigma)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim_result = ou.sample(n_paths=n_paths, horizon=horizon, dt=dt)
        assert any("falling back to Euler-Maruyama" in str(warn.message).lower() for warn in w)
    
    assert isinstance(sim_result, KestrelResult)
    assert sim_result.paths.shape == (horizon + 1, n_paths)
    # Check if paths are changing, indicating simulation occurred
    assert not np.all(sim_result.paths.iloc[0, :] == sim_result.paths.iloc[1, :])


# --- Edge Cases and Error Handling ---
def test_ou_process_fit_invalid_data_type():
    """Test fitting with invalid data type."""
    ou = OUProcess()
    with pytest.raises(ValueError, match="Input data must be a pandas Series."):
        ou.fit([1, 2, 3])

def test_ou_process_fit_unknown_method():
    """Test fitting with an unknown method."""
    ou = OUProcess()
    data = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Unknown estimation method: invalid. Choose 'mle', 'ar1', or 'kalman'."):
        ou.fit(data, method='invalid')

def test_ou_process_fit_single_data_point():
    """Test fitting with a single data point (should potentially fail or be handled)."""
    ou = OUProcess()
    data = pd.Series([10.0])
    # For AR(1) and MLE, need at least 2 points for diff() or regression.
    # We expect an error or a fallback behavior.
    with pytest.raises(ValueError, match="[Ee]stimation requires at least 2 data points."):
        ou.fit(data, dt=1.0, method='mle')
    with pytest.raises(ValueError, match="[Ee]stimation requires at least 2 data points."):
        ou.fit(data, dt=1.0, method='ar1')

def test_ou_process_fit_two_data_points():
    """Test fitting with two data points."""
    ou = OUProcess()
    data = pd.Series([10.0, 10.5])
    result_mle = ou.fit(data, dt=1.0, method='mle')
    assert ou.is_fitted
    assert result_mle.params is not None
    result_ar1 = ou.fit(data, dt=1.0, method='ar1')
    assert ou.is_fitted
    assert result_ar1.params is not None

def test_residuals_properties(sample_ou_data):
    """Test residuals and associated properties."""
    ou = OUProcess()
    result = ou.fit(sample_ou_data)

    assert result.residuals is not None
    assert isinstance(result.residuals, pd.Series)
    assert np.isclose(result.residuals.mean(), 0, atol=0.1) # Standardized residuals should have mean close to 0
    assert np.isclose(result.residuals.std(), 1, atol=0.1)  # Standardized residuals should have std dev close to 1

    lb_test_results = result.ljung_box()
    assert lb_test_results is not None
    assert isinstance(lb_test_results, pd.DataFrame)
    assert 'lb_stat' in lb_test_results.columns

    normality_test_results = result.normality_test()
    assert normality_test_results is not None
    assert 'statistic' in normality_test_results
    assert 'p-value' in normality_test_results
    result.qq_plot()