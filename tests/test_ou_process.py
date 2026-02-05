# tests/test_ou_process.py
import pytest
import pandas as pd
import numpy as np
from kestrel.diffusion.ou import OUProcess
from kestrel.utils.kestrel_result import KestrelResult

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
    ou.fit(sample_ou_data, dt=1/252, method='mle')
    
    assert ou.is_fitted
    assert hasattr(ou, 'theta_')
    assert hasattr(ou, 'mu_')
    assert hasattr(ou, 'sigma_')
    assert hasattr(ou, '_last_data_point')
    assert hasattr(ou, '_dt_')
    assert hasattr(ou, 'theta_se_') # Check for standard error
    assert hasattr(ou, 'mu_se_')    # Check for standard error
    assert hasattr(ou, 'sigma_se_') # Check for standard error
    
    # Check if estimated parameters are reasonably close to true parameters (from sample_ou_data generation)
    # The actual values will vary due to stochastic nature, so we check for plausible ranges
    assert 0.1 < ou.theta_ < 2.5 # True theta was 0.5
    assert 8.0 < ou.mu_ < 12.0   # True mu was 10.0
    assert 0.5 < ou.sigma_ < 1.5 # True sigma was 1.0
    assert ou.theta_se_ > 0 # Standard errors should be positive
    assert ou.mu_se_ > 0
    assert ou.sigma_se_ > 0

def test_ou_process_fit_mle_datetime_index(sample_ou_datetime_data):
    """Test MLE fitting with DatetimeIndex and inferred dt."""
    ou = OUProcess()
    ou.fit(sample_ou_datetime_data, method='mle', freq='B') # dt should be inferred
    
    assert ou.is_fitted
    assert hasattr(ou, 'theta_')
    assert hasattr(ou, 'mu_')
    assert hasattr(ou, 'sigma_')
    assert hasattr(ou, '_last_data_point')
    assert hasattr(ou, '_dt_')
    assert ou._dt_ == pytest.approx(1/252, rel=0.1) # Check inferred dt is close
    assert hasattr(ou, 'theta_se_') # Check for standard error
    assert hasattr(ou, 'mu_se_')    # Check for standard error
    assert hasattr(ou, 'sigma_se_') # Check for standard error
    assert ou.theta_se_ > 0
    assert ou.mu_se_ > 0
    assert ou.sigma_se_ > 0

# --- Fit Method Tests (AR1) ---
def test_ou_process_fit_ar1(sample_ou_data):
    """Test AR(1) fitting of OUProcess."""
    ou = OUProcess()
    ou.fit(sample_ou_data, dt=1/252, method='ar1')
    
    assert ou.is_fitted
    assert hasattr(ou, 'theta_')
    assert hasattr(ou, 'mu_')
    assert hasattr(ou, 'sigma_')
    assert hasattr(ou, '_last_data_point')
    assert hasattr(ou, '_dt_')
    assert hasattr(ou, 'theta_se_') # Check for standard error
    assert hasattr(ou, 'mu_se_')    # Check for standard error
    assert hasattr(ou, 'sigma_se_') # Check for standard error
    
    # Check if estimated parameters are reasonably close to true parameters
    assert 0.1 < ou.theta_ < 2.5 # True theta was 0.5
    assert 8.0 < ou.mu_ < 12.0   # True mu was 10.0
    assert 0.5 < ou.sigma_ < 1.5 # True sigma was 1.0
    assert ou.theta_se_ > 0
    assert ou.mu_se_ > 0
    assert ou.sigma_se_ > 0

def test_ou_process_fit_ar1_datetime_index(sample_ou_datetime_data):
    """Test AR(1) fitting with DatetimeIndex and inferred dt."""
    ou = OUProcess()
    ou.fit(sample_ou_datetime_data, method='ar1', freq='B') # dt should be inferred
    
    assert ou.is_fitted
    assert hasattr(ou, 'theta_')
    assert hasattr(ou, 'mu_')
    assert hasattr(ou, 'sigma_')
    assert hasattr(ou, '_last_data_point')
    assert hasattr(ou, '_dt_')
    assert ou._dt_ == pytest.approx(1/252, rel=0.1) # Check inferred dt is close
    assert hasattr(ou, 'theta_se_') # Check for standard error
    assert hasattr(ou, 'mu_se_')    # Check for standard error
    assert hasattr(ou, 'sigma_se_') # Check for standard error
    assert ou.theta_se_ > 0
    assert ou.mu_se_ > 0
    assert ou.sigma_se_ > 0

# --- Sample Method Tests ---
def test_ou_process_sample_after_fit(sample_ou_data):
    """Test sampling after fitting the model."""
    ou = OUProcess()
    ou.fit(sample_ou_data, dt=1/252)
    
    n_paths = 5
    horizon = 10
    sim_paths = ou.sample(n_paths=n_paths, horizon=horizon)
    
    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (horizon + 1, n_paths)
    
    # Check if simulation starts from the last fitted data point
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
    ou.fit(sample_ou_data, dt=1/252)
    
    # Override dt for sampling
    override_dt = 1/12 # Monthly dt
    sim_paths = ou.sample(n_paths=1, horizon=10, dt=override_dt)
    
    assert isinstance(sim_paths, KestrelResult)
    assert sim_paths.paths.shape == (10 + 1, 1)

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
    with pytest.raises(ValueError, match="Unknown estimation method: invalid. Choose 'mle' or 'ar1'."):
        ou.fit(data, method='invalid')

def test_ou_process_fit_single_data_point():
    """Test fitting with a single data point (should potentially fail or be handled)."""
    ou = OUProcess()
    data = pd.Series([10.0])
    # For AR(1) and MLE, need at least 2 points for diff() or regression.
    # We expect an error or a fallback behavior.
    with pytest.raises(ValueError, match="estimation requires at least 2 data points."):
        ou.fit(data, dt=1.0, method='mle')
    with pytest.raises(ValueError, match="regression requires at least 2 data points."):
        ou.fit(data, dt=1.0, method='ar1')

def test_ou_process_fit_two_data_points():
    """Test fitting with two data points."""
    ou = OUProcess()
    data = pd.Series([10.0, 10.5])
    ou.fit(data, dt=1.0, method='mle')
    assert ou.is_fitted
    ou.fit(data, dt=1.0, method='ar1')
    assert ou.is_fitted
