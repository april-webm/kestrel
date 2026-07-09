"""Tests for KestrelResult utilities."""

import pandas as pd

from kestrel.utils.kestrel_result import KestrelResult


def test_simulation_only_repr_handles_missing_fit_metrics():
    """Simulation-only results should be printable without fitted metrics."""
    result = KestrelResult(pd.DataFrame([[1.0]]), initial_value=1.0)

    text = repr(result)

    assert "log_likelihood=N/A" in text
    assert "aic=N/A" in text
    assert "bic=N/A" in text
