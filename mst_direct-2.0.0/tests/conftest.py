"""Shared fixtures for the v2 reproduction-code test suite.

Tests use small synthetic clouds (fast, deterministic) for the unit tests and
the bundled reference distribution only for the integration smoke test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running the tests without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def make_cloud():
    """Factory for a non-linear, heteroscedastic 3-variate cloud of size ``n``."""

    def _make(n, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        # bimodal (Y = +-X) + heteroscedastic third variable
        y = rng.choice([-1.0, 1.0], size=n) * x + 0.1 * rng.normal(size=n)
        z = x + np.abs(x) * rng.normal(size=n)
        return np.column_stack([x, y, z])

    return _make
