import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_alpha.data.synthetic import simulate_book, simulate_panel  # noqa: E402


@pytest.fixture(scope="session")
def small_book() -> pd.DataFrame:
    return simulate_book(n_events=4000, mode="ofi", beta=0.15, seed=1)


@pytest.fixture(scope="session")
def small_panel() -> pd.DataFrame:
    return simulate_panel(n_instruments=2, n_events=3000, mode="ofi", beta=0.15, seed=2)


@pytest.fixture(scope="session")
def null_book() -> pd.DataFrame:
    return simulate_book(n_events=8000, mode="null", beta=0.0, seed=5)
