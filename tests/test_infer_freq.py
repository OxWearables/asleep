from __future__ import annotations

import numpy as np
import pandas as pd

from asleep.utils import infer_freq


def test_infer_freq_regular_index() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="10ms")
    assert infer_freq(idx) == pd.Timedelta("10ms")


def test_infer_freq_ignores_occasional_gaps() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="10ms").delete([5, 50])
    assert infer_freq(idx) == pd.Timedelta("10ms")


def test_infer_freq_sub_millisecond_steps() -> None:
    steps = pd.to_timedelta(np.arange(50) * 33333, unit="us")
    idx = pd.DatetimeIndex(pd.Timestamp("2020-01-01") + steps)
    assert infer_freq(idx) == pd.Timedelta(microseconds=33333)
