import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

import asleep.get_sleep as get_sleep
import asleep.utils as utils


def test_read_uses_explicit_sample_rate_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.pkl"
    input_path.write_bytes(b"placeholder")
    data = pd.DataFrame(
        {"x": [0.0], "y": [0.0], "z": [1.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    observed: List[int] = []

    monkeypatch.setattr("asleep.utils.pd.read_pickle", lambda path: data)
    monkeypatch.setattr(
        utils,
        "infer_freq",
        lambda index: (_ for _ in ()).throw(AssertionError("frequency inferred")),
    )

    def process(
        frame: pd.DataFrame, sample_rate: int, **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        observed.append(sample_rate)
        return frame, {}

    monkeypatch.setattr("asleep.utils.actipy.process", process)
    monkeypatch.setattr(utils, "detect_nonwear", lambda frame: (frame, {}))

    _, info = utils.read(input_path, sample_rate=25)

    assert observed == [25]
    assert info["SampleRate"] == 25


def test_read_infers_sample_rate_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.pkl"
    input_path.write_bytes(b"placeholder")
    data = pd.DataFrame(
        {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="50ms"),
    )
    observed: List[int] = []

    monkeypatch.setattr("asleep.utils.pd.read_pickle", lambda path: data)

    def process(
        frame: pd.DataFrame, sample_rate: int, **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        observed.append(sample_rate)
        return frame, {}

    monkeypatch.setattr("asleep.utils.actipy.process", process)
    monkeypatch.setattr(utils, "detect_nonwear", lambda frame: (frame, {}))

    _, info = utils.read(input_path)

    assert observed == [20]
    assert info["SampleRate"] == 20


def test_get_parsed_data_forwards_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: List[Optional[int]] = []
    source_data = pd.DataFrame(
        {"x": [0.0], "y": [0.0], "z": [1.0], "non_wear": [False]},
        index=pd.DatetimeIndex(["2024-01-01"], name="time"),
    )

    def read(
        filepath: str, resample_hz: int, sample_rate: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        observed.append(sample_rate)
        return source_data, {}

    monkeypatch.setattr(get_sleep, "read", read)
    args = argparse.Namespace(
        filepath="input.csv",
        sample_rate=40,
        time_shift="0",
        force_run=True,
        outdir=str(tmp_path),
    )

    get_sleep.get_parsed_data(
        tmp_path / "raw.csv", tmp_path / "info.json", 30, args
    )

    assert observed == [40]
