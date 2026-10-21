from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

import asleep.utils as utils


def test_read_uses_same_lowpass_for_csv_and_device_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = pd.DataFrame(
        {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="50ms"),
    )
    observed: List[Tuple[str, Any]] = []

    def process(
        frame: pd.DataFrame, sample_rate: int, **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        observed.append(("process", kwargs["lowpass_hz"]))
        return frame, {}

    def read_device(
        filepath: str, **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        observed.append(("read_device", kwargs["lowpass_hz"]))
        return data, {"SampleRate": 20}

    monkeypatch.setattr("asleep.utils.pd.read_pickle", lambda path: data)
    monkeypatch.setattr("asleep.utils.actipy.process", process)
    monkeypatch.setattr("asleep.utils.actipy.read_device", read_device)
    monkeypatch.setattr(utils, "detect_nonwear", lambda frame: (frame, {}))

    pkl_path = tmp_path / "input.pkl"
    pkl_path.write_bytes(b"placeholder")
    bin_path = tmp_path / "input.bin"
    bin_path.write_bytes(b"placeholder")

    utils.read(pkl_path)
    utils.read(bin_path)

    assert observed == [
        ("process", utils.LOWPASS_HZ),
        ("read_device", utils.LOWPASS_HZ),
    ]
    assert utils.LOWPASS_HZ == 15
