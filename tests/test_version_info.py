import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pytest

import asleep.get_sleep as get_sleep


def test_get_parsed_data_writes_package_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw.csv"
    info_path = tmp_path / "info.json"
    source_data = pd.DataFrame(
        {"x": [0.0], "y": [0.0], "z": [1.0], "non_wear": [False]},
        index=pd.DatetimeIndex(["2024-01-01"], name="time"),
    )
    monkeypatch.setattr(get_sleep, "version", lambda package: "0.4.18")
    monkeypatch.setattr(
        get_sleep, "read", lambda *args, **kwargs: (source_data, {})
    )
    args = argparse.Namespace(
        filepath="input.csv", time_shift="0", force_run=True, outdir=str(tmp_path)
    )

    _, info = get_sleep.get_parsed_data(raw_path, info_path, 30, args)

    assert info["AsleepVersion"] == "0.4.18"
    with info_path.open(encoding="utf-8") as info_file:
        persisted_info: Dict[str, Any] = json.load(info_file)
    assert persisted_info["AsleepVersion"] == "0.4.18"


@pytest.mark.parametrize("cached_version", [None, "old"], ids=["legacy", "stale"])
def test_get_parsed_data_refreshes_version_in_cached_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached_version: Optional[str]
) -> None:
    raw_path = tmp_path / "raw.csv"
    info_path = tmp_path / "info.json"
    pd.DataFrame(
        {
            "time": ["2024-01-01"],
            "x": [0.0],
            "y": [0.0],
            "z": [1.0],
            "non_wear": [False],
        }
    ).to_csv(raw_path, index=False)
    raw_before = raw_path.read_bytes()
    cached_info: Dict[str, Any] = {
        "StartTime": "2024-01-01 00:00:00",
        "EndTime": "2024-01-01 00:00:00",
        "SampleRate": 30.0,
    }
    if cached_version is not None:
        cached_info["AsleepVersion"] = cached_version
    info_path.write_text(json.dumps(cached_info), encoding="utf-8")
    monkeypatch.setattr(get_sleep, "version", lambda package: "0.4.18")
    monkeypatch.setattr(
        get_sleep,
        "read",
        lambda *args: (_ for _ in ()).throw(AssertionError("input reparsed")),
    )
    args = argparse.Namespace(
        filepath="input.csv", time_shift="0", force_run=False, outdir=str(tmp_path)
    )

    _, info = get_sleep.get_parsed_data(raw_path, info_path, 30, args)

    expected_info = {**cached_info, "AsleepVersion": "0.4.18"}
    assert info == expected_info
    with info_path.open(encoding="utf-8") as info_file:
        persisted_info = json.load(info_file)
    assert persisted_info == expected_info
    assert raw_path.read_bytes() == raw_before
