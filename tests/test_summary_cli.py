import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asleep.test_summary import main


def test_main_round_trips_saved_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    times = np.array(
        [
            "2024-01-01 20:59:30",
            "2024-01-01 21:00:00",
            "2024-01-01 21:00:30",
            "2024-01-01 21:01:00",
            "2024-01-01 21:01:30",
            "2024-01-01 21:02:00",
        ]
    )
    np.save(output_dir / "times.npy", times)

    pd.DataFrame(
        {
            "start": ["2024-01-01 21:00:00"],
            "end": ["2024-01-01 21:01:30"],
            "interval_start": ["2024-01-01 12:00:00"],
            "interval_end": ["2024-01-02 11:59:59"],
            "wear_duration_H": [24.0],
            "is_longest_block": [True],
        }
    ).to_csv(output_dir / "sleep_block.csv", index=False)
    pd.DataFrame({"raw_label": [0, 1, 1, 2, 3, 0]}).to_csv(
        output_dir / "predictions.csv", index=False
    )

    monkeypatch.chdir(tmp_path)
    main()

    day_summary = pd.read_csv(output_dir / "day_summary.csv")
    assert day_summary.loc[0, "start_day"] == "2024-01-01"
    assert day_summary.loc[0, "wear_duration_H"] == 24.0
    assert day_summary.loc[0, "tst_min"] == 2.0
    assert day_summary.loc[0, "n1_min"] == 1.0

    with (output_dir / "results.json").open(encoding="utf-8") as results_file:
        results = json.load(results_file)
    assert results["overall_mean_tst_min"] == 2.0
