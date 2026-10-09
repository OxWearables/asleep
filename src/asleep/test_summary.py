
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from asleep.summary import generate_sleep_parameters, summarize_daily_sleep


def main() -> None:
    output_dir = "outputs"
    sleep_block_path = os.path.join(output_dir, "sleep_block.csv")
    prediction_path = os.path.join(output_dir, "predictions.csv")
    day_summary_path = os.path.join(output_dir, "day_summary.csv")
    json_path = os.path.join(output_dir, "results.json")
    times_path = os.path.join(output_dir, "times.npy")

    sleep_block_df = pd.read_csv(
        sleep_block_path,
        parse_dates=["start", "end", "interval_start", "interval_end"],
    )
    prediction_df = pd.read_csv(prediction_path)
    times = pd.to_datetime(np.load(times_path)).to_numpy()

    day_summary_df = generate_sleep_parameters(
        sleep_block_df, times, prediction_df, day_summary_path
    )
    summarize_daily_sleep(day_summary_df, json_path, min_wear_time_h=22)


if __name__ == "__main__":
    main()
