import pathlib
import argparse
import numpy as np
import pandas as pd
import json
import os
import joblib

import asleep.sleep_windows as sw
from asleep.utils import data_long2wide, read, NpEncoder
from asleep.sleepnet import start_sleep_net

"""
How to run the script:

```bash
python src/asleep/get_sleep.py data/test.bin
```

"""


def get_parsed_data(raw_data_path, info_data_path, resample_hz, args):
    if os.path.exists(raw_data_path) is False or os.path.exists(
            info_data_path) is False or args.force_run is True:
        data, info = read(args.filepath, resample_hz)
        data = data.reset_index()
        pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
        data.to_csv(raw_data_path)
        with open(info_data_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        print("Raw data file saved to: {}".format(raw_data_path))
        print("Info data file saved to: {}".format(info_data_path))

    else:
        print("Raw data file already exists. Skipping raw data parsing.")
        data = pd.read_csv(raw_data_path)
        with open(info_data_path, 'r') as f:
            info = json.load(f)
    print(data.head())
    return data, info


def transform_data2model_input(data2model_path, times_path, data, args):
    """
    Current:
                                   x         y         z
    time
    2014-12-17 18:00:00.500000000  0.359059  0.195311 -0.950869
    2014-12-17 18:00:00.533333333  0.331267  0.226415 -0.931257

    Desired:
    times array: 1 x N
    data array: N x 3 x 900 (30 seconds of data at 30Hz)
    """
    if os.path.exists(data2model_path) is False or os.path.exists(
            times_path) is False or args.force_run is True:
        times = data.time.to_numpy()
        data = data[['x', 'y', 'z']].to_numpy()
        data2model, times = data_long2wide(data, times=times)

        np.save(os.path.join(args.outdir, 'data2model.npy'), data2model)
        np.save(os.path.join(args.outdir, 'times.npy'), times)
        print(
            "Data2model file saved to: {}".format(
                os.path.join(
                    args.outdir,
                    'data2model.npy')))
        print(
            "Times file saved to: {}".format(
                os.path.join(
                    args.outdir,
                    'times.npy')))

    else:
        print("Data2model file already exists. "
              "Skip data transformation.")
        data2model = np.load(data2model_path)
        times = np.load(times_path)
    return data2model, times


def get_sleep_windows(data2model, times, args):
    ssl_sleep_path = os.path.join(args.outdir, 'ssl_sleep.npy')
    if os.path.exists(ssl_sleep_path) is False or args.force_run is True:
        sleep_window_detector = joblib.load('assets/ssl.joblib.lzma')

        sleep_window_detector.device = 'cpu'  # expect channel last
        data_channel_last = np.swapaxes(data2model, 1, -1)
        window_pred = sleep_window_detector.predict(data_channel_last)
        print(window_pred.shape)
        print(np.unique(window_pred, return_counts=True))
        np.save(ssl_sleep_path, window_pred)
    else:
        window_pred = np.load(ssl_sleep_path)

    # Testing plan
    # TODO: Create visu tool to visualize the results
    # TODO 2.2 Window correction for false negative

    # 2.3 Sleep window identification
    SLEEPNET_LABELS = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
    }
    binary_y = np.vectorize(SLEEPNET_LABELS.get)(window_pred)
    my_data = {
        'time': times,
        'label': binary_y
    }
    my_df = pd.DataFrame(my_data)
    all_sleep_wins, sleep_wins_long_per_day = sw.time_series2sleep_blocks(
        my_df)

    # convert all_sleep_wins to a dataframe
    all_sleep_wins_df = pd.DataFrame(all_sleep_wins, columns=['start', 'end'])
    sleep_wins_long_per_day_df = pd.DataFrame(
        sleep_wins_long_per_day, columns=['start', 'end'])

    # 2.4 Extract and concatenate the sleep windows for the sleepnet
    master_acc, master_npids = get_master_df(
        all_sleep_wins_df, times, data2model)
    return \
        binary_y, \
        all_sleep_wins_df, \
        sleep_wins_long_per_day_df, \
        master_acc, \
        master_npids


def main():
    parser = argparse.ArgumentParser(
        description="A tool to estimate sleep stages from accelerometer data",
        add_help=True
    )
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument(
        "--outdir",
        "-o",
        help="Enter folder location to save output files",
        default="outputs/")
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download of model file")
    parser.add_argument(
        "--force_run",
        action="store_true",
        help="asleep package won't rerun the analysis to save "
             "time. force_run will make sure everything is "
             "regenerated")
    parser.add_argument(
        "--pytorch_device",
        "-d",
        help="Pytorch device to use, e.g.: 'cpu' or 'cuda:0' (for SSL only)",
        type=str,
        default='cpu')
    args = parser.parse_args()

    resample_hz = 30
    raw_data_path = os.path.join(args.outdir, 'raw.csv')
    info_data_path = os.path.join(args.outdir, 'info.json')
    data2model_path = os.path.join(args.outdir, 'data2model.npy')
    times_path = os.path.join(args.outdir, 'times.npy')

    # 1. Parse raw files into a dataframe
    data, info = get_parsed_data(
        raw_data_path, info_data_path, resample_hz, args)
    print(data.head())

    # 1.1 Transform data into a usable format for inference
    # TODO: refactor the saving and loading functions
    data2model, times = transform_data2model_input(
        data2model_path, times_path, data, args)
    print("data2model shape: {}".format(data2model.shape))
    print("times shape: {}".format(times.shape))

    # 2. sleep window detection and inference
    (binary_y, all_sleep_wins_df,
     sleep_wins_long_per_day_df,
     master_acc, master_npids) = get_sleep_windows(data2model, times, args)

    y_pred, test_pids = start_sleep_net(master_acc, master_npids, args.outdir)

    for block_id in range(len(all_sleep_wins_df)):
        start_t = all_sleep_wins_df.iloc[block_id]['start']
        end_t = all_sleep_wins_df.iloc[block_id]['end']

        time_filter = (times >= start_t) & (times < end_t)

        # get the corresponding sleepnet predictions
        sleepnet_pred = y_pred[test_pids == block_id]

        # fill the sleepnet predictions back to the original dataframe
        binary_y[time_filter] = sleepnet_pred

    # 3.2 TODO: add features to have wake/sleep or wake/REM/NREM label
    print(binary_y)

    # 4. save summary statistics


def get_master_df(block_time_df, times, acc_array):
    # extract interval based on times
    master_acc = []
    master_npids = []  # night ids

    for index, row in block_time_df.iterrows():
        start_t = row["start"]
        end_t = row["end"]

        time_filter = (times >= start_t) & (times < end_t)
        current_day_acc = acc_array[time_filter]

        day_pid = np.ones(np.sum(time_filter)) * index

        if len(master_npids) == 0:
            master_acc = current_day_acc
            master_npids = day_pid
        else:
            master_acc = np.concatenate((master_acc, current_day_acc))
            master_npids = np.concatenate((master_npids, day_pid))

    return master_acc, master_npids


if __name__ == '__main__':
    main()
