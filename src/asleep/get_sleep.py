import pathlib
import argparse
import numpy as np
import pandas as pd
import json
import os
import joblib
import urllib
import shutil
import datetime

import asleep.sleep_windows as sw
from asleep.utils import data_long2wide, read, NpEncoder
from asleep.sleepnet import start_sleep_net
from asleep.macros import SLEEPNET_LABELS, SLEEPNET_BINARY_LABELS, SLEEPNET_THRE_CLASS_LABELS
from asleep.summary import generate_sleep_parameters, summarize_daily_sleep

"""
How to run the script:

```bash
python src/asleep/get_sleep.py data/test.bin -m 22
-w /Users/hangy/Dphil/code/asleep/bi_sleepnet.mdl -l /Users/hangy/Dphil/code/asleep

python src/asleep/get_sleep.py data/sample.cwa.gz -m 22

```

All the prediction data will be saved for all epochs including non-wear
but just the non-wear epoch labels
will always be -1.

"""

NON_WEAR_THRESHOLD = 3  # H
NON_WEAR_PREDICTION_FLAG = -1


def load_model(model_path, force_download=False):
    """ Load trained model. Download if not exists. """

    pth = pathlib.Path(model_path)

    if force_download or not pth.exists():
        url = "https://github.com/OxWearables/asleep/releases/download/0.3.1/ssl.joblib.lzma"

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    return joblib.load(pth)


def get_parsed_data(raw_data_path, info_data_path, resample_hz, args):
    time_shift = 0
    if args.time_shift != '0':
        if args.time_shift[0] == '-':
            time_shift = -int(args.time_shift[1:])
        else:
            time_shift = int(args.time_shift[1:])

    # add the time shift to the parsed data frame
    if os.path.exists(raw_data_path) is False or os.path.exists(
            info_data_path) is False or args.force_run is True:
        data, info = read(args.filepath, resample_hz)
        # data will have the following columns:
        # time, x, y, z, non_wear
        data = data.reset_index()

        # apply time shift
        start_time = pd.to_datetime(info['StartTime'])
        end_time = pd.to_datetime(info['EndTime'])
        info['StartTime'] = start_time + datetime.timedelta(hours=time_shift)
        info['EndTime'] = end_time + datetime.timedelta(hours=time_shift)
        info['StartTime'] = info['StartTime'].strftime('%Y-%m-%d %H:%M:%S')
        info['EndTime'] = info['EndTime'].strftime('%Y-%m-%d %H:%M:%S')
        data['time'] = data['time'] + datetime.timedelta(hours=time_shift)
        print("Time shift applied: {} hours".format(time_shift))

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


def transform_data2model_input(
        data2model_path,
        times_path,
        non_wear_path,
        data,
        args):
    """
    Current:
                                   x         y         z          non_wear
    time
    2014-12-17 18:00:00.500000000  0.359059  0.195311 -0.950869   True
    2014-12-17 18:00:00.533333333  0.331267  0.226415 -0.931257   False

    Desired:
    times array: 1 x N
    non_wear_flag array: 1 x N
    data array: N x 3 x 900 (30 seconds of data at 30Hz)
    """
    if os.path.exists(data2model_path) is False or os.path.exists(
            times_path) is False or args.force_run is True:
        times = data['time'].to_numpy()
        non_wear = data['non_wear'].to_numpy()
        data = data[['x', 'y', 'z']].to_numpy()
        data2model, times, non_wear = data_long2wide(data, times, non_wear)

        np.save(data2model_path, data2model)
        np.save(times_path, times)
        np.save(non_wear_path, non_wear)

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
        print(
            "Non-wear file saved to: {}".format(
                os.path.join(
                    args.outdir,
                    'non_wear.npy')))
    else:
        print("Data2model file already exists. "
              "Skip data transformation.")
        data2model = np.load(data2model_path)
        times = np.load(times_path)
        non_wear = np.load(non_wear_path)
    return data2model, times, non_wear


def get_sleep_windows(data2model, times, non_wear, args):
    # data2model: N x 3 x 900
    # non_wear_flag: N x 1
    # TODO: only inference on the periods when non-wear is false
    ssl_sleep_path = os.path.join(args.outdir, 'ssl_sleep.npy')
    sleep_prediction = np.ones(len(times)) * NON_WEAR_PREDICTION_FLAG
    original_data2model = data2model.copy()
    data2model = data2model[~non_wear]

    if os.path.exists(ssl_sleep_path) is False or args.force_run is True:
        model_path = os.path.join(
            pathlib.Path(__file__).parent,
            "ssl.joblib.lzma")
        sleep_window_detector = load_model(
            model_path, force_download=args.force_download)
        sleep_window_detector.device = 'cpu'  # expect channel last
        data_channel_last = np.swapaxes(data2model, 1, -1)
        window_pred = sleep_window_detector.predict(data_channel_last)
        sleep_prediction[~non_wear] = window_pred
        print(sleep_prediction.shape)
        print(np.unique(sleep_prediction, return_counts=True))
        np.save(ssl_sleep_path, sleep_prediction)
    else:
        sleep_prediction = np.load(ssl_sleep_path)

    # 2.1 Sleep window identification
    binary_y = np.vectorize(SLEEPNET_LABELS.get)(sleep_prediction)
    my_data = {
        'time': times,
        'label': binary_y,
        'is_wear': binary_y != NON_WEAR_PREDICTION_FLAG
    }

    # non-wear fix for false positive
    my_df = pd.DataFrame(my_data)
    counter = sw.find_sleep_block_duration(my_df)
    epoch_length = 30  # unit in sec
    valid_sleep_block_idxes = sw.find_valid_sleep_blocks(counter, epoch_length)
    gap2fill = sw.find_gaps2fill(
        valid_sleep_block_idxes, epoch_length, counter)
    my_df = sw.fill_gaps(my_df, counter, gap2fill)

    all_sleep_wins, sleep_wins_long_per_day, \
        interval_start, interval_end, wear_time = sw.time_series2sleep_blocks(my_df)

    # convert all_sleep_wins to a dataframe
    all_sleep_wins_df = pd.DataFrame(all_sleep_wins, columns=['start', 'end'])
    all_sleep_wins_df['interval_start'] = interval_start
    all_sleep_wins_df['interval_end'] = interval_end
    all_sleep_wins_df['wear_duration_H'] = wear_time
    sleep_wins_long_per_day_df = pd.DataFrame(
        sleep_wins_long_per_day, columns=['start', 'end'])

    # 2.2 Extract and concatenate the sleep windows for the sleepnet
    master_acc, master_npids = get_master_df(
        all_sleep_wins_df, times, original_data2model)

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
        "--remove_intermediate_files",
        action="store_true",
        help="Remove intermediate files to save space but it "
             "will take longer to run the next time.")
    parser.add_argument(
        "--pytorch_device",
        "-d",
        help="Pytorch device to use, e.g.: 'cpu' or 'cuda:0' (for SSL only)",
        type=str,
        default='cpu')
    parser.add_argument(
        "--model_weight_path",
        "-w",
        help="For cluster job where there is no internet connection,"
             "you might want to specify the path to the model weight file",
        type=str,
        default='')
    parser.add_argument(
        "--local_repo_path",
        "-l",
        help="Load model definition from local repo",
        type=str,
        default='')
    parser.add_argument(
        "--min_wear",
        "-m",
        help="Min wear time in hours to be eligible for summary statistics "
             "computation. The sleepnet paper uses 22",
        type=int,
        default=22)
    parser.add_argument(
        "--time_shift",
        type=str,
        help="The number hours to shift forward or backward from "
             "the current device time. e.g. +1 or -1",
        default="0")
    args = parser.parse_args()

    resample_hz = 30

    # get file name and create a folder for the output
    filename = os.path.basename(args.filepath)
    filename = filename.split('.')[0]
    os.makedirs(args.outdir, exist_ok=True)

    args.outdir = os.path.join(args.outdir, filename)
    print("Saving files to dir: {}".format(args.outdir))

    raw_data_path = os.path.join(args.outdir, 'raw.csv')
    info_data_path = os.path.join(args.outdir, 'info.json')
    data2model_path = os.path.join(args.outdir, 'data2model.npy')
    times_path = os.path.join(args.outdir, 'times.npy')
    non_wear_path = os.path.join(args.outdir, 'non_wear.npy')

    # 1. Parse raw files into a dataframe
    # Add non-wear detection
    data, info = get_parsed_data(
        raw_data_path, info_data_path, resample_hz, args)
    if args.remove_intermediate_files:
        os.remove(raw_data_path)
        os.remove(info_data_path)

    # 1.1 Transform data into a usable format for inference
    data2model, times, non_wear = transform_data2model_input(
        data2model_path, times_path, non_wear_path, data, args)
    print("data2model shape: {}".format(data2model.shape))
    print("times shape: {}".format(times.shape))
    print("Non_wear flag shape: {}".format(non_wear.shape))

    # times and non-wear flag need to be stored for visualization
    if args.remove_intermediate_files:
        os.remove(data2model_path)

    # 2. sleep window detection and inference
    (binary_y, all_sleep_wins_df, sleep_wins_long_per_day_df, master_acc,
     master_npids) = get_sleep_windows(data2model, times, non_wear, args)

    if len(master_npids) <= 0:
        print("No sleep windows >30 mins detected. Exiting...")
        print("Non-wear time has been written to %s" % non_wear_path)
        print("Current sleep classification has been written to %s" %
              os.path.join(args.outdir, 'ssl_sleep.npy'))
        exit()
    else:
        y_pred, test_pids = start_sleep_net(
            master_acc, master_npids, args.outdir,
            args.model_weight_path, local_repo_path=args.local_repo_path,
            device_id=args.pytorch_device)
        sleepnet_output = binary_y

        for block_id in range(len(all_sleep_wins_df)):
            start_t = all_sleep_wins_df.iloc[block_id]['start']
            end_t = all_sleep_wins_df.iloc[block_id]['end']

            time_filter = (times >= start_t) & (times < end_t)

            # get the corresponding sleepnet predictions
            sleepnet_pred = y_pred[test_pids == block_id]

            # fill the sleepnet predictions back to the original dataframe
            sleepnet_output[time_filter] = sleepnet_pred

    # 3. Skip this step if predictions already exist
    # Output pandas dataframe
    # Times, Sleep/Wake, Sleep Stage
    sleep_wake_predictions = np.vectorize(
        SLEEPNET_BINARY_LABELS.get)(sleepnet_output)
    sleep_stage_predictions = np.vectorize(
        SLEEPNET_THRE_CLASS_LABELS.get)(sleepnet_output)

    predictions_df = pd.DataFrame(
        {
            'time': times,
            'sleep_wake': sleep_wake_predictions,
            'sleep_stage': sleep_stage_predictions,
            'raw_label': sleepnet_output,
        }
    )
    final_prediction_path = os.path.join(args.outdir, 'predictions.csv')
    print("predictions_df shape: {}".format(predictions_df.shape))
    print(predictions_df.head())
    print("Predictions saved to: {}".format(final_prediction_path))
    predictions_df.to_csv(final_prediction_path, index=False)

    # 4. Summary statistics
    # 4.1 Generate sleep block df and indicate the longest block per day
    # time start, time end, is_longest_block
    all_sleep_wins_df['is_longest_block'] = False
    for index, row in sleep_wins_long_per_day_df.iterrows():
        start_t = row['start']
        end_t = row['end']
        all_sleep_wins_df.loc[(all_sleep_wins_df['start'] == start_t) & (
            all_sleep_wins_df['end'] == end_t), 'is_longest_block'] = True
    sleep_block_path = os.path.join(args.outdir, 'sleep_block.csv')
    print(all_sleep_wins_df.head())
    print("Sleep block saved to: {}".format(sleep_block_path))
    all_sleep_wins_df.to_csv(sleep_block_path, index=False)

    # 4.2  Generate daily summary statistics
    output_json_path = os.path.join(args.outdir, 'summary.json')
    day_summary_path = os.path.join(args.outdir, 'day_summary.csv')
    # save day level df to csv
    day_summary_df = generate_sleep_parameters(
        all_sleep_wins_df, times, predictions_df, day_summary_path)

    # 4.3 Generate summary statistics across different days
    summarize_daily_sleep(day_summary_df, output_json_path, args.min_wear)


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
