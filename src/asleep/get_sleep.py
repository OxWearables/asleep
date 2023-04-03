import pathlib
import time
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import actipy
import json
import os

from utils import data_long2wide

"""
How to run the script:

```bash
python src/asleep/get_sleep.py data/test.bin
```

"""


def main():
    parser = argparse.ArgumentParser(
        description="A tool to estimate sleep stages from accelerometer data",
        add_help=True
    )
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument("--outdir", "-o", help="Enter folder location to save output files", default="outputs/")
    parser.add_argument("--force_download", action="store_true", help="Force download of model file")
    parser.add_argument("--force_run", action="store_true", help="asleep package won't rerun the analysis to save "
                                                                 "time. force_run will make sure everything is "
                                                                 "regenerated")
    parser.add_argument("--pytorch_device", "-d", help="Pytorch device to use, e.g.: 'cpu' or 'cuda:0' (for SSL only)",
                        type=str, default='cpu')
    args = parser.parse_args()

    before = time.time()

    # 1. Parse raw files into a dataframe
    resample_hz = 30
    raw_data_path = os.path.join(args.outdir, 'raw.csv')
    info_data_path = os.path.join(args.outdir, 'info.json')

    if os.path.exists(raw_data_path) is False or os.path.exists(info_data_path) is False or args.force_run is True:
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

    # 1.1 Transform data into a usable format for inference
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

    data2model_path = os.path.join(args.outdir, 'data2model.npy')
    times_path = os.path.join(args.outdir, 'times.npy')
    if os.path.exists(data2model_path) is False or os.path.exists(times_path) is False or args.force_run is True:
        times = data.time.to_numpy()
        data = data[['x', 'y', 'z']].to_numpy()
        data2model, times = data_long2wide(data, times=times)

        np.save(os.path.join(args.outdir, 'data2model.npy'), data2model)
        np.save(os.path.join(args.outdir, 'times.npy'), times)
        print("Data2model file saved to: {}".format(os.path.join(args.outdir, 'data2model.npy')))
        print("Times file saved to: {}".format(os.path.join(args.outdir, 'times.npy')))

    else:
        print("Data2model file already exists. Skipping data2model transformation.")
        data2model = np.load(data2model_path)
        times = np.load(times_path)

    # print data2model shape info
    print("data2model shape: {}".format(data2model.shape))
    print("times shape: {}".format(times.shape))

    # 2. sleep window detection

    # 2.1 window correction for false negative

    # 3. sleep stage classification

    # 3.1 save window-specific classification results

    # 4. save summary statistics


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def infer_freq(x):
    """ Like pd.infer_freq but more forgiving """
    freq, _ = stats.mode(np.diff(x), keepdims=False)
    freq = pd.Timedelta(freq)
    return freq


def read(filepath, resample_hz='uniform'):
    p = pathlib.Path(filepath)
    ftype = p.suffixes[0].lower()
    fsize = round(p.stat().st_size / (1024 * 1024), 1)

    if ftype in (".csv", ".pkl"):

        if ftype == ".csv":
            data = pd.read_csv(
                filepath,
                usecols=['time', 'x', 'y', 'z'],
                parse_dates=['time'],
                index_col='time'
            )
        elif ftype == ".pkl":
            data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unknown file format: {ftype}")

        freq = infer_freq(data.index)
        sample_rate = int(np.round(pd.Timedelta('1s') / freq))

        data, info = actipy.process(
            data, sample_rate,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
        )

        info = {
            **{"Filename": filepath,
               "Device": ftype,
               "Filesize(MB)": fsize,
               "SampleRate": sample_rate},
            **info
        }

    elif ftype in (".cwa", ".gt3x", ".bin"):

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
        )

    if 'ResampleRate' not in info:
        info['ResampleRate'] = info['SampleRate']

    return data, info


if __name__ == '__main__':
    main()
