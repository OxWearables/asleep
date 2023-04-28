import joblib
from utils import resize
from asleep.models import SleepWindowSSL
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('..')


"""
Run this on the GPU node 1.

This script will train a binary sleep classifier
using the sleep diary data from capture24.
We use the pre-trained model with HMM smoothing to do the classification.

It will save the network weights for harnet30 and HMM.

Usage:
```
python train_capture24.py
```
"""


def load_data():
    root = '/data/UKBB/capture24_30s/'  # change this path if needed

    X = np.load(os.path.join(root, 'X.npy'))  # accelerometer data
    y = np.load(os.path.join(root, 'Y_Walmsley.npy'))  # true labels
    pid = np.load(os.path.join(root, 'pid.npy'))  # participant IDs

    print(f'X shape: {X.shape}')
    print(f'Y shape: {y.shape}')  # same shape as pid
    print(f'Label distribution:\n{pd.Series(y).value_counts()}')

    CAPTURE24_LABELS = {
        'light': 0,
        'moderate-vigorous': 1,
        'sedentary': 2,
        'sleep': 3,
    }
    y = np.vectorize(CAPTURE24_LABELS.get)(y)
    print(f'Label distribution:\n{pd.Series(y).value_counts()}')

    # down sample if required.
    # our pre-trained model expects windows of 30s at 30Hz = 900 samples
    input_size = 900  # 30s at 30Hz

    if X.shape[1] == input_size:
        print("No need to downsample")
    else:
        X = resize(X, input_size)

    X = X.astype(
        "f4"
    )  # PyTorch defaults to float32

    return (
        X, y, pid
    )


def main():
    X, y, pid = load_data()

    my_device = 'cuda:0'

    sleep_window_detector = SleepWindowSSL(
        device=my_device, batch_size=2000, verbose=True)
    sleep_window_detector.fit(X, y, groups=pid)
    joblib.dump(sleep_window_detector, 'ssl.joblib.lzma', compress=('lzma', 3))


if __name__ == '__main__':
    main()
