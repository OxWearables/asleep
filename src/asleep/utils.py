import torch
import random
import numpy as np
import scipy.stats as stats
import actipy
import pandas as pd
import pathlib
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from transforms3d.axangles import axangle2mat
from torchvision import transforms
from scipy.interpolate import interp1d
import math
import json


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


class RandomSwitchAxis:
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # print(sample.shape)
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)
        # print(sample.shape)
        return sample


class RotationAxis:
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample


class NormalDataset(Dataset):
    """Implements a map-style torch dataset."""

    def __init__(self,
                 X,
                 y=None,
                 pid=None,
                 name="",
                 transform=False,
                 transpose_channels_first=True):

        if transpose_channels_first:
            # PyTorch expects channels first data format
            X = np.transpose(X, (0, 2, 1))

        self.X = torch.from_numpy(X)  # convert data to Tensor

        if y is not None:
            self.y = torch.tensor(y)  # label should be a Tensor too
        else:
            self.y = None

        self.pid = pid

        if transform:
            self.transform = transforms.Compose(
                [RandomSwitchAxis(), RotationAxis()])
        else:
            self.transform = None

        print(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        if self.y is not None:
            y = self.y[idx]
        else:
            y = np.NaN

        if self.pid is not None:
            pid = self.pid[idx]
        else:
            pid = np.NaN

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, y, pid


def resize(x, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """

    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return x


def get_inverse_class_weights(y):
    """ Return a list with inverse class frequencies in y """
    import collections

    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Inverse class weights: ")
    print(weights)

    return weights


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
            self,
            patience=5,
            verbose=False,
            delta=0,
            path="checkpoint.pt",
            trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f" ({self.val_loss_min:.6f} --> {val_loss:.6f}). "
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), self.path)
        else:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def data_long2wide(X, times):
    """Convert long-format acc data to wide-format data.

    Parameters
    ----------
    X : np.ndarray N x 3
        Long-format data.
    times: np.ndarray N x 1

    Returns
    -------
    X : np.narray N x 3 x 900
        Wide-format data.
    times: np.ndarray N x 1
    """
    # get multiple of 900
    remainder = X.shape[0] % 900
    X = X[:-remainder]
    times = times[:-remainder]

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    x = x.reshape(-1, 900)
    y = y.reshape(-1, 900)
    z = z.reshape(-1, 900)
    times = times.reshape(-1, 900)

    X = np.stack((x, y, z), axis=1)
    times = times[:, 0]
    return X, times


class cnnLSTMInFerDataset:
    def __init__(self, X, pid=[], transform=None, target_transform=None):
        """
        X needs to be in N * Width
        Pid is a numpy array of size N
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.pid = pid
        self.unique_pid_list = np.unique(pid)
        self.transform = transform
        self.targetTransform = target_transform
        print(len(self.unique_pid_list))
        print("Total sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.unique_pid_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pid_of_choice = self.unique_pid_list[idx]
        sample_filter = self.pid == pid_of_choice
        sample = self.X[sample_filter, :]

        if self.transform:
            sample = self.transform(sample)
        sample = torch.as_tensor(sample)

        return sample, self.pid[sample_filter]


def cnn_lstm_infer_collate(batch):
    data = [item[0] for item in batch]
    data = torch.cat(data)

    pid = [item[1] for item in batch]
    pid = np.concatenate(pid)
    pid = torch.Tensor(pid)
    return [data, pid]


def prepare_infer_data_cnnlstm(val, my_device):
    x = val[0]
    pid = val[1]
    x = Variable(x)

    x = x.to(my_device, dtype=torch.float)
    seq_lengths = get_seq_lens(pid)
    return x, seq_lengths, pid


class RandomSwitchAxisTimeSeries(object):
    """
    Randomly switch the three axises for the raw files
    """

    def __call__(self, sample):
        # TIME_STEP * 3 * FEATURE_SIZE
        x = sample[:, 0, :]
        y = sample[:, 1, :]
        z = sample[:, 2, :]

        choice = random.randint(1, 6)
        if choice == 1:
            sample = torch.stack([x, y, z], dim=1)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=1)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=1)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=1)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=1)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=1)
        return sample


class RotationAxisTimeSeries(object):
    """
    Every sample belongs to one subject
    Rotation along an axis
    """

    def __call__(self, sample):
        # TIME_STEP * 3 * FEATURE_SIZE
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        sample = np.swapaxes(sample, 1, 2)
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 1, 2)
        return sample


class Permutation_TimeSeries(object):
    """
    Rearrange certain segments of the data
    """

    def __call__(self, sample):
        # TIME_STEP * 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 1, 2)
        # MIN one segment
        sample = np.array(
            [
                DA_Permutation(
                    xi, nPerm=max(
                        math.ceil(
                            np.random.normal(
                                2, 5)), 1))
                for xi in sample
            ]
        )

        sample = np.swapaxes(sample, 1, 2)
        sample = torch.tensor(sample)
        return sample


def get_seq_lens(pid_list):
    lens = np.where(pid_list[:-1] != pid_list[1:])[0]
    lens = np.concatenate((lens, [len(pid_list) - 1]))
    seq_lengths = []
    pre_len = -1
    for my_len in lens:
        seq_lengths.append(my_len - pre_len)
        pre_len = my_len

    return torch.LongTensor(seq_lengths)


# Taken from
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile is True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(
            np.random.randint(
                minSegLength,
                X.shape[0] - minSegLength,
                nPerm - 1)
        )
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]: segs[idx[ii] + 1], :]
        X_new[pp: pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new


class ClampTrans(object):
    """
    Randomly switch the three axises for the raw files
    """

    def __call__(self, sample):
        max_abs_val = 3
        sample = torch.clamp(sample, min=-max_abs_val, max=max_abs_val)
        return sample


def setup_transforms(augment_mode, is_train):
    my_transform = ClampTrans()
    if augment_mode == "cnn_lstm" and is_train:
        my_transform = transforms.Compose(
            [
                RandomSwitchAxisTimeSeries(),
                RotationAxisTimeSeries(),
                Permutation_TimeSeries(),
                ClampTrans(),
            ]
        )
    return my_transform
