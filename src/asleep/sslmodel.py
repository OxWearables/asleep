""" Helper classes and functions for the SSL model """

from __future__ import annotations

from os import PathLike
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from asleep.utils import EarlyStopping, RandomSwitchAxis, RotationAxis


verbose = False
torch_cache_path = Path(__file__).parent / 'torch_hub_cache'
Array = NDArray[Any]
Device = Union[str, torch.device]


class NormalDataset(Dataset[Tuple[torch.Tensor, Any, Any]]):
    def __init__(self,
                 X: Array,
                 y: Optional[Array] = None,
                 pid: Optional[Array] = None,
                 name: str = "",
                 augmentation: bool = False,
                 transpose_channels_first: bool = True) -> None:

        X = X.astype(
            "f4"
        )  # PyTorch defaults to float32

        if transpose_channels_first:
            X = np.transpose(X, (0, 2, 1))
        self.X = torch.from_numpy(X)
        self.y: Optional[torch.Tensor]

        if y is not None:
            self.y = torch.tensor(y)
        else:
            self.y = None

        self.pid = pid

        if augmentation:
            self.transform = transforms.Compose(
                [RandomSwitchAxis(), RotationAxis()])
        else:
            self.transform = None

        if verbose:
            print(f"{name} set sample count: {len(self.X)}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: Any) -> Tuple[torch.Tensor, Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]

        y: Any = self.y[idx] if self.y is not None else np.nan

        if self.pid is not None:
            pid = self.pid[idx]
        else:
            pid = np.NaN

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, y, pid


def get_sslnet(tag: str = 'v1.0.0', pretrained: bool = False) -> nn.Module:
    """
    Load and return the Self Supervised Learning (SSL) model from pytorch hub.
    :param str tag: Tag on the ssl-wearables repo to check out
    :param bool pretrained: Initialise the model with UKB self-supervised pretrained weights.
    :return: pytorch SSL model
    :rtype: nn.Module
    """

    repo_name = 'ssl-wearables'
    repo = f'OxWearables/{repo_name}:{tag}'

    if not torch_cache_path.exists():
        Path.mkdir(torch_cache_path, parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_cache_path))

    # find repo cache dir that matches repo name and tag
    cache_dirs = [f for f in torch_cache_path.iterdir() if f.is_dir()]
    cached_repo_path = next(
        (f for f in cache_dirs if repo_name in f.name and tag in f.name),
        None)

    if cached_repo_path is None:
        repo_location = repo
        source = 'github'
    else:
        repo_location = str(cached_repo_path)
        source = 'local'
        if verbose:
            print(f'Using local {repo_location}')

    sslnet: nn.Module = torch.hub.load(
        repo_location,
        'harnet30',
        trust_repo=True,
        source=source,
        class_num=4,
        pretrained=pretrained,
        verbose=verbose)
    return sslnet


def predict(
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: Device,
    output_logits: bool = False,
    name: str = 'train',
) -> Tuple[Array, Array, Array]:
    """
    Iterate over the dataloader and do prediction with a pytorch model.
    :param nn.Module model: pytorch Module
    :param DataLoader data_loader: pytorch dataloader
    :param str device: pytorch map device
    :param bool output_logits: When True, output the raw outputs (logits)
        from the last layer (before classification).
        When False, argmax the logits and output a classification scalar.
    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    predictions_list: List[torch.Tensor] = []
    true_list: List[torch.Tensor] = []
    pid_list: List[Any] = []
    model.eval()

    for i, (x, y, pid) in enumerate(
            tqdm(data_loader, mininterval=60, disable=not verbose)):
        with torch.inference_mode():
            x = x.to(device, dtype=torch.float)
            logits = model(x)
            true_list.append(y)
            if output_logits:
                predictions_list.append(logits.cpu())
            else:
                pred_y = torch.argmax(logits, dim=1)
                predictions_list.append(pred_y.cpu())
            pid_list.extend(pid)

    predictions = torch.cat(predictions_list)
    true_values = torch.cat(true_list)

    if output_logits:
        return (
            torch.flatten(true_values).numpy(),
            predictions.numpy(),
            np.array(pid_list),
        )
    else:
        return (
            torch.flatten(true_values).numpy(),
            torch.flatten(predictions).numpy(),
            np.array(pid_list),
        )


def train(
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: Device,
        class_weights: Optional[Sequence[float]] = None,
        weights_path: Union[str, PathLike[str]] = 'weights.pt',
        num_epoch: int = 100,
        learning_rate: float = 0.0001,
        patience: int = 5,
) -> nn.Module:
    """
    Iterate over the training dataloader and train a pytorch model.
    After each epoch, validate model and early stop when validation
    loss function bottoms out.
    Trained model weights will be saved to disk (weights_path).
    :param nn.Module model: pytorch model
    :param DataLoader train_loader: training data loader
    :param DataLoader val_loader: validation data loader
    :param str device: pytorch map device
    :param class_weights: Array of training class weights to use with
                        weighted cross entropy loss.
                        Leave empty to use unweighted loss.
    :param weights_path: save location for the trained weights (state_dict)
    :param num_epoch: number of training epochs
    :param learning_rate: Adam learning rate
    :param patience: early stopping patience
    """
    optimizer = Adam(
        model.parameters(), lr=learning_rate, amsgrad=True
    )

    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(
        patience=patience, path=weights_path, verbose=verbose, trace_func=print
    )

    for epoch in range(num_epoch):
        model.train()
        train_losses: List[torch.Tensor] = []
        train_acces: List[torch.Tensor] = []
        for i, (x, y, _) in enumerate(tqdm(train_loader, disable=not verbose)):
            x.requires_grad_(True)
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.long)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, true_y)
            loss.backward()
            optimizer.step()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach())
            train_acces.append(train_acc.cpu().detach())

        val_loss, val_acc = _validate_model(model, val_loader, device, loss_fn)

        epoch_len = len(str(num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{num_epoch:>{epoch_len}}] ",
            f"train_loss: {np.mean(train_losses):.3f} ",
            f"train_acc: {np.mean(train_acces):.3f} ",
            f"val_loss: {val_loss:.3f} | ",
            f"val_acc: {val_acc:.2f}"
        )

        early_stopping(val_loss, model)

        if verbose:
            print(print_msg)

        if early_stopping.early_stop:
            if verbose:
                print('Early stopping')
                print(f'SSLNet weights saved to {weights_path}')
            break

    return model


def _validate_model(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: Device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """ Iterate over a validation data loader and return
        mean model loss and accuracy. """
    model.eval()
    losses: List[torch.Tensor] = []
    acces: List[torch.Tensor] = []
    for i, (x, y, _) in enumerate(val_loader):
        with torch.inference_mode():
            x = x.to(device, dtype=torch.float)
            true_y = y.to(device, dtype=torch.long)

            logits = model(x)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            val_acc = torch.sum(pred_y == true_y)
            val_acc = val_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach())
            acces.append(val_acc.cpu().detach())
    loss_values: Array = np.array(losses)
    accuracy_values: Array = np.array(acces)
    return float(np.mean(loss_values)), float(np.mean(accuracy_values))
