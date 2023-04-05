import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from utils import NormalDataset, resize, get_inverse_class_weights, EarlyStopping
from tqdm import tqdm



"""
This script will train a binary sleep classifier using the sleep diary data from capture24.
We use the pre-trained model with HMM smoothing to do the classification.

It will save the network weights for harnet30 and HMM.

Usage: 
```
python 
```



"""


def load_data():
    root = '/data/UKBB/capture24_30s/'  # change this path if needed

    X = np.load(os.path.join(root, 'X.npy'))  # accelerometer data
    Y = np.load(os.path.join(root, 'Y_Walmsley.npy'))  # true labels
    pid = np.load(os.path.join(root, 'pid.npy'))  # participant IDs

    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')  # same shape as pid
    print(f'Label distribution:\n{pd.Series(Y).value_counts()}')

    # The original labels in Y are in categorical format (e.g.: 'light', 'sleep', etc). PyTorch expects numerical labels (e.g.: 0, 1, etc).
    # LabelEncoder transforms categorical labels -> numerical.
    # After obtaining the test predictions, you can use le.inverse_transform(y) to go from numerical -> categorical (the fitted le object is returned at the end of this function)
    le = LabelEncoder()
    le.fit(np.unique(Y))

    y = le.transform(Y)

    ## add manual conversion based on a dictionary
    print(f'Original labels: {le.classes_}')
    print(f'Transformed labels: {le.transform(le.classes_)}')

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

    # generate train/test splits
    folds = GroupShuffleSplit(
        1, test_size=0.2, random_state=42
    ).split(X, y, groups=pid)
    train_idx, test_idx = next(folds)

    x_train = X[train_idx]
    y_train = y[train_idx]
    group_train = pid[train_idx]

    x_test = X[test_idx]
    y_test = y[test_idx]
    group_test = pid[test_idx]

    return (
        x_train, y_train, group_train,
        x_test, y_test, group_test,
        le
    )


def train(model, train_loader, val_loader, my_device, weights=None):
    """
    Iterate over the training dataloader and train a pytorch model.
    After each epoch, validate model and early stop when validation loss function bottoms out.

    Trained model weights will be saved to disk (state_dict.pt).

    :param nn.Module model: pytorch model
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param str my_device: pytorch map device.
    :param weights: training class weights (to enable weighted loss function)
    """

    state_dict = 'state_dict.pt'
    num_epoch = 100

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, amsgrad=True
    )

    if weights:
        weights = torch.FloatTensor(weights).to(my_device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(
        patience=5, path=state_dict, verbose=True
    )

    for epoch in range(num_epoch):
        model.train()
        train_losses = []
        train_acces = []
        for i, (x, y, _) in enumerate(tqdm(train_loader)):
            x.requires_grad_(True)
            x = x.to(my_device, dtype=torch.float)
            true_y = y.to(my_device, dtype=torch.long)

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

        val_loss, val_acc = _validate_model(model, val_loader, my_device, loss_fn)

        epoch_len = len(str(num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{num_epoch:>{epoch_len}}] | "
            + f"train_loss: {np.mean(train_losses):.3f} | "
            + f"train_acc: {np.mean(train_acces):.3f} | "
            + f"val_loss: {val_loss:.3f} | "
            + f"val_acc: {val_acc:.2f}"
        )

        early_stopping(val_loss, model)
        print(print_msg)

        if early_stopping.early_stop:
            print('Early stopping')
            print(f'SSLNet weights saved to {state_dict}')
            break


def _validate_model(model, val_loader, my_device, loss_fn):
    """ Iterate over a validation data loader and return mean model loss and accuracy. """
    model.eval()
    losses = []
    acces = []
    for i, (x, y, _) in enumerate(val_loader):
        with torch.inference_mode():
            x = x.to(my_device, dtype=torch.float)
            true_y = y.to(my_device, dtype=torch.long)

            logits = model(x)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            val_acc = torch.sum(pred_y == true_y)
            val_acc = val_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach())
            acces.append(val_acc.cpu().detach())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)


def predict(model, data_loader, my_device):
    """
    Iterate over the dataloader and do inference with a pytorch model.

    :param nn.Module model: pytorch Module
    :param data_loader: pytorch dataloader
    :param str my_device: pytorch map device
    :return: true labels, model predictions, pids
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """


    predictions_list = []
    true_list = []
    pid_list = []
    model.eval()

    for i, (x, y, pid) in enumerate(tqdm(data_loader)):
        with torch.inference_mode():
            x = x.to(my_device, dtype=torch.float)
            logits = model(x)
            true_list.append(y)
            pred_y = torch.argmax(logits, dim=1)
            predictions_list.append(pred_y.cpu())
            pid_list.extend(pid)
    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)

    return (
        torch.flatten(true_list).numpy(),
        torch.flatten(predictions_list).numpy(),
        np.array(pid_list),
    )


def main():
    (
        x_train, y_train, group_train,
        x_test, y_test, group_test,
        le
    ) = load_data()

    repo = 'OxWearables/ssl-wearables'
    my_device = 'cuda:0'

    # load the pretrained model
    sslnet: nn.Module = torch.hub.load(repo, 'harnet30', trust_repo=True, class_num=4, pretrained=True)
    sslnet.to(my_device)

    # construct dataloaders
    train_dataset = NormalDataset(x_train, y_train, group_train, name="training", transform=True)
    test_dataset = NormalDataset(x_test, y_test, group_test, name="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2000,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2000,
        shuffle=False,
        num_workers=0,
    )

    train(sslnet, train_loader, test_loader, my_device, get_inverse_class_weights(y_train))


if __name__ == '__main__':
    main()


