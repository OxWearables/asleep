import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm
import time
import os
import gzip
import os.path


# Model utils
from asleep.utils import cnnLSTMInFerDataset, cnn_lstm_infer_collate, \
    prepare_infer_data_cnnlstm, setup_transforms

# Torch
import torch
from torch.utils.data import DataLoader

from datetime import datetime
import torch.nn.functional as F

cuda = torch.cuda.is_available()
now = datetime.now()
my_abs_path = os.path.dirname(os.path.abspath(__file__))


def forward_batches(model, my_data_loader, my_device):
    model.eval()
    test_y_pred = []
    test_pid = []
    probs = []

    # accumulate all the losses into one
    for i, val in tqdm(enumerate(my_data_loader)):
        my_X, seq_lengths, my_pid = prepare_infer_data_cnnlstm(val, my_device)
        with torch.no_grad():
            logits = model(my_X, seq_lengths)
            batch_prob = F.softmax(logits, dim=1)
            probs.extend(batch_prob.cpu().detach().numpy())
            test_y_pred.extend(torch.max(logits, 1)[1].cpu().detach().numpy())
            test_pid.extend(my_pid)
    test_pid = np.stack(test_pid)
    test_y_pred = np.stack(test_y_pred)
    test_probs = np.stack(probs)

    return test_y_pred, test_pid, test_probs


def load_data(cfg):
    ####################
    #   Load data
    ###################
    start = time.time()
    print("Loading")
    if len(cfg.data.subject_file) > 0:
        print("Loading subject file!")
        X = np.load(cfg.data.subject_file)
        pid = np.ones(len(X))
    else:
        # prioritizes compressed format
        if os.path.exists(cfg.data.X_zip_path):
            f = gzip.GzipFile(cfg.data.X_zip_path, "r")
            X = np.load(f)
        else:
            X = np.load(cfg.data.X_path)
        pid = np.load(cfg.data.PID_path)
    end = time.time()
    print("Loading completed. Used: %d sec" % (end - start))
    print(X.shape)

    num_nights = len(np.unique(pid))
    print("#Number of nights %d" % num_nights)
    return X, pid


def setup_dataset(X, pid, cfg, is_train=False):
    num_workers = 0
    torch.multiprocessing.set_start_method("spawn", force=True)

    my_transform = setup_transforms(cfg.model.augment, is_train)
    dataset = cnnLSTMInFerDataset(
        X,
        pid=pid,
        transform=my_transform,
    )
    col_fn = cnn_lstm_infer_collate
    my_loader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        collate_fn=col_fn,
        shuffle=False,
        num_workers=num_workers,
    )
    return my_loader


def config_device(cfg):
    if cfg.gpu != 'cpu':
        my_device = str(cfg.gpu)
        print("pytorch device: " + my_device)
    else:
        my_device = "cpu"
        print("pytorch device defaulting to 'cpu'")
    return my_device


def setup_cnn(cfg, my_device, weight_path, local_repo_path=""):
    print("setting up cnn")
    if len(local_repo_path) > 0:
        print("access local repo")
        model = torch.hub.load(local_repo_path,
                               'sleepnet',
                               source='local',
                               num_classes=cfg.data.num_classes,
                               my_device=my_device,
                               lstm_nn_size=cfg.model.lstm_nn_size,
                               dropout_p=cfg.model.dropout_p,
                               bi_lstm=cfg.model.bi_lstm,
                               lstm_layer=cfg.model.lstm_layer,
                               local_weight_path=weight_path,
                               trust_repo=True
                               )
    else:
        print("access remote repo")

        repo = 'OxWearables/asleep'
        model = torch.hub.load(repo,
                               'sleepnet',
                               num_classes=cfg.data.num_classes,
                               my_device=my_device,
                               lstm_nn_size=cfg.model.lstm_nn_size,
                               dropout_p=cfg.model.dropout_p,
                               bi_lstm=cfg.model.bi_lstm,
                               lstm_layer=cfg.model.lstm_layer,
                               local_weight_path=weight_path,
                               trust_repo=True
                               )

    model.to(my_device, dtype=torch.float)
    return model


def get_unique_pid_in_place(my_pids):
    # Ensure that pid ordering is in place.
    # np.unique() will sort the array so that the y pred order will mismatch
    pre = -1
    pids = []
    for ele in my_pids:
        if ele != pre:
            if pre != -1:
                pids.append(pre)
            pre = ele
    pids.append(ele)
    return np.array(pids)


def align_output(y_red, real_pid, test_pid):
    my_ids = get_unique_pid_in_place(real_pid)
    aligned_pred = []

    for my_id in my_ids:
        subject_filter = test_pid == my_id
        subject_y_pred = y_red[subject_filter]
        aligned_pred.extend(subject_y_pred)
    return np.array(aligned_pred)


def sleepnet_inference(X, pid, weight_path, cfg, local_repo_path=""):
    start = time.time()
    my_device = config_device(cfg)

    model = setup_cnn(cfg, my_device, weight_path,
                      local_repo_path=local_repo_path)
    test_loader = setup_dataset(X, pid, cfg)

    test_y_pred, test_pid, test_probs = forward_batches(
        model, test_loader, my_device)
    end = time.time()

    # realign output
    aligned_y_pred = align_output(test_y_pred, pid, test_pid)
    aligned_y_probs = align_output(test_probs, pid, test_pid)

    if len(cfg.data.subject_file) > 0:
        prediction_path = os.path.join(
            "/".join(cfg.data.subject_file.split("/")[:-1]), "pred.npy"
        )
        prob_path = os.path.join(
            "/".join(cfg.data.subject_file.split("/")[:-1]), "pred_prob.npy"
        )
    else:
        prediction_path = cfg.data.y_pred_path
        prob_path = cfg.data.y_prob_path

    print("Save predictions to %s" % prediction_path)
    print("Save prediction probs to %s" % prob_path)
    np.save(prob_path, aligned_y_probs)
    np.save(prediction_path, aligned_y_pred)
    print("Time used % s" % str(end - start))
    return aligned_y_pred, test_pid


def start_sleep_net(X, pid, data_root, weight_path, device_id='cpu', local_repo_path=""):
    initialize(config_path="conf")
    cfg = compose(
        "config_eval",
        overrides=[
            "deployment=true",
            "data=deployment",
            "model=cnn_lstm_eval",
            "data.data_root=" + data_root,
            "gpu=" + str(device_id),
        ],
    )
    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    return sleepnet_inference(X, pid, weight_path, cfg, local_repo_path=local_repo_path)
