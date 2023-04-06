import os
import numpy as np
from glob import glob
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import utils  # helper functions -- check out utils.py

# For reproducibility
np.random.seed(42)


def load_all_and_make_windows(datafiles, N=999):

    def worker(datafile):
        X, Y, T = utils.make_windows(utils.load_data(datafile), winsec=30)
        pid = os.path.basename(datafile).split(".")[0]  # participant ID
        pid = np.asarray([pid] * len(X))
        return X, Y, T, pid

    results = Parallel(n_jobs=4)(
        delayed(worker)(datafile) for datafile in tqdm(datafiles[:N])
    )

    X = np.concatenate([result[0] for result in results])
    Y = np.concatenate([result[1] for result in results])
    T = np.concatenate([result[2] for result in results])
    pid = np.concatenate([result[3] for result in results])

    return X, Y, T, pid


# ------------------------------------------
# Process all files
# ------------------------------------------
DATAFILES = 'capture24/P[0-9][0-9][0-9].csv.gz'
X, Y, T, pid = load_all_and_make_windows(glob(DATAFILES))
# Save arrays for future use
os.makedirs("processed_data/", exist_ok=True)
np.save("processed_data/X.npy", X)
np.save("processed_data/Y.npy", Y)
np.save("processed_data/T.npy", T)
np.save("processed_data/pid.npy", pid)
