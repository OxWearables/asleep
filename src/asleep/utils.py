import numpy as np


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
