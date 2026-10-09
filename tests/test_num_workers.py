import argparse
from typing import Any, List, cast

import numpy as np
from numpy.typing import NDArray
import pytest

import asleep.get_sleep as get_sleep
import asleep.models as models


class StopTest(Exception):
    pass


@pytest.mark.parametrize("num_workers", [0, 3])
def test_sleep_window_predict_passes_num_workers(
    monkeypatch: pytest.MonkeyPatch, num_workers: int
) -> None:
    observed: List[int] = []
    detector = object.__new__(models.SleepWindowSSL)
    detector.verbose = False
    detector.batch_size = 2

    monkeypatch.setattr(
        "asleep.models.sslmodel.NormalDataset", lambda *args, **kwargs: object()
    )

    def capture_loader(*args: object, **kwargs: object) -> None:
        observed.append(cast(int, kwargs["num_workers"]))
        raise StopTest

    monkeypatch.setattr(models, "DataLoader", capture_loader)

    with pytest.raises(StopTest):
        detector.predict(np.zeros((1, 2, 3)), num_workers=num_workers)

    assert observed == [num_workers]


def test_sleep_window_fit_passes_num_workers_to_both_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: List[int] = []
    detector = object.__new__(models.SleepWindowSSL)
    detector.verbose = False
    detector.batch_size = 2

    class Splitter:
        def split(self, *args: object, **kwargs: object) -> object:
            return iter([(np.array([0, 1]), np.array([2, 3]))])

    monkeypatch.setattr(models, "GroupShuffleSplit", lambda *args, **kwargs: Splitter())
    monkeypatch.setattr(
        "asleep.models.sslmodel.NormalDataset", lambda *args, **kwargs: object()
    )

    def capture_loader(*args: object, **kwargs: object) -> object:
        observed.append(cast(int, kwargs["num_workers"]))
        if len(observed) == 2:
            raise StopTest
        return object()

    monkeypatch.setattr(models, "DataLoader", capture_loader)

    values = np.arange(4)
    with pytest.raises(StopTest):
        detector.fit(values, values, groups=values, num_workers=4)

    assert observed == [4, 4]


def test_get_sleep_windows_forwards_cli_num_workers(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: List[int] = []

    class Detector:
        device = "cpu"

        def predict(self, data: NDArray[Any], num_workers: int) -> NDArray[Any]:
            observed.append(num_workers)
            return np.zeros(len(data))

    args = argparse.Namespace(
        outdir=str(tmp_path),
        force_run=True,
        force_download=False,
        pytorch_device="cpu",
        num_workers=5,
    )
    monkeypatch.setattr(get_sleep, "load_model", lambda *args, **kwargs: Detector())

    def stop_after_prediction(*args: object, **kwargs: object) -> None:
        raise StopTest

    monkeypatch.setattr("asleep.get_sleep.np.save", stop_after_prediction)

    with pytest.raises(StopTest):
        get_sleep.get_sleep_windows(
            np.zeros((2, 3, 4)),
            np.arange(2),
            np.array([False, False]),
            args,
        )

    assert observed == [5]
