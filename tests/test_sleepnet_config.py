from __future__ import annotations

import warnings

from asleep.sleepnet import load_eval_config


def test_load_eval_config_no_hydra_warning_and_repeatable() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = load_eval_config("/tmp/data", "cpu")
        # A second call must not raise "GlobalHydra is already initialized".
        second = load_eval_config("/tmp/other", "cpu")

    assert first.deployment is True
    assert first.data.data_root == "/tmp/data"
    assert second.data.data_root == "/tmp/other"
    assert str(first.gpu) == "cpu"
