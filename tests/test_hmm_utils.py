from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pytest

from asleep import hmm_utils
from asleep.get_sleep import load_model


def _fitted_smoother() -> hmm_utils.HMMSmoother:
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
    return hmm_utils.HMMSmoother().fit(y_pred, y_true)


def test_viterbi_smooths_isolated_flips() -> None:
    smoother = _fitted_smoother()
    noisy = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1])
    smoothed = smoother.predict(noisy)
    assert smoothed.shape == noisy.shape
    assert set(np.unique(smoothed)) <= {0, 1}
    # Fewer label changes after smoothing.
    assert np.count_nonzero(np.diff(smoothed)) < np.count_nonzero(np.diff(noisy))


def test_predict_by_group_matches_separate_calls() -> None:
    smoother = _fitted_smoother()
    a = np.array([0, 1, 0, 0, 1, 1])
    b = np.array([1, 1, 0, 1, 1])
    groups = np.array([7] * len(a) + [3] * len(b))
    together = smoother.predict(np.concatenate([a, b]), groups=groups)
    np.testing.assert_array_equal(
        together, np.concatenate([smoother.predict(a), smoother.predict(b)])
    )


def test_setstate_fills_attributes_missing_from_old_pickles() -> None:
    smoother = _fitted_smoother()
    state = dict(smoother.__dict__)
    for key in ("n_components", "hmm", "score"):
        del state[key]
    restored = hmm_utils.HMMSmoother.__new__(hmm_utils.HMMSmoother)
    restored.__setstate__(state)
    assert restored.n_components is None
    assert restored.hmm is None and restored.score is None
    x = np.array([0, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(restored.predict(x), smoother.predict(x))


def _dump_as_legacy_stepcount_pickle(obj: Any, path: Path) -> None:
    """Write ``obj`` so its class is referenced as ``stepcount.hmm_utils``."""
    cls = hmm_utils.HMMSmoother
    fake_pkg = ModuleType("stepcount")
    fake_pkg.__path__ = []
    fake_mod = ModuleType("stepcount.hmm_utils")
    fake_mod.HMMSmoother = cls  # type: ignore[attr-defined]
    saved = {n: sys.modules.get(n) for n in ("stepcount", "stepcount.hmm_utils")}
    original_module = cls.__module__
    sys.modules["stepcount"] = fake_pkg
    sys.modules["stepcount.hmm_utils"] = fake_mod
    cls.__module__ = "stepcount.hmm_utils"
    try:
        joblib.dump({"hmms": obj}, path)
    finally:
        cls.__module__ = original_module
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_load_model_reads_legacy_stepcount_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoother = _fitted_smoother()
    path = tmp_path / "legacy.joblib.lzma"
    _dump_as_legacy_stepcount_pickle(smoother, path)

    # Simulate an environment where stepcount is not installed.
    monkeypatch.delitem(sys.modules, "stepcount", raising=False)
    monkeypatch.delitem(sys.modules, "stepcount.hmm_utils", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockStepcount(), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError):
        joblib.load(path)

    loaded = load_model(path)["hmms"]
    assert type(loaded) is hmm_utils.HMMSmoother
    x = np.array([0, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(loaded.predict(x), smoother.predict(x))
    assert "stepcount" not in sys.modules
    assert "stepcount.hmm_utils" not in sys.modules


class _BlockStepcount:
    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "stepcount" or name.startswith("stepcount."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None
