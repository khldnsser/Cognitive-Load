"""Tests for timestamp-based sliding window."""
import numpy as np
import pandas as pd
import pytest

from cogload.features.windowing import sliding_windows


def _make_signal(col: str, start: float, duration: float, hz: float) -> pd.DataFrame:
    n = int(duration * hz)
    t = start + np.arange(n) / hz
    return pd.DataFrame({"time": t, col: np.random.randn(n)})


def test_window_count():
    bvp  = _make_signal("bvp",  0, 100, 64)
    eda  = _make_signal("eda",  0, 100,  4)
    temp = _make_signal("temp", 0, 100,  4)
    windows = list(sliding_windows(bvp, eda, temp, window_s=30, hop_s=15))
    # overlap = 100s; first window at 0, last at ≤ 70 → (100-30)/15 + 1 = 5 windows
    assert len(windows) == 5


def test_slice_lengths_match_window():
    bvp  = _make_signal("bvp",  0, 60, 64)
    eda  = _make_signal("eda",  0, 60,  4)
    temp = _make_signal("temp", 0, 60,  4)
    for t_start, slices in sliding_windows(bvp, eda, temp, window_s=30, hop_s=30):
        assert len(slices["bvp"])  == pytest.approx(30 * 64, abs=2)
        assert len(slices["eda"])  == pytest.approx(30 *  4, abs=2)
        assert len(slices["temp"]) == pytest.approx(30 *  4, abs=2)


def test_no_windows_when_signal_too_short():
    bvp  = _make_signal("bvp",  0, 10, 64)
    eda  = _make_signal("eda",  0, 10,  4)
    temp = _make_signal("temp", 0, 10,  4)
    windows = list(sliding_windows(bvp, eda, temp, window_s=30, hop_s=15))
    assert len(windows) == 0


def test_none_signal_handled():
    bvp  = _make_signal("bvp",  0, 60, 64)
    eda  = _make_signal("eda",  0, 60,  4)
    windows = list(sliding_windows(bvp, eda, None, window_s=30, hop_s=30))
    assert len(windows) > 0
    for _, slices in windows:
        assert slices["temp"] is None
