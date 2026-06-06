"""Tests for cross-modality overlap clipping and baseline offset."""
import numpy as np
import pandas as pd

from cogload.data.alignment import clip_to_overlap


def _df(col: str, t_start: float, t_end: float, n: int) -> pd.DataFrame:
    t = np.linspace(t_start, t_end, n)
    return pd.DataFrame({"time": t, col: np.zeros(n)})


def test_clip_trims_to_overlap():
    bvp  = _df("bvp",  0, 100, 200)
    eda  = _df("eda",  5,  90,  50)
    temp = _df("temp", 2,  95,  50)
    b, e, t = clip_to_overlap(bvp, eda, temp, condition="cognitive_load", baseline_offset_s=0)
    assert b["time"].min() >= 5.0
    assert b["time"].max() <= 90.0


def test_baseline_offset_advances_start():
    bvp  = _df("bvp",  0, 200, 400)
    eda  = _df("eda",  0, 200, 100)
    temp = _df("temp", 0, 200, 100)
    b, _, _ = clip_to_overlap(bvp, eda, temp, condition="baseline", baseline_offset_s=30)
    assert b["time"].min() >= 30.0


def test_no_overlap_returns_none():
    bvp  = _df("bvp",  0,  10, 20)
    eda  = _df("eda",  50, 60, 10)
    temp = _df("temp", 0,  10, 10)
    b, e, t = clip_to_overlap(bvp, eda, temp, condition="cognitive_load", baseline_offset_s=0)
    assert b is None and e is None and t is None


def test_none_input_handled():
    bvp  = _df("bvp",  0, 100, 200)
    eda  = _df("eda",  0, 100,  50)
    b, e, t = clip_to_overlap(bvp, eda, None, condition="cognitive_load", baseline_offset_s=0)
    assert b is not None
    assert t is None
