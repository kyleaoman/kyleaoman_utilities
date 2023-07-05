import numpy as np
import pandas as pd


def cantor(x, y):
    return np.array((x + y + 1) * (x + y) / 2 + y, dtype=int)


def icantor(z):
    w = np.floor((np.sqrt(8 * z + 1) - 1) / 2)
    t = (w * w + w) / 2
    y = z - t
    x = w - y
    return np.array(x, dtype=int), np.array(y, dtype=int)


def best_match(df, on=None, filterneg=False):
    candidate_matches, counts = np.unique(df[on], return_counts=True)
    if filterneg:
        mask = candidate_matches > 0
        if not mask.any():
            return -1
    else:
        mask = np.s_[...]
    return candidate_matches[mask][counts[mask].argmax()]


def bijective_match(pid_1, fof_1, sub_1, pid_2, fof_2, sub_2):
    mask1 = np.logical_and.reduce((fof_1 >= 1, sub_1 >= 0, sub_1 != 2**30))
    mask2 = np.logical_and.reduce((fof_2 >= 1, sub_2 >= 0, sub_2 != 2**30))
    sim1 = pd.DataFrame(
        index=pid_1[mask1],
        data=dict(sid=cantor(fof_1[mask1], sub_1[mask1])),
    )
    sim2 = pd.DataFrame(
        index=pid_2[mask2],
        data=dict(sid=cantor(fof_2[mask2], sub_2[mask2])),
    )
    df = sim1.join(sim2, how="outer", lsuffix="1", rsuffix="2").fillna(
        -1, downcast="infer"
    )
    _m1 = (
        df.groupby("sid1")
        .apply(best_match, **dict(on="sid2", filterneg=True))
        .rename("sid2")
    )
    _m1 = _m1[_m1.index > 0]
    _m2 = (
        df.groupby("sid2")
        .apply(best_match, **dict(on="sid1", filterneg=True))
        .rename("sid1")
    )
    _m2 = _m2[_m2.index > 0]
    _b1 = pd.merge(_m2, _m1, left_index=True, right_on="sid2", how="right").fillna(
        -1, downcast="infer"
    )
    _b2 = pd.merge(_m1, _m2, left_index=True, right_on="sid1", how="right").fillna(
        -1, downcast="infer"
    )
    sim1_matches = pd.DataFrame(
        index=_m1.index, data=dict(sid2=_m1, bijective=_b1.index == _b1["sid1"])
    )
    sim2_matches = pd.DataFrame(
        index=_m2.index, data=dict(sid1=_m2, bijective=_b2.index == _b2["sid2"])
    )
    return (
        np.rec.fromarrays(
            icantor(sim1_matches.index.to_numpy())
            + icantor(sim1_matches["sid2"].to_numpy())
            + (sim1_matches["bijective"],),
            dtype=[
                ("fof_1", int),
                ("sub_1", int),
                ("fof_2", int),
                ("sub_2", int),
                ("bijective", bool),
            ],
        ),
        np.rec.fromarrays(
            icantor(sim2_matches.index.to_numpy())
            + icantor(sim2_matches["sid1"].to_numpy())
            + (sim2_matches["bijective"],),
            dtype=[
                ("fof_2", int),
                ("sub_2", int),
                ("fof_1", int),
                ("sub_1", int),
                ("bijective", bool),
            ],
        ),
    )
