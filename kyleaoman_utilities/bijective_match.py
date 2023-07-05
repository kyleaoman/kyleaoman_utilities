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


def best_match(df, on=None, avoidneg=False):
    candidate_matches, counts = np.unique(df[on], return_counts=True)
    mask = candidate_matches > 0 if avoidneg else np.s_[...]
    if not mask.any():
        return -999
    return candidate_matches[mask][counts[mask].argmax()]


def bijective_match(pid_1, fof_1, sub_1, pid_2, fof_2, sub_2):
    mask1 = np.logical_and(fof_1 >= 0, sub_1 >= 0)
    mask2 = np.logical_and(fof_2 >= 0, sub_2 >= 0)
    sim1 = pd.DataFrame(
        index=pid_1[mask1],
        data=dict(sid=cantor(fof_1[mask1], sub_1[mask1])),
    )
    sim2 = pd.DataFrame(
        index=pid_2[mask2],
        data=dict(sid=cantor(fof_2[mask2], sub_2[mask2])),
    )
    df = sim1.join(sim2, how="outer", lsuffix="1", rsuffix="2")
    _m1 = (
        df.groupby("sid1")
        .apply(best_match, **dict(on="sid2", avoidneg=True))
        .rename("sid2")
    )
    _m1 = _m1[_m1.index > 0]
    _m2 = (
        df.groupby("sid2")
        .apply(best_match, **dict(on="sid1", avoidneg=True))
        .rename("sid1")
    )
    _m2 = _m2[_m2.index > 0]

    _b1 = pd.merge(_m2, _m1, left_index=True, right_on="sid2", how="right")
    _b2 = pd.merge(_m1, _m2, left_index=True, right_on="sid1", how="right")
    sim1_matches = pd.DataFrame(
        index=_m1.index, data=dict(sid2=_m1, bijective=_b1.index == _b1["sid1"])
    )
    sim2_matches = pd.DataFrame(
        index=_m2.index, data=dict(sid1=_m2, bijective=_b2.index == _b2["sid2"])
    )
    return sim1_matches, sim2_matches


pid_1 = np.arange(1, 19)
fof_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])
sub_1 = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pid_2 = np.arange(1, 18)
fof_2 = np.array([1, 1, 1, 1, 1, 1, 1, -999, -999, 2, 2, -999, -999, -999, 3, 3, 3])
sub_2 = np.array([0, 0, 0, 0, 0, 0, 0, -999, -999, 0, 0, -999, -999, -999, 0, 0, 0])

bm1, bm2 = bijective_match(pid_1, fof_1, sub_1, pid_2, fof_2, sub_2)
print(np.vstack(icantor(bm1.index.to_numpy())))
print(np.vstack(icantor(bm1["sid2"].to_numpy())))
print(bm1["bijective"].to_numpy())
print(np.vstack(icantor(bm2.index.to_numpy())))
print(np.vstack(icantor(bm2["sid1"].to_numpy())))
print(bm2["bijective"].to_numpy())
