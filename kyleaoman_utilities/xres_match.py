import pandas as pd
from scipy.spatial import KDTree


def bijective_match_via_ic_coordinates(
    sim1_pids,
    sim1_hids,
    sim2_pids,
    sim2_hids,
    sim1_ic_pids,
    sim1_ic_coords,
    sim2_ic_pids,
    sim2_ic_coords,
    boxsize,
):
    """
    Match groups based on particle membership between two simulations.

    The two simulations must be of the same spatial region, but can have different
    particle IDs (e.g. because the resolution differs). Particles in the first
    simulation have their IDs mapped to that of the closest particle in the second
    simulation. Then match candidates for groups in the first simulation are
    identified by finding the group in the second simulation that has the most
    particles from each group in the first simulation. Match candidates for groups
    in the second simulation are similarly evaluated. Matches are returned, flagging
    those that are "bijective" (the candidate pairs in both directions match).

    Nearest neighbour searching is performed via a periodic KD-tree (via scipy).

    ID matching and grouping operations are implemented via pandas.

    When matching across resolution levels, the first simulation (``sim1``) should
    preferably be the lower resolution simulation, and the second (``sim2``) the
    higher resolution counterpart. Matching should normally be done on dark matter
    particles (stars don't exist in the ICs to have their IDs matched, and gas
    motions tend to be less predictable, plus gas tends to change particle type in
    star formation, etc.).

    Parameters
    ----------
    sim1_pids : np.array
        Unique IDs for simulation 1 particles.
    sim1_hids : np.array
        Group IDs for particles from simulation 1.
    sim2_pids : np.array
        Unique IDs for simulation 2 particles.
    sim2_hids : np.array
        Group IDs for particles from simulation 2.
    sim1_ic_pids : np.array
        Unique IDs (same set as ``sim1_pids``) in simulation 1 initial condition.
    sim1_ic_coords : np.array
        Particle coodinates in simulation 1 initial condition.
    sim2_ic_pids : np.array
        Unique IDs (same set as ``sim2_pids``) in simulation 2 initial condition.
    sim2_ic_coords : np.array
        Particle coodinates in simulation 2 initial condition.
    boxsize : np.array
        Maximum coordinate extent along each axis. Must have shape corresponding
        to dimension of coordinates (even if all boxsize lengths match). Used for
        setting the periodic size of the KD-tree.
    """
    pid_map = pd.DataFrame(
        index=pd.Index(sim1_ic_pids, name="pid_sim1"),
        data={
            "mapped_id": sim2_ic_pids[
                KDTree(sim2_ic_coords, boxsize=boxsize).query(
                    sim1_ic_coords, workers=8
                )[1]
            ]
        },
        dtype=pd.Int64Dtype(),
    )
    mask_sim1 = sim1_hids >= 0
    mask_sim2 = sim2_hids >= 0
    sim1_mapped = pd.DataFrame(
        index=sim1_pids[mask_sim1],
        data={"hid": sim1_hids[mask_sim1]},
        dtype=pd.Int64Dtype(),
    ).join(pid_map, how="outer", lsuffix="", rsuffix="")
    sim1_mapped.set_index("mapped_id", inplace=True, verify_integrity=False)
    crossmatched_hids = sim1_mapped.join(
        pd.DataFrame(
            index=sim2_pids[mask_sim2],
            data={"hid": sim2_hids[mask_sim2]},
            dtype=pd.Int64Dtype(),
        ),
        how="outer",
        lsuffix="_sim1",
        rsuffix="_sim2",
    )
    # use min() in case of multiple modes:
    sim1_match_candidates = crossmatched_hids.groupby("hid_sim1")["hid_sim2"].agg(
        lambda x: pd.Series.mode(x).min()
    )
    sim2_match_candidates = crossmatched_hids.groupby("hid_sim2")["hid_sim1"].agg(
        lambda x: pd.Series.mode(x).min()
    )
    sim1_merged_candidates = pd.merge(
        sim2_match_candidates,
        sim1_match_candidates,
        left_index=True,
        right_on="hid_sim2",
        how="right",
    )
    sim2_merged_candidates = pd.merge(
        sim1_match_candidates,
        sim2_match_candidates,
        left_index=True,
        right_on="hid_sim1",
        how="right",
    )
    return (
        sim1_match_candidates.to_frame().assign(
            bijective=sim1_merged_candidates.index == sim1_merged_candidates["hid_sim1"]
        ),
        sim2_match_candidates.to_frame().assign(
            bijective=sim2_merged_candidates.index == sim2_merged_candidates["hid_sim2"]
        ),
    )


def box_wrap(coords, boxsize):
    """
    Wrap coordinates periodically to be in [0, boxsize).

    Parameters
    ----------
    coords : np.array
        Coordinate array, can be (3, N) or (N, ) shaped.
    boxsize : float or np.array
        Boxsize, can be shape (3, ) or single-valued.

    Returns
    -------
    out : np.array
        The wrapped coordinate array.
    """
    return (coords + boxsize / 2.0) % boxsize
