#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""One static plane-shaped SDF exposes a false sweep and a real subsequent motion."""
import json

import numpy as np
import torch
from ropesim import RopeBatch, sdf

NODES = np.array([[[x, 0., .01] for x in (-.04, -.02, 0., .02, .04)]], dtype=np.float32)
RESET = np.array([0., 0., 0.], dtype=np.float32)
PRE_RESET = RESET + [0., 0., .04]
TARGET = RESET + [0., 0., .009]
IDENTITY = np.array([0., 0., 0., 1.], dtype=np.float32)
GRID = sdf.from_function(lambda p: p[:, 2], [-.12, -.04, -.07], [.12, .04, .07], .005)

def owned(view):
    return torch.as_tensor(view).cpu().numpy().copy()

def trial(mode, *, second_pending=False):
    rope = RopeBatch.from_positions(NODES, k_stretch=500., k_bend=.01, density=.07, max_h=1/5760, num_itr=8,
                                    deterministic=True, straight_rest=True)
    rope.set_gravity(np.zeros(3, np.float32))
    rope.set_contact(radius=.004)
    rope.enable_contact_forces(True)
    assert rope.add_mesh(*GRID.as_args(), pos=RESET if mode=="direct" else PRE_RESET, quat=IDENTITY) == 0
    if second_pending:
        lower_plane = RESET + [0., 0., -.025]
        assert rope.add_mesh(*GRID.as_args(), pos=lower_plane, quat=IDENTITY) == 1
        rope.set_mesh_poses(1, RESET + [0., 0., .012], IDENTITY, teleport=False)
    rope.enable_contact_pair_capture(5*12*(1 + int(second_pending)))
    if mode == "commit":
        before = owned(rope.positions())
        initial_time = rope.time
        rope.commit_mesh_reset_pose(0, RESET, IDENTITY)
        assert rope.time == initial_time and np.array_equal(owned(rope.positions()), before)
        try:
            rope.contact_pair_samples()
        except ValueError:
            pass
        else:
            raise AssertionError("zero-time commit may not publish a measured contact packet")
    elif mode == "legacy":
        rope.set_mesh_poses(0, RESET, IDENTITY, teleport=True)
    rope.set_mesh_poses(0, TARGET, IDENTITY, teleport=False)
    rope.step_free(1/480, 12)
    result = rope.contact_pair_samples()
    assert not result["overflow"]
    if second_pending:
        assert np.any(result["indices"][:,2] == 1), "the second queued motion must reach the rod"
    pos, vel = owned(rope.positions()), owned(rope.velocities())
    if mode == "commit":
        rope.commit_mesh_reset_pose(0, RESET, IDENTITY)
        try:
            rope.contact_pair_samples()
        except ValueError:
            pass
        else:
            raise AssertionError("the preceding pair packet is stale after Reset")
        assert np.array_equal(owned(rope.positions()), pos)
        assert np.array_equal(owned(rope.velocities()), vel)
    return result, pos, vel

def main():
    commit, pos, vel = trial("commit")
    direct, pos_ref, vel_ref = trial("direct")
    legacy, _, _ = trial("legacy")
    assert np.array_equal(pos, pos_ref) and np.array_equal(vel, vel_ref)
    for key in ("indices", "vectors", "node_forces_on_rod_world_n", "mesh_wrenches_on_mesh_world"):
        a,b = commit[key],direct[key]
        if key in ("indices","vectors"):
            ka=np.lexsort(commit["indices"].T[::-1]); kb=np.lexsort(direct["indices"].T[::-1])
            a,b=a[ka],b[kb]
        assert np.array_equal(a,b), key
    assert commit["attempted"] and (commit["attempted"] != legacy["attempted"] or
           not np.array_equal(commit["mesh_wrenches_on_mesh_world"],legacy["mesh_wrenches_on_mesh_world"]))
    assert set(commit["indices"][:,0]) != {0}
    other, _, _ = trial("commit", second_pending=True)
    other_ref, _, _ = trial("direct", second_pending=True)
    np.testing.assert_array_equal(other["mesh_wrenches_on_mesh_world"],
                                  other_ref["mesh_wrenches_on_mesh_world"])
    print(json.dumps(dict(second_pending_target_preserved=True, direct_pairs=direct["attempted"],committed_pairs=commit["attempted"],
        legacy_pairs=legacy["attempted"],latest_contact_substep=int(commit["indices"][:,0].max()),
        earliest_contact_substep=int(commit["indices"][:,0].min()),zero_time_commit=True,
        controlled_sweep_matches_direct_initialization=True,synthetic_cuda_solver_fixture=True)))


if __name__ == "__main__":
    main()
