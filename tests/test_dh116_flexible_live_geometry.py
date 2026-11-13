"""CPU SDK API doubles exercise the same live geometry reader as the exchange."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding, DT
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import HAND_NAMES
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
from test_dh116_deformable_bimanual_step import Hand


def setup():
    clock = dict(physics_step=2, env_id=7, episode_uid="cpu", rod_uid="rod",
                 rod_tick=0, timestamp_s=2*DT, reset_id=1, solver_time_s=0.,
                 run_uuid="cpu", clock_valid=True)
    hands = [Hand(s, (7,)) for s in ("left", "right")]
    bindings, pad_map, shapes = [], {}, []
    for index, (side, hand) in enumerate(zip(("left", "right"), hands)):
        root = f"/World/envs/env_7/{side}Hand"
        hand.body_names = ["base_link", *[name.split(".")[1] for name in HAND_NAMES if name.startswith(side+".")]]
        n = len(hand.body_names)
        hand.transforms = np.zeros((1, n, 7), dtype=np.float32)
        hand.transforms[..., 2] = .35 + index*.01
        hand.transforms[..., 5:7] = np.sqrt(.5)
        hand.coms = np.zeros_like(hand.transforms)
        hand.coms[..., 0] = .01
        hand.root_physx_view.prim_paths = [root+"/wrist_fixed"]
        hand.root_physx_view.link_paths = [[f"{root}/{b}" for b in hand.body_names]]
        hand.root_physx_view.get_link_transforms = lambda hand=hand: hand.transforms
        hand.root_physx_view.get_coms = lambda hand=hand: hand.coms
        bindings.append(HandBinding(side, hand, {7: root+"/wrist_fixed"}, MountView(hand)))
        for bi, body in enumerate(hand.body_names[1:], 1):
            pad_map[side+"."+body] = dict(side=side, body_index=bi, prim_path=f"{root}/{body}")
        shapes.extend((index, bi, np.array([[0., 0., -.04], [0., .01, .01]]))
                      for bi in range(n))
    geometry = dict(identity=[deepcopy(clock)], pad_bindings=pad_map,
                    shapes=shapes, table_top_m=.30, native_qualified=False)
    sim = SimpleNamespace(current_time_step_index=2, current_time=2*DT)
    reader = LiveGeometryReader(sim, bindings, geometry, read_clock=lambda: deepcopy(clock))
    return reader, hands, sim, clock


def test_quaternion_conversion_and_com_offset_share_current_link_transform():
    reader, hands, sim, clock = setup()
    packet = reader()
    assert packet["physics_step"] == 2
    assert packet["root_poses_w"].shape == (1, 2, 7)
    assert packet["clearance_m"][0].tolist() == pytest.approx([.01, .02])
    name = "right.finger14_link"
    xyz, xyzw = packet["pad_poses"][name]
    assert xyz[2] == pytest.approx(.36)
    assert xyzw.tolist() == pytest.approx([0., 0., np.sqrt(.5), np.sqrt(.5)])
    assert packet["centers_of_mass"][name].tolist() == pytest.approx([0., .01, float(np.float32(.36))], abs=1e-8)
    assert not packet["native_clearance_qualified"]
    hands[1].transforms[..., 2] += .001
    assert packet["pad_poses"][name][0][2] == pytest.approx(.36)
    assert reader()["pad_poses"][name][0][2] == pytest.approx(.361)


def test_changed_instance_or_clock_refuses_geometry():
    reader, hands, sim, clock = setup()
    hands[0].root_physx_view.link_paths[0].reverse()
    with pytest.raises(ValueError, match="identity"):
        reader()
    hands[0].root_physx_view.link_paths[0].reverse()
    def advance():
        sim.current_time_step_index += 1
        return hands[1].coms
    hands[1].root_physx_view.get_coms = advance
    with pytest.raises(ValueError, match="clock"):
        reader()
