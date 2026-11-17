"""Read current full-hand clearance and pad poses for the physical exchange."""
from copy import deepcopy
import math

import numpy as np
import torch

from scripts.verify_dh116_deformx_rigid_bridge import rotation_wxyz
from .tabletop_batch_geometry import collision_clearance
from .tabletop_deformable_bimanual_step import DT, _owned
from .tabletop_dual_hand_rod_runtime import HAND_NAMES, SIDES


class LiveGeometryReader:
    """Keep the inspected scene mapping fixed and read only current SDK buffers.

    Clearance remains sourced from the original USD collision envelopes. This
    reader does not replace native cooked-hull qualification or move the SDK.
    """

    def __init__(self, sim, bindings, inspected_geometry, *, read_clock):
        self._sim, self._bindings = sim, tuple(bindings)
        self._read_clock = read_clock
        clock = self._clock()
        if (tuple(b.side for b in bindings) != SIDES
                or any(b.mount is None for b in bindings)
                or bindings[0].hand is bindings[1].hand):
            raise ValueError("two distinct mounted hand identities required")
        if (inspected_geometry["identity"][0]["physics_step"] != clock["physics_step"]
                or any(inspected_geometry["identity"][0].get(k) != v for k, v in clock.items())
                or inspected_geometry["table_top_m"] != .30):
            raise ValueError("same-clock original scene with 0.30 m table top required")
        self._env = clock["env_id"]
        self._pads = deepcopy(inspected_geometry["pad_bindings"])
        if set(self._pads) != set(HAND_NAMES):
            raise ValueError("all ten original pad identities required")
        self._shapes = deepcopy(inspected_geometry["shapes"])
        self._ids = tuple(self._identity(b) for b in self._bindings)
        if {h for h, _, _ in self._shapes} != {0, 1}:
            raise ValueError("both hands require inspected source collision shapes")
        for index, b in enumerate(self._bindings):
            if b.mount.hand is not b.hand or self._ids[index][2][b.mount.base_id] != "base_link":
                raise ValueError("original mounted base_link identity required")
            for key, pad in self._pads.items():
                if key.startswith(b.side + "."):
                    i = pad["body_index"]
                    if (pad["side"] != b.side or self._ids[index][2][i] != key.split(".")[1]
                            or self._ids[index][3][i] != pad["prim_path"]):
                        raise ValueError("inspected pad link identity differs")

    def _clock(self, previous_rod_clock=None):
        row = deepcopy(self._read_clock())
        sdk_step = self._sim.current_time_step_index
        if row["clock_valid"] is not True or type(row["physics_step"]) is not int:
            raise ValueError("live geometry requires a valid rod clock")
        if previous_rod_clock is None:
            expected_sdk_step = row["physics_step"]
        else:
            if row != previous_rod_clock:
                raise ValueError("rod clock changed before the SDK-to-rod handoff")
            expected_sdk_step = row["physics_step"] + 1
        if (sdk_step != expected_sdk_step
                or not math.isclose(float(self._sim.current_time), sdk_step * DT,
                                    abs_tol=2e-7, rel_tol=0)):
            raise ValueError("live geometry SDK clock differs")
        return row

    def _identity(self, binding):
        hand, view = binding.hand, binding.hand.root_physx_view
        if list(view.prim_paths) != [binding.env_paths[self._env]] or len(view.link_paths) != 1:
            raise ValueError("live geometry articulation identity differs")
        names, paths = tuple(hand.body_names), tuple(view.link_paths[0])
        if len(names) != len(paths) or len(set(names)) != len(names):
            raise ValueError("complete named live link identity required")
        return id(view), tuple(view.prim_paths), names, paths

    def __call__(self):
        return self._read()

    def read_after_sdk_step(self, previous_rod_clock):
        """Read new hand poses while the rod still owns its preceding boundary."""
        if not isinstance(previous_rod_clock, dict):
            raise ValueError("explicit preceding rod clock required")
        return self._read(deepcopy(previous_rod_clock))

    def _read(self, previous_rod_clock=None):
        before = self._clock(previous_rod_clock)
        sdk_step = self._sim.current_time_step_index
        all_poses, base_poses, pad_poses, centers = [], [], {}, {}
        for index, b in enumerate(self._bindings):
            if self._identity(b) != self._ids[index]:
                raise ValueError("live geometry link identity changed")
            view = b.hand.root_physx_view
            xyzw = _owned(view.get_link_transforms())
            local_com = _owned(view.get_coms())
            shape = (1, len(self._ids[index][2]), 7)
            if (xyzw.shape != shape or local_com.shape != shape
                    or not np.isfinite(xyzw).all() or not np.isfinite(local_com).all()
                    or not np.allclose(np.linalg.norm(xyzw[..., 3:], axis=-1), 1, atol=1e-5, rtol=0)):
                raise ValueError("finite PhysX link pose and local COM buffers required")
            wxyz = xyzw[..., [0, 1, 2, 6, 3, 4, 5]].astype(float)
            all_poses.append(wxyz)
            base_poses.append(wxyz[0, b.mount.base_id])
            for name, row in self._pads.items():
                if row["side"] == b.side:
                    i = row["body_index"]
                    xyz = wxyz[0, i, :3].copy()
                    pad_poses[name] = xyz, xyzw[0, i, 3:].astype(float)
                    centers[name] = xyz + rotation_wxyz(wxyz[0, i, 3:]) @ local_com[0, i, :3]
        if self._clock(previous_rod_clock) != before or any(self._identity(b) != self._ids[i]
                                          for i, b in enumerate(self._bindings)):
            raise ValueError("live geometry clock or link identity changed during readback")
        clearances, lowest = collision_clearance(np.stack(all_poses, axis=1), self._shapes)
        if not np.isfinite(clearances).all():
            raise ValueError("nonfinite full-hand clearance")
        return dict(physics_step=sdk_step, rod_source_physics_step=before["physics_step"],
                    geometry_phase=("synchronized" if previous_rod_clock is None
                                    else "sdk_complete_rod_pending"),
                    root_poses_w=torch.from_numpy(np.array([base_poses])),
                    clearance_m=torch.from_numpy(clearances), pad_poses=pad_poses,
                    centers_of_mass=centers, lowest_shape_index=lowest,
                    source="same_clock_physx_link_transforms_local_com_and_original_collision_envelopes",
                    native_clearance_qualified=False)
