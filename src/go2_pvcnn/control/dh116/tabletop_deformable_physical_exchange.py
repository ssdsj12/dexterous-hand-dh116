"""Single-environment flexible control exchange. An owner supplies live handles.

This code is exercised with CPU doubles. It does not create an app, choose
reset poses, grant native qualification, or feed rewards to PPO.
"""
from copy import deepcopy
import math

import numpy as np
import torch

from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
from .linear_coupling import MIMIC, active_limits
from .tabletop_deformable_bimanual_step import BimanualStepConnection, DT
from .tabletop_deformable_wrist_projection import project_policy_wrist_targets
from .tabletop_dual_hand_rod_runtime import SIDES, write_composer_reactions
from .tabletop_surface_contract import validate_actions
from .tabletop_task import joint_safety_failure


class FlexiblePhysicalExchange:
    """Preserve the previous measured rod reaction through the next SDK write."""

    def __init__(self, sim, rod, bindings, *, root_seed_w, read_geometry,
                 physics_budget_end, read_geometry_after_sdk=None):
        clock = rod.clock()
        if (clock["clock_valid"] is not True or clock["rod_tick"] != 0
                or clock["reset_id"] < 1 or clock["physics_step"] != sim.current_time_step_index
                or type(physics_budget_end) is not int or physics_budget_end <= clock["physics_step"]):
            raise ValueError("fresh synchronized reset and a future physics budget required")
        if tuple(binding.side for binding in bindings) != SIDES or any(b.mount is None for b in bindings):
            raise ValueError("two original six-axis mounts required")
        if (not isinstance(root_seed_w, torch.Tensor) or root_seed_w.shape != (1, 2, 7)
                or not root_seed_w.is_floating_point() or not bool(torch.isfinite(root_seed_w).all())):
            raise ValueError("finite original single-environment wrist seed required")
        env = clock["env_id"]
        self._composers = {}
        self._body_ids = {}
        for b in bindings:
            path = b.env_paths[env]
            view = b.hand.root_physx_view
            if list(view.prim_paths) != [path] or len(view.link_paths) != 1:
                raise ValueError("one actual articulation instance per side required")
            pads = [value for value in rod.bindings.values() if value["side"] == b.side]
            if len(pads) != 5 or any(view.link_paths[0][pad["body_index"]] != pad["prim_path"]
                                     for pad in pads):
                raise ValueError("same PhysX links required for action and rod reaction")
            self._body_ids[b.side] = [pad["body_index"] for pad in pads]
            self._composers[b.side] = b.hand.permanent_wrench_composer
        devices = {b.hand.data.joint_pos.device for b in bindings}
        if len(devices) != 1:
            raise ValueError("both hand composers must use the same device")
        self._device = devices.pop()
        self._sim, self._rod, self._bindings = sim, rod, tuple(bindings)
        self._seed = root_seed_w.detach().cpu().double().clone()
        if (not callable(read_geometry)
                or (read_geometry_after_sdk is not None and not callable(read_geometry_after_sdk))):
            raise ValueError("geometry callbacks must be callable")
        self._read_geometry = read_geometry
        self._read_geometry_after_sdk = read_geometry_after_sdk
        self._end = physics_budget_end
        self._control = 0
        self._closed = False
        self._active = False
        self._reset = tuple(clock[k] for k in ("run_uuid", "episode_uid", "reset_id", "rod_uid"))
        self._expected_sdk_step = clock["physics_step"]
        self._palms = torch.zeros((1, 2, 3), dtype=self._seed.dtype)
        self._angles = torch.zeros_like(self._palms)
        self._reaction = None

    def _clock(self, substep):
        clock = self._rod.clock()
        if (clock["clock_valid"] is not True
                or tuple(clock[k] for k in ("run_uuid", "episode_uid", "reset_id", "rod_uid")) != self._reset
                or clock["physics_step"] != self._sim.current_time_step_index
                or not math.isclose(clock["timestamp_s"], float(self._sim.current_time), rel_tol=0, abs_tol=2e-7)):
            raise ValueError("live rod and SDK identity or clock changed")
        return [{**clock, "policy_step": self._control, "substep": substep}]

    def _geometry(self, expected, *, previous_rod_clock=None):
        if previous_rod_clock is not None and self._read_geometry_after_sdk is not None:
            row = deepcopy(self._read_geometry_after_sdk(deepcopy(previous_rod_clock)))
            if (not isinstance(row, dict) or row.get("geometry_phase") != "sdk_complete_rod_pending"
                    or row.get("rod_source_physics_step") != previous_rod_clock["physics_step"]):
                raise ValueError("explicit preceding rod clock and handoff geometry phase required")
        else:
            row = deepcopy(self._read_geometry())
        if (not isinstance(row, dict) or type(row.get("physics_step")) is not int
                or row["physics_step"] != expected or self._sim.current_time_step_index != expected):
            raise ValueError("geometry must be measured on the current physical step")
        return row

    def _guard_failure(self, hands, routes=None):
        limits = np.stack(active_limits(), axis=-1)
        for side in SIDES:
            hand = hands[side]
            q = hand["q"][0]
            names = hand["joint_names"]
            mimic = [q[names.index(f)] - (ratio * q[names.index(m)] + offset)
                     for f, m, ratio, offset in MIMIC]
            failure = joint_safety_failure(q[hand["active_indices"]], limits, mimic)
            if failure:
                return f"{side}:{failure}"
        if routes is not None and any(np.linalg.norm(routes[side]["forces"], axis=-1).max() > 30
                                      for side in SIDES):
            return "excess_force"
        return None

    def _step(self, action, substep):
        read_clock = lambda: self._clock(substep)
        pre = read_clock()[0]
        expected = pre["physics_step"]
        geometry = self._geometry(expected)
        connection = BimanualStepConnection(env_ids=(pre["env_id"],),
                                            solver_substeps=self._rod._base.cfg["rod"]["solver_substeps"])
        request = connection.begin(self._bindings, action, read_clock=read_clock,
                                   palms=self._palms, angles=self._angles)
        projection = project_policy_wrist_targets(
            root_seed_w=self._seed, measured_base_w=geometry["root_poses_w"],
            common_lift_offset_m=self._seed.new_zeros(1), measured_clearance_m=geometry["clearance_m"],
            palm_offsets_m=self._palms, requested_palm_offsets_m=request["requested_palm_offsets_m"],
            angles_rad=self._angles, requested_angles_rad=request["requested_angles_rad"],
            allowed=torch.ones(1, dtype=torch.bool), live=torch.ones(1, dtype=torch.bool))
        source = self._reaction
        if source is None:
            if pre["rod_tick"] != 0:
                raise ValueError("reaction may be absent only at the fresh reset boundary")
            routes = {side: dict(body_ids=self._body_ids[side], forces=np.zeros((1, 5, 3)),
                                 torques=np.zeros((1, 5, 3)), is_global=True) for side in SIDES}
            source_step = None
        else:
            source_step, routes = source
            if source_step != expected:
                raise ValueError("queued reaction does not belong to the prior completed SDK step")
        write_composer_reactions(self._composers, routes,
            to_tensor=lambda value: torch.as_tensor(value, dtype=torch.float32, device=self._device))
        connection.submit(self._bindings, read_clock=read_clock,
            projected_root_poses=projection["projected_root_poses"],
            projected_angles=projection["projected_angles"],
            projection_source=projection["projection_source"])
        previous_rod_clock = self._rod.clock()
        if any(pre.get(key) != value for key, value in previous_rod_clock.items()):
            raise ValueError("rod clock changed during the action submission")
        sdk = step_thumb_physics(self._sim, expected, budget_end=self._end)
        for binding in self._bindings:
            binding.hand.update(DT)
        after = self._geometry(sdk, previous_rod_clock=previous_rod_clock)
        if self._rod.clock() != previous_rod_clock:
            raise ValueError("rod clock changed during the SDK-to-rod handoff")
        rod_row = self._rod.step(after["pad_poses"], after["centers_of_mass"], physics_step=sdk)
        action_row = connection.finish(self._bindings, read_clock=read_clock)
        guard_failure = self._guard_failure(action_row["post"]["hands"], rod_row["hand_wrenches_com_world"])
        self._palms, self._angles = projection["palm_offsets_m"], projection["projected_angles"]
        self._reaction = sdk, rod_row["hand_wrenches_com_world"]
        self._expected_sdk_step = sdk
        return dict(physics_step=sdk, policy_step=self._control, substep=substep,
                    reaction_source_physics_step=source_step, issued_composer_wrench=deepcopy(routes),
                    rod=deepcopy(rod_row), action_readback=action_row, wrist_projection=projection,
                    guard_failure=guard_failure,
                    native_qualification=False, teacher_admission=False)

    def control(self, action, *, observe):
        """Deliver each terminal row once; the owner persists each delivered row."""
        if self._closed or self._active:
            raise RuntimeError("closed or in-progress control exchange")
        validate_actions(action)
        if action.shape != (1, 24) or not callable(observe):
            raise ValueError("one canonical action and a physical row observer required")
        if self._sim.current_time_step_index != self._expected_sdk_step:
            self._closed = True
            raise RuntimeError("external advance of the owned SDK clock")
        if self._expected_sdk_step + 8 > self._end:
            raise RuntimeError("whole control interval exceeds the physics budget")
        action = action.detach().cpu().double().clone()
        self._active = True
        try:
            reason = None
            for substep in range(8):
                row = self._step(action, substep)
                observed_reason = observe(deepcopy(row))
                reason = row["guard_failure"] or observed_reason
                if reason is not None:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("terminal observation must be a nonempty reason or None")
                    break
            result = dict(control_step=self._control, physical_samples=substep + 1,
                          termination_reason=reason, training_reward=None, teacher_admission=False)
            self._control += 1
            self._closed = reason is not None
            return result
        except BaseException:
            self._closed = True
            raise
        finally:
            self._active = False
