"""Bounded device-side evidence; copied to disk only on a physical failure."""
import json
import math
from pathlib import Path

import torch


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    return value


class PhysicsFaultTrace:
    def __init__(self, path, capacity=32):
        if capacity < 1:
            raise ValueError('fault trace capacity must be positive')
        self.path = Path(path)
        self.capacity = capacity
        self.count = 0
        self.buffers = {}
        self.substeps = [0] * capacity
        self.joint_names = None

    def record(self, env, measurement):
        """Capture after physics, before reward/termination can overwrite state."""
        world, task = env.world, env.task
        values = dict(joint_position_rad=world.q, joint_velocity_rad_s=world.qd,
                      command_rad=task.command, phase=task.phase, elapsed_s=task.elapsed,
                      episode=env.episodes, pending_reset=env.pending,
                      lift_offset_m=task.offset)
        for key in ('palm_targets', 'palm_actions', 'f_target', 'f_requested',
                    'f_filtered', 'f_offset', 'f_blocked', 'pressure_limited', 'paused'):
            if hasattr(task, key):
                values[key] = getattr(task, key)
        if hasattr(task, 'palm_reposition'):
            values['reposition_state'] = task.palm_reposition.state
            for key in ('goal', 'hold', 'clock', 'dwell', 'aborted'):
                if hasattr(task.palm_reposition, key):
                    values['reposition_' + key] = getattr(task.palm_reposition, key)
        for key in ('q', 'qd'):
            values['active_' + key] = measurement[key]
        for key in ('hand_clearance', 'clearance', 'bottom', 'mimic_errors',
                    'hand_object_loads', 'object_link_loads', 'adaptive_palm_offset',
                    'vertical_hand_force_n', 'contact_center_w', 'contact_center_relative_m',
                    'contact_normal_w', 'contact_normal_load_n', 'contact_valid',
                    'contact_all_below_centerline', 'contact_count'):
            if key in measurement:
                values[key] = measurement[key]
        for key, label in (('palms', 'commanded_palm_offset_m'), ('roots', 'root_pose_w'),
                           ('root_targets', 'root_target_w'), ('vertical_components', 'vertical_components_n'),
                           ('gains', 'grip_stiffness'), ('previous_gains', 'previous_grip_stiffness'),
                           ('commands', 'applied_command_rad'), ('wp', 'object_position_w'),
                           ('wq', 'object_quat_w'), ('wv', 'object_velocity_w')):
            if hasattr(world, key):
                values[label] = getattr(world, key)
        slot = self.count % self.capacity
        for name, value in values.items():
            if name not in self.buffers:
                self.buffers[name] = torch.empty((self.capacity, *value.shape),
                                                device=value.device, dtype=value.dtype)
            self.buffers[name][slot].copy_(value.detach())
        if self.joint_names is None and hasattr(world, 'hands'):
            self.joint_names = [list(hand.joint_names) for hand in world.hands]
        self.substeps[slot] = env.substep
        self.count += 1

    def write(self, ids, context):
        frames = []
        for index in range(max(0, self.count - self.capacity), self.count):
            slot = index % self.capacity
            frame = dict(sequence=index, physics_substep=self.substeps[slot])
            frame.update({name: value[slot, ids].cpu().tolist()
                          for name, value in self.buffers.items()})
            frames.append(frame)
        payload = dict(version='physics_fault_trace_v1', context=context,
                       timing='after_physics_before_consume', capacity=self.capacity,
                       env_indices=ids.cpu().tolist(), joint_names=self.joint_names,
                       frames=frames, replayable_simulator_checkpoint=False)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(_json_finite(payload), allow_nan=False) + '\n')
        temporary.replace(self.path)
