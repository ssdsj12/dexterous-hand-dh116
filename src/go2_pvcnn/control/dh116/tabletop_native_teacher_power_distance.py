"""Power pose error from actual palm links and fixed DeformX material stations."""
from copy import deepcopy
import json

import numpy as np

from .tabletop_deformable_r0_replay import IDENTITY, _digest
from .tabletop_native_teacher_profile import checked_file
from .tabletop_power_shared_rod import PALM_NAMES, PROXIMAL_POWER_MESH_NAMES, SIDES
from .tabletop_training_reward import SourceFailure

VERSION = "dh116-native-power-rod-relative-palm-v1"


def power_pose_error(reference_nodes, target_palms, actual_nodes, actual_palms):
    """Translate the original palm targets with the same two material stations."""
    arrays = tuple(np.asarray(x, dtype=np.float64)
                   for x in (reference_nodes, target_palms, actual_nodes, actual_palms))
    if any(a.shape != shape or not np.isfinite(a).all()
           for a, shape in zip(arrays, ((41, 3), (2, 3), (41, 3), (2, 3)), strict=True)):
        raise ValueError("finite original and measured 41-node and two-palm positions required")
    reference, targets, current, palms = arrays
    stations = np.argmin(np.linalg.norm(targets[:, None] - reference[None], axis=-1), axis=1)
    moving_targets = targets + current[stations] - reference[stations]
    return np.linalg.norm(palms - moving_targets, axis=-1).tolist(), stations.tolist()


class NativePowerPoseApproach:
    """Diagnostic Power goal. Geometry error and contact qualification are separate."""

    def __init__(self, project, manifest, *, scope="owner_source_unqualified"):
        from .tabletop_tripod_reset_manifest import PowerResetManifest

        if type(manifest) is not PowerResetManifest or scope not in ("cpu_fixture", "owner_source_unqualified"):
            raise ValueError("source-derived Power Reset and diagnostic scope required")
        data = manifest.data()
        task = data["task_config"]
        task_config = json.loads(checked_file(task["path"], task["sha256"]))
        reference = project / "configs/dh116_tasks/flexible_power_near_open_settle_v1.json"
        goal_sha = task_config["near_open_source_sha256"]
        goal = json.loads(checked_file(reference, goal_sha))
        self.scope = scope
        self.reference_nodes = np.asarray(data["rod_state"]["positions"], np.float64).reshape(41, 3)
        self.targets = np.asarray([goal["candidate"][side]["target_pose_wxyz"][:3] for side in SIDES])
        self.goal_sha = goal_sha
        self.task_sha = task["sha256"]
        self.mesh_sha = data["mesh_bindings_sha256"]

    def _check_layout(self, rod):
        from .tabletop_native_teacher_profile import digest
        from .tabletop_deformable_r0_replay import encode_owned

        if (tuple(rod["mesh_names"]) != PROXIMAL_POWER_MESH_NAMES
                or digest(encode_owned(rod["mesh_bindings"])) != self.mesh_sha):
            raise SourceFailure("actual Power seventeen-slot mesh identity differs")

    def _metric(self, rod, palms, *, binding):
        self._check_layout(rod)
        identity = {key: rod[key] for key in IDENTITY}
        nodes = np.asarray(rod["node_positions"])
        if nodes.shape != (1, 41, 3) or identity["clock_valid"] is not True:
            raise SourceFailure("actual same-clock 41 material nodes required")
        values, ids = power_pose_error(self.reference_nodes, self.targets, nodes[0], palms)
        source = dict(version=VERSION, goal_sha256=self.goal_sha, task_sha256=self.task_sha,
                      mesh_bindings_sha256=self.mesh_sha, identity=identity, material_ids=ids,
                      input_sha256=_digest(dict(nodes=nodes, palms=palms)), history_binding=binding,
                      quantity="world_palm_translation_error_to_fixed_material_station", units="m")
        source_sha = _digest(source)
        return dict(version=VERSION, source=source, source_sha256=source_sha, source_scope=self.scope,
                    auxiliary=dict(identity=identity, samples=dict(approach_distance_m=dict(
                        value=values, source=VERSION+":"+source_sha))),
                    missing_reason=None, contact_observed=None, native_qualification=False,
                    teacher_admission=False)

    def capture_reset(self, reset, parameter_packet, *, grasp):
        rod = reset["rod_reset"]
        if (grasp != "power" or reset["status"] != "complete" or reset["new_sdk_steps"] != 0
                or reset["new_solver_steps"] != 0 or rod["rod_tick"] != 0
                or parameter_packet["identity"] != reset["after_clock"]):
            raise SourceFailure("actual zero-step Power Reset and same-clock parameters required")
        palms = np.stack([reset["pad_poses_xyzw"][name][0] for name in PALM_NAMES])
        return self._metric(rod, palms, binding="actual_power_reset_pad_pose_readback")

    def capture_geometry(self, row, *, sim, bindings, read_clock):
        from .tabletop_deformable_bimanual_step import _owned

        rod = row["rod"]
        self._check_layout(rod)
        clock = read_clock()
        keys = tuple(clock)
        post = row["action_readback"]["post"]["identity"][0]
        if (any(rod[k] != clock[k] or post.get(k) != clock[k] for k in keys)
                or sim.current_time_step_index != clock["physics_step"]
                or tuple(b.side for b in bindings) != SIDES):
            raise SourceFailure("Power palm readback requires the same completed SDK/rod boundary")
        palms = []
        for binding, name in zip(bindings, PALM_NAMES, strict=True):
            view = binding.hand.root_physx_view
            slot = rod["mesh_bindings"][name]
            index = slot["body_index"]
            pose = _owned(view.get_link_transforms())
            if (pose.shape != (1, 23, 7) or binding.hand.body_names[index] != "base_link"
                    or view.link_paths[0][index] != slot["prim_path"]):
                raise SourceFailure("actual side-specific SDK palm link required")
            palms.append(pose[0, index, :3].copy())
        if read_clock() != clock or sim.current_time_step_index != clock["physics_step"]:
            raise SourceFailure("Power palm readback advanced the owner clock")
        return dict(identity=deepcopy(clock), palms=np.stack(palms), source_complete=True,
                    failure_reasons=[], terminal_reason=None)

    def capture(self, event, *, history_owner, grasp):
        from .tabletop_native_teacher_action_history import NativeAction24HistoryOwner

        row = event["exchange"]
        if grasp != "power" or type(history_owner) is not NativeAction24HistoryOwner:
            raise SourceFailure("independent Power source and actual action history required")
        history_owner._check()
        packet = row["closure_geometry"]
        post = row["action_readback"]["post"]["identity"][0]
        if (history_owner._decision is None or not history_owner._row_hashes
                or history_owner._row_hashes[-1] != _digest(event)
                or any(post.get(k) != v for k, v in packet["identity"].items())
                or not np.array_equal(np.asarray(row["action_readback"]["request"]["action24"]),
                                      history_owner._issued.astype(np.float64))
                or history_owner._target_hashes[-1] != _digest(row["action_readback"]["submission"]["expected_backend_targets"])):
            raise SourceFailure("Power pose evidence differs from original SDK action/history")
        return self._metric(row["rod"], packet["palms"], binding=dict(
            collection_token=history_owner._decision["policy"]["collection_token"],
            physical_row_sha256=history_owner._row_hashes[-1],
            sdk_targets_sha256=history_owner._target_hashes[-1]))
