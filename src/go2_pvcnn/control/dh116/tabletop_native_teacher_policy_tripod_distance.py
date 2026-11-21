"""Independent Tripod sampled-material distance; never contact qualification.

Three real locked pad grids per hand, fixed 41 material nodes, actual SDK
poses, seven COM wrench routes and both supports remain distinct sources.
This is an unqualified source reader; no reset/step/policy is created here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .tabletop_native_teacher_policy_pinch_distance import (
    PolicyOwnedPinchGeometry, _array, _bytes, _require, _radius,
)
from .tabletop_measured_closure_geometry import required_sdf_valid
from .tabletop_power_shared_rod import (
    PROXIMAL_POWER_MESH_NAMES, SIDES, contact_links, _bindings as validate_bindings,
)
from .tabletop_deformable_r0_replay import _digest
from .tabletop_training_reward import SourceFailure

REQUIRED = ("finger14_link", "finger23_link", "finger33_link")
VERSION = "dh116-policy-owned-tripod-distance-v1"
GEOMETRY_VERSION = "dh116-policy-owned-tripod-geometry-v1"


def positive_pad_mean(samples):
    """Every required pad must be observable; missing is not zero distance."""
    if set(samples) != set(REQUIRED) or any(
            not required_sdf_valid(samples[name]) for name in REQUIRED):
        return None
    import math
    return math.fsum(max(0., samples[name]["minimum_gap_m"]) for name in REQUIRED)/3.


class PolicyOwnedTripodGeometry(PolicyOwnedPinchGeometry):
    REQUIRED_PADS = REQUIRED
    ROUTE_COUNT = 7
    VERSION = GEOMETRY_VERSION

    def __init__(self, project, *, scope, task_config, task_config_sha256, mesh_bindings):
        super().__init__(project, scope=scope)
        from .tabletop_power_supported_task import prepare_initial_state, TRIPOD_SCOPE
        path = Path(task_config)
        _require(not path.is_symlink() and path.is_file(), "frozen Tripod task source required")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == task_config_sha256,
                 "frozen Tripod task SHA differs")
        task = json.loads(raw)
        # Validate the existing free-rod/support scene contract. This does not
        # assign its authored hand poses to any measured SDK body.
        from .tabletop_mesh_reset_order import TASK_CONFIG, TASK_SHA
        table_raw = (self.project/TASK_CONFIG).read_bytes()
        _require(hashlib.sha256(table_raw).hexdigest() == TASK_SHA, "original table source differs")
        table_scene = json.loads(table_raw)["scene"]
        prepare_initial_state(task, dict(center_m=table_scene["table_center_m"],
                                        size_m=table_scene["table_size_m"]),
                              expected_scope=TRIPOD_SCOPE)
        self._mesh_bindings = validate_bindings(mesh_bindings)
        _require(contact_links(self._mesh_bindings)[-2:] == ("base_link", "finger52_link"),
                 "complete Tripod seven-route layout required")
        for name, binding in self._mesh_bindings.items():
            side = binding["side"]
            _require(binding.get("frame") == "PhysX_link_origin" and binding.get("unit") == "m"
                     and binding["source_usd_sha256"] == self.sources[side]["source_sha256"]["usd"],
                     "original side-specific contact source frame/unit/USD differs")
        self._support = deepcopy(task["support_scene"])
        self.sources["tripod"] = dict(
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            task_config_sha256=task_config_sha256,
            mesh_bindings_sha256=_digest(self._mesh_bindings),
            required_pads=list(REQUIRED), mesh_names=list(PROXIMAL_POWER_MESH_NAMES),
            quantity="fixed_material41_sampled_node_distance_not_contact")
        dependencies = (
            "scripts/verify_dh116_power_shared_rod.py",
            "scripts/audit_dh116_certified_segment_distance.py",
            "scripts/screen_dh116_tripod_supported_reset.py",
            "scripts/certify_dh116_tripod_distal_route.py",
            "src/go2_pvcnn/control/dh116/tabletop_power_supported_task.py",
            "src/go2_pvcnn/control/dh116/tabletop_power_shared_rod.py",
            "src/go2_pvcnn/control/dh116/tabletop_batch_geometry.py",
        )
        self.sources["tripod"]["dependencies"] = {
            name: hashlib.sha256((self.project/name).read_bytes()).hexdigest()
            for name in dependencies}

    def _routes(self, rod):
        _require(tuple(rod["mesh_names"]) == PROXIMAL_POWER_MESH_NAMES
                 and _bytes(rod["mesh_bindings"]) == _bytes(self._mesh_bindings),
                 "complete actual Tripod mesh layout/bindings differ")
        routes = rod["hand_wrenches_com_world"]
        _require(set(routes) == set(SIDES), "both actual hand wrench sources required")
        links = contact_links(self._mesh_bindings)
        for side in SIDES:
            route = routes[side]
            _require(route.get("is_global") is True
                     and route["body_ids"] == [self._mesh_bindings[side+"."+n]["body_index"] for n in links],
                     "actual seven-route COM/world/body identity differs")
            for field in ("forces", "torques"):
                _array(route[field], (1, 7, 3))
        support = rod["external_support_wrenches_world"]
        _require(set(support) == {"table", "support_0", "support_1"},
                 "table and both external support wrench sources required")
        named = rod["named_wrenches"]
        _require(set(named) == set(PROXIMAL_POWER_MESH_NAMES), "all17 named mesh wrenches required")
        for name, wrench in named.items():
            _array(wrench, (6,))
        for name, wrench in support.items():
            _require(np.array_equal(_array(wrench, (6,)), np.asarray(named[name])),
                     "same-step external support totals differ")
        pairs = rod["rod_mesh_contact_pairs"]
        _require(rod["pair_capture_status"] == "validated_solver_samples"
                 and pairs["physics_step"] == rod["physics_step"]
                 and pairs["rod_tick"] == rod["rod_tick"],
                 "same-clock complete pair capture required")
        buffer = pairs["solver_buffer"]
        _require(all(type(buffer[key]) is int for key in ("substeps", "mesh_count", "capacity", "attempted"))
                 and 0 <= buffer["attempted"] <= buffer["capacity"]
                 and buffer["substeps"] == 12 and buffer["mesh_count"] == 17
                 and buffer["capacity"] == 41*17*12 and buffer["overflow"] is False
                 and buffer["attempted"] == len(pairs["pairs"]),
                 "complete bounded seventeen-slot pair allocation required")

    def _evaluate(self, raw, exchange):
        self._routes(exchange["rod"])
        packet = super()._evaluate(raw, exchange)
        from scripts.verify_dh116_power_shared_rod import check_supported_shape
        from .tabletop_batch_geometry import rotated_points
        from scripts.screen_dh116_tripod_supported_reset import archived_tube
        from scripts.audit_dh116_certified_segment_distance import ENVELOPE_REFINEMENT_VERSION
        nodes = _array(raw["rod_nodes_m"], (41, 3))
        tube = archived_tube(nodes[None])
        boxes = [(box["name"], np.asarray(box["center_m"])-np.asarray(box["size_m"])/2,
                  np.asarray(box["center_m"])+np.asarray(box["size_m"])/2)
                 for box in self._support["supports"]]
        failures = {}
        for side in SIDES:
            hand = raw["hands"][side]
            poses = hand["link_poses_xyzw"][0]
            for kind, shapes in self._geometry[side].items():
                for name, vertices in shapes.items():
                    pose = poses[hand["body_names"].index(name)]
                    # Preserve the existing native source/cooked guard's raw
                    # quaternion arithmetic; do not renormalize away a boundary.
                    world = rotated_points(np.r_[pose[:3], pose[6], pose[3:6]], vertices)
                    target = name in REQUIRED
                    result = check_supported_shape(world, boxes, tube, contact_shape=target,
                        contact_penetration_m=.00015 if target else None,
                        reference_nodes=nodes[None],
                        envelope_refinement=ENVELOPE_REFINEMENT_VERSION if target else None)
                    if result is not None:
                        failures[side+"."+kind+":"+name] = result
        packet["supported_geometry_failures"] = failures
        if failures:
            packet["failure_reasons"].append("tripod_supported_source_or_cooked_geometry_guard")
            packet["passed"] = False
            packet["terminal_reason"] = "tripod_supported_geometry_guard"
        return packet


class PolicyOwnedTripodApproach:
    VERSION = VERSION

    def __init__(self, project, *, scope="owner_source_unqualified", **sources):
        self.geometry = PolicyOwnedTripodGeometry(project, scope=scope, **sources)
        self.scope = scope

    def _metric(self, packet, radius):
        identity = packet["raw"]["identity"]
        nodes = packet["raw"]["rod_nodes_m"]
        per_hand = {}
        for side in SIDES:
            hand = packet["raw"]["hands"][side]
            poses = hand["link_poses_xyzw"][0]
            samples = {}
            for name in REQUIRED:
                pose = poses[hand["body_names"].index(name)]
                matrix = self.geometry._pose_matrix(np.r_[pose[:3], pose[6], pose[3:6]])
                samples[name] = self.geometry._sdf(
                    self.geometry._grids[side][name], matrix, nodes, radius=radius["value"])
            per_hand[side] = dict(required_pads=list(REQUIRED), pad_samples=samples,
                                  positive_gap_mean_m=positive_pad_mean(samples))
        distances = [per_hand[side]["positive_gap_mean_m"] for side in SIDES]
        value = None if any(d is None for d in distances) else distances
        from .tabletop_deformable_r0_replay import IDENTITY
        source = dict(version=VERSION, geometry_raw_sha256=packet["raw_sha256"],
            geometry_sources=deepcopy(packet["sources"]), radius_provenance=radius,
            units="m", coordinate_frame="world_to_actual_pad_local",
            quantity="discrete_solver_node_to_three_locked_pad_SDF_positive_gap",
            uncertainty_m=None, contact_qualified=False, hardware_measured=False,
            timestamp_s=identity["timestamp_s"], target_material_ids=list(range(41)))
        digest = _digest(source)
        return dict(version=VERSION, source_scope=self.scope, per_hand=per_hand,
            auxiliary=dict(identity={k:identity[k] for k in IDENTITY},
                samples=dict(approach_distance_m=dict(value=value,
                    source=None if value is None else "actual_tripod_SDK_material41_locked_SDF:"+digest))),
            source=source, source_sha256=digest,
            missing_reason="required_tripod_pad_lattice_coverage_unavailable" if value is None else None,
            reset_baseline_available=False, contact_observed=None,
            contact_inferred_from_geometry=False, stability_measured=False,
            new_physics_steps=0, ppo_updates=0, native_qualification=False, teacher_admission=False)

    def capture(self, event, *, history_owner, grasp="tripod"):
        try:
            _require(grasp == "tripod", "independent Tripod source cannot be relabelled")
            from .tabletop_action24_history_owner import Action24HistoryOwner
            from .tabletop_native_teacher_action_history import NativeAction24HistoryOwner
            _require(type(history_owner) in (Action24HistoryOwner, NativeAction24HistoryOwner),
                     "actual policy history owner required")
            history_owner._check()
            row = event["exchange"]
            _require(history_owner._decision is not None and history_owner._row_hashes
                     and history_owner._row_hashes[-1] == _digest(event),
                     "actual durable row/history hash differs")
            decision = history_owner._decision
            _require(np.array_equal(np.asarray(row["action_readback"]["request"]["action24"]),
                                    history_owner._issued.astype(np.float64))
                     and history_owner._target_hashes[-1] == _digest(
                         row["action_readback"]["submission"]["expected_backend_targets"]),
                     "actual issued policy request/SDK target history differs")
            endpoint = row["action_readback"]["post"]["identity"][0]
            _require(all(history_owner._sequence.clock[k] == endpoint[k]
                         for k in history_owner._sequence.clock), "same history endpoint clock required")
            packet = self.geometry.validate(row["closure_geometry"], row)
            _require(packet["source_complete"], "required actual Tripod SDF distance unavailable: "+str(packet["source_errors"]))
            result = self._metric(packet, _radius(row))
            binding = dict(collection_token=decision["policy"]["collection_token"],
                policy=deepcopy(decision["policy"]), physical_row_sha256=history_owner._row_hashes[-1],
                sdk_targets_sha256=history_owner._target_hashes[-1],
                history_owner_uid=history_owner._owner_uid, physical_samples=len(history_owner._row_hashes))
            result["history_binding"] = binding
            result["source"].update(geometry_contract_version=GEOMETRY_VERSION, history_binding=binding)
            result["source_sha256"] = _digest(result["source"])
            sample = result["auxiliary"]["samples"]["approach_distance_m"]
            if sample["value"] is not None:
                sample["source"] = "actual_tripod_SDK_material41_locked_SDF:"+result["source_sha256"]
            return result
        except SourceFailure:
            raise
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise SourceFailure("policy-owned Tripod source: "+str(error)) from error
